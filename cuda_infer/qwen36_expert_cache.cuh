/*
 * qwen36_expert_cache.cuh — LRU cache for routed expert weights resident in
 * VRAM, backed by a packed_experts/layer_N.bin disk pool on miss.
 *
 * Each expert is a fixed 3,146,112-byte block (the layout written by
 * repack_experts_qwen36.py): gate_w | gate_s | up_w | up_s | down_w | down_s.
 *
 *   key   : (layer_idx * 256 + expert_idx)
 *   value : device pointer to the expert's 3.1 MB block, sub-pointers at
 *           fixed offsets into it.
 *
 * Pool: one giant cudaMalloc of capacity × EXPERT_BYTES at startup; cache
 * slots are integer indices into the pool (no per-expert cudaMalloc/Free
 * churn — significant when active_set_size > capacity).
 *
 * Eviction: simple list-based LRU.  O(1) hit, O(1) miss-with-eviction.
 */
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <string>
#include <unordered_map>
#include <vector>
#include <list>
#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifndef CUDA_OK
#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)
#endif

namespace q36 {

class ExpertCache {
public:
    static constexpr size_t GW_OFF = 0;
    static constexpr size_t GW_LEN = 512 * 2048;
    static constexpr size_t GS_OFF = GW_OFF + GW_LEN;
    static constexpr size_t GS_LEN = 4 * 16 * 2;
    static constexpr size_t UW_OFF = GS_OFF + GS_LEN;
    static constexpr size_t UW_LEN = 512 * 2048;
    static constexpr size_t US_OFF = UW_OFF + UW_LEN;
    static constexpr size_t US_LEN = 4 * 16 * 2;
    static constexpr size_t DW_OFF = US_OFF + US_LEN;
    static constexpr size_t DW_LEN = 2048 * 512;
    static constexpr size_t DS_OFF = DW_OFF + DW_LEN;
    static constexpr size_t DS_LEN = 16 * 4 * 2;
    static constexpr size_t EXPERT_BYTES = DS_OFF + DS_LEN;     // 3,146,112

    ExpertCache() = default;

    // capacity = max number of (layer, expert) entries kept in VRAM.
    // packed_dir = directory holding layer_N.bin files.
    // num_layers = used to open one fd per layer.
    bool init(uint32_t capacity, const std::string& packed_dir, uint32_t num_layers) {
        capacity_ = capacity;
        packed_dir_ = packed_dir;
        // open one file per layer up front (reused on every miss)
        layer_fds_.assign(num_layers, -1);
        for (uint32_t li = 0; li < num_layers; li++) {
            char p[1024]; snprintf(p, sizeof(p), "%s/layer_%u.bin", packed_dir.c_str(), li);
            int fd = ::open(p, O_RDONLY);
            if (fd < 0) {
                fprintf(stderr, "ExpertCache: open %s failed: %s\n", p, std::strerror(errno));
                return false;
            }
            layer_fds_[li] = fd;
        }
        // allocate the VRAM pool: capacity * 3.1 MB
        size_t pool_bytes = (size_t)capacity_ * EXPERT_BYTES;
        if (cudaMalloc(&pool_, pool_bytes) != cudaSuccess) {
            fprintf(stderr, "ExpertCache: cudaMalloc(%.2f GB) failed; reduce --cache-experts\n",
                    (double)pool_bytes / 1e9);
            return false;
        }
        free_slots_.reserve(capacity_);
        for (uint32_t i = 0; i < capacity_; i++) free_slots_.push_back(i);
        host_block_.resize(EXPERT_BYTES);
        fprintf(stderr, "ExpertCache: %u slots × %.1f MB = %.2f GB pool, %u layer fds\n",
                capacity_, (double)EXPERT_BYTES / 1e6,
                (double)pool_bytes / 1e9, num_layers);
        return true;
    }

    void close_() {
        for (int fd : layer_fds_) if (fd >= 0) ::close(fd);
        layer_fds_.clear();
        if (pool_) { cudaFree(pool_); pool_ = nullptr; }
        slot_for_.clear(); lru_.clear(); slot_iter_.clear(); free_slots_.clear();
    }

    // Return device pointer to the expert's 3 MB block. Loads from disk on
    // miss, evicts LRU if cache is full.
    uint8_t* get(uint32_t layer_idx, uint32_t expert_idx) {
        uint32_t key = layer_idx * 256u + expert_idx;
        auto it = slot_for_.find(key);
        if (it != slot_for_.end()) {
            // hit: bump to front of LRU
            uint32_t slot = it->second;
            lru_.splice(lru_.begin(), lru_, slot_iter_[slot]);
            slot_iter_[slot] = lru_.begin();
            hits_++;
            return slot_ptr(slot);
        }
        // miss: pick a slot
        uint32_t slot;
        if (!free_slots_.empty()) {
            slot = free_slots_.back(); free_slots_.pop_back();
            slot_iter_.resize(std::max((size_t)slot + 1, slot_iter_.size()));
        } else {
            // evict the LRU tail
            uint32_t evict_key = lru_.back();
            uint32_t evict_slot = slot_for_[evict_key];
            slot_for_.erase(evict_key);
            lru_.pop_back();
            slot = evict_slot;
            evictions_++;
        }
        // load from disk + memcpy
        off_t off = (off_t)expert_idx * (off_t)EXPERT_BYTES;
        int fd = layer_fds_[layer_idx];
        size_t total = 0;
        while (total < EXPERT_BYTES) {
            ssize_t got = ::pread(fd, host_block_.data() + total,
                                  EXPERT_BYTES - total, off + (off_t)total);
            if (got <= 0) {
                fprintf(stderr, "ExpertCache: pread l%u e%u: %s\n",
                        layer_idx, expert_idx, std::strerror(errno));
                std::exit(1);
            }
            total += (size_t)got;
        }
        CUDA_OK(cudaMemcpy(slot_ptr(slot), host_block_.data(), EXPERT_BYTES,
                           cudaMemcpyHostToDevice));
        slot_for_[key] = slot;
        lru_.push_front(key);
        slot_iter_[slot] = lru_.begin();
        misses_++;
        return slot_ptr(slot);
    }

    void unpack(uint8_t* d_block,
                const uint8_t** gw, const __nv_bfloat16** gs,
                const uint8_t** uw, const __nv_bfloat16** us,
                const uint8_t** dw, const __nv_bfloat16** ds) const
    {
        *gw = d_block + GW_OFF; *gs = (const __nv_bfloat16*)(d_block + GS_OFF);
        *uw = d_block + UW_OFF; *us = (const __nv_bfloat16*)(d_block + US_OFF);
        *dw = d_block + DW_OFF; *ds = (const __nv_bfloat16*)(d_block + DS_OFF);
    }

    void print_stats(const char* tag) const {
        size_t total = hits_ + misses_;
        if (total == 0) { fprintf(stderr, "[cache %s] no requests\n", tag); return; }
        fprintf(stderr,
                "[cache %s] hits=%zu misses=%zu evictions=%zu hit_rate=%.1f%% used=%zu/%u\n",
                tag, hits_, misses_, evictions_,
                100.0 * (double)hits_ / (double)total,
                slot_for_.size(), capacity_);
    }

    void reset_stats() { hits_ = misses_ = evictions_ = 0; }

private:
    uint8_t* slot_ptr(uint32_t slot) const { return pool_ + (size_t)slot * EXPERT_BYTES; }

    uint32_t capacity_ = 0;
    std::string packed_dir_;
    std::vector<int> layer_fds_;
    uint8_t* pool_ = nullptr;
    std::vector<uint8_t> host_block_;
    // (layer, expert) → slot index
    std::unordered_map<uint32_t, uint32_t> slot_for_;
    // LRU: front = most recent, back = oldest. Holds the (layer*256+expert) keys.
    std::list<uint32_t> lru_;
    // For O(1) splice on hit: per-slot iterator into lru_
    std::vector<std::list<uint32_t>::iterator> slot_iter_;
    std::vector<uint32_t> free_slots_;
    size_t hits_ = 0, misses_ = 0, evictions_ = 0;
};

} // namespace q36
