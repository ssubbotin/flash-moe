#!/bin/bash
# Build flash-moe CUDA engine for Qwen3.5-35B-A3B
#
# Architecture: 35B total, 3B active
#   hidden=2048, layers=40, heads=16, kv_heads=2, head_dim=256
#   experts=256, active=8, moe_intermediate=512
#   linear: v_heads=32, k_heads=16, key_dim=128, value_dim=128

NVCC=/usr/local/cuda-13.1/bin/nvcc

DEFINES="-DHIDDEN_DIM=2048 \
  -DNUM_LAYERS=40 \
  -DNUM_ATTN_HEADS=16 \
  -DNUM_KV_HEADS=2 \
  -DHEAD_DIM=256 \
  -DNUM_EXPERTS=256 \
  -DMOE_INTERMEDIATE=512 \
  -DSHARED_INTERMEDIATE=512 \
  -DFULL_ATTN_INTERVAL=4 \
  -DLINEAR_NUM_V_HEADS=32 \
  -DLINEAR_NUM_K_HEADS=16 \
  -DLINEAR_KEY_DIM=128 \
  -DLINEAR_VALUE_DIM=128 \
  -DCONV_KERNEL_SIZE=4 \
  -DMAX_K=8"

$NVCC -O2 -arch=sm_120 $DEFINES \
  -o infer_35b infer.cu tokenizer_impl.o \
  -lpthread -L/usr/local/cuda-13.1/targets/x86_64-linux/lib \
  -lcufile -lcublas -lcublasLt \
  && echo "BUILD OK: infer_35b" \
  || echo "BUILD FAILED"
