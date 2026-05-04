HIPCC = hipcc
# gfx942 = MI300X (CDNA 3, wavefront-64)
# gfx1100 = RX 7900 XTX (RDNA 3, wavefront-32)
# gfx1201 = RX 9070 XT (RDNA 4, wavefront-32)
GPU_TARGETS ?= gfx942

# Wavefront size must be explicit: __gfx942__ / __gfx1100__ macros are only
# defined in the device pass, but kernel launch dims (dim3 block(WARP_SIZE,..))
# are evaluated in host code, so host and device must agree at compile time.
ifeq ($(GPU_TARGETS),gfx942)
  WARP_SIZE ?= 64
else ifeq ($(GPU_TARGETS),gfx940)
  WARP_SIZE ?= 64
else ifeq ($(GPU_TARGETS),gfx941)
  WARP_SIZE ?= 64
else
  WARP_SIZE ?= 32
endif

# aotriton (optional, for Flash Attention on full-attention layers).
# Set AOTRITON_PREFIX=/opt/aotriton or similar if not in a standard path.
AOTRITON_PREFIX ?= /opt/aotriton
AOTRITON_CFLAGS ?= -I$(AOTRITON_PREFIX)/include
AOTRITON_LIB ?= -L$(AOTRITON_PREFIX)/lib -laotriton_v2 -Wl,-rpath,$(AOTRITON_PREFIX)/lib

CFLAGS = -O2 --offload-arch=$(GPU_TARGETS) -DWARP_SIZE=$(WARP_SIZE) $(AOTRITON_CFLAGS)
LDFLAGS = -lpthread $(AOTRITON_LIB)

TARGET = infer

tokenizer_impl.o: tokenizer_impl.c ../metal_infer/tokenizer.h
	gcc -O2 -c tokenizer_impl.c -o tokenizer_impl.o

infer.o: infer.hip kernels.hip.h kernels_aotriton_attn.hip.h kernels_fla_gdn.hip.h kernels_fused_moe.hip.h ../metal_infer/tokenizer.h
	$(HIPCC) $(CFLAGS) -c infer.hip -o infer.o

$(TARGET): infer.o tokenizer_impl.o
	$(HIPCC) $(CFLAGS) -o $(TARGET) infer.o tokenizer_impl.o $(LDFLAGS)

clean:
	rm -f $(TARGET) infer.o tokenizer_impl.o

.PHONY: clean
