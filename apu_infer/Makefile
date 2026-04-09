HIPCC = hipcc
# gfx1151 = AMD Ryzen AI MAX+ 395 (Strix Halo, RDNA 3.5, wavefront-32, unified memory APU)
# gfx1150 = Strix Point (older Ryzen AI, also unified memory)
# gfx1100 = RX 7900 XTX (RDNA 3 discrete — works but use rocm_infer for discrete cards)
GPU_TARGETS ?= gfx1151

# Wavefront size on RDNA (gfx10/gfx11) is 32; on CDNA (gfx9, MI200/MI300) it's 64.
# Has to be passed via -D so host code (kernel launch dims) and device code agree.
WARP_SIZE ?= 32

CFLAGS = -O2 --offload-arch=$(GPU_TARGETS) -DWARP_SIZE=$(WARP_SIZE)
LDFLAGS = -lpthread

# -fPIC for tokenizer_impl.o is required when linking through hipcc/lld on
# distros that ship a PIE-default libstdc++ (Fedora 43+, recent Ubuntu).
TARGET = infer

tokenizer_impl.o: tokenizer_impl.c ../metal_infer/tokenizer.h
	gcc -O2 -fPIC -c tokenizer_impl.c -o tokenizer_impl.o

infer.o: infer.hip kernels.hip.h ../metal_infer/tokenizer.h
	$(HIPCC) $(CFLAGS) -c infer.hip -o infer.o

$(TARGET): infer.o tokenizer_impl.o
	$(HIPCC) $(CFLAGS) -o $(TARGET) infer.o tokenizer_impl.o $(LDFLAGS)

clean:
	rm -f $(TARGET) infer.o tokenizer_impl.o

.PHONY: clean
