HIPCC = hipcc
# gfx942 = MI300X (CDNA 3, wavefront-64)
# gfx1100 = RX 7900 XTX (RDNA 3, wavefront-32)
# gfx1201 = RX 9070 XT (RDNA 4, wavefront-32)
GPU_TARGETS ?= gfx942
CFLAGS = -O2 --offload-arch=$(GPU_TARGETS)
LDFLAGS = -lpthread

TARGET = infer

tokenizer_impl.o: tokenizer_impl.c ../metal_infer/tokenizer.h
	gcc -O2 -c tokenizer_impl.c -o tokenizer_impl.o

$(TARGET): infer.hip kernels.hip.h ../metal_infer/tokenizer.h tokenizer_impl.o
	$(HIPCC) $(CFLAGS) -o $(TARGET) infer.hip tokenizer_impl.o $(LDFLAGS)

clean:
	rm -f $(TARGET) tokenizer_impl.o

.PHONY: clean
