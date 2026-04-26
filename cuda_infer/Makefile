CUDA_HOME ?= $(firstword $(wildcard /usr/local/cuda-12.8 /usr/local/cuda))
NVCC = $(CUDA_HOME)/bin/nvcc
CFLAGS = -O2 -Wno-deprecated-gpu-targets
CUDA_LIB = $(CUDA_HOME)/targets/x86_64-linux/lib
CUDA_INC = $(CUDA_HOME)/targets/x86_64-linux/include
LDFLAGS = -lpthread -L$(CUDA_LIB) -lcufile -lcublas -lcublasLt

KIMI_HEADERS = kimi_loader.cuh kimi_forward.cuh kimi_moe.cuh mla_forward.cuh kernels.cuh
KIMI_LDFLAGS = -lpthread -L$(CUDA_LIB) -lcublas -lcublasLt
KIMI_CFLAGS  = $(CFLAGS) -Xcompiler -fopenmp -lgomp

TARGET = infer

tokenizer_impl.o: tokenizer_impl.c ../metal_infer/tokenizer.h
	gcc -O2 -c tokenizer_impl.c -o tokenizer_impl.o

$(TARGET): infer.cu kernels.cuh ../metal_infer/tokenizer.h tokenizer_impl.o
	$(NVCC) $(CFLAGS) -o $(TARGET) infer.cu tokenizer_impl.o $(LDFLAGS)

infer_kimi: infer_kimi.cu $(KIMI_HEADERS)
	$(NVCC) $(KIMI_CFLAGS) -o infer_kimi infer_kimi.cu $(KIMI_LDFLAGS)

test_fp8_block: test_fp8_block.cu qwen36_kernels.cuh safetensors_io.cuh
	$(NVCC) $(CFLAGS) -o test_fp8_block test_fp8_block.cu -lpthread

test_qwen36_linear_attn: test_qwen36_linear_attn.cu qwen36_kernels.cuh safetensors_io.cuh kernels.cuh
	$(NVCC) $(CFLAGS) -arch=sm_89 -o test_qwen36_linear_attn test_qwen36_linear_attn.cu -lpthread

test_qwen36_moe: test_qwen36_moe.cu qwen36_kernels.cuh safetensors_io.cuh kernels.cuh
	$(NVCC) $(CFLAGS) -arch=sm_89 -o test_qwen36_moe test_qwen36_moe.cu -lpthread

clean:
	rm -f $(TARGET) infer_kimi test_fp8_block test_qwen36_linear_attn test_qwen36_moe

.PHONY: clean
