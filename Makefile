CC = /usr/bin/g++

LD_FLAGS = -lrt

CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_30,code=sm_30 \
        -gencode arch=compute_35,code=sm_35 \
        -gencode arch=compute_50,code=sm_50 \
        -gencode arch=compute_52,code=sm_52 \
        -gencode arch=compute_60,code=sm_60 \
        -gencode arch=compute_61,code=sm_61 \
        -gencode arch=compute_61,code=compute_61
        
# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifeq ($(shell uname),Darwin)
	LDFLAGS       := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lcufft -lcublas
	CCFLAGS   	  := -arch $(OS_ARCH)
else
	ifeq ($(OS_SIZE),32)
		LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart -lculbas
		CCFLAGS   := -m32
	else
		CUDA_LIB_PATH := $(CUDA_LIB_PATH)64
		LDFLAGS       := -L$(CUDA_LIB_PATH) -lcudart -lcublas 
		CCFLAGS       := -m64
	endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
	NVCCFLAGS := -m32
else
	NVCCFLAGS := -m64
endif

TARGETS = Test1_Regression Test2_Regression

all: $(TARGETS)

Test1_Regression: main_multi_regression.cpp multi_regression.o
	$(CC) $^ -o $@ -O3 $(LDFLAGS) -D TEST=1 -Wall -I$(CUDA_INC_PATH) 

Test2_Regression: main_multi_regression.cpp multi_regression.o
	$(CC) $^ -o $@ -O3 $(LDFLAGS) -D TEST=2 -Wall -I$(CUDA_INC_PATH) 

multi_regression.o: multi_regression.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
