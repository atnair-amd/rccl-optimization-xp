ROCM    ?= /opt/rocm
MPI_INC ?= /usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB ?= /usr/lib/x86_64-linux-gnu/openmpi/lib

CXX      = $(ROCM)/bin/hipcc
CXXFLAGS = -O2 -std=c++17 -I$(ROCM)/include -I$(MPI_INC)
LDFLAGS  = -L$(ROCM)/lib -L$(MPI_LIB) -lrccl -lmpi

SRCS = src/main.cpp \
       src/fake_grads.cpp \
       src/baseline.cpp \
       src/bucketing.cpp \
       src/fusion.cpp

HDRS = src/fake_grads.h \
       src/baseline.h \
       src/bucketing.h \
       src/fusion.h

TARGET = bench

$(TARGET): $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: clean
