# Makefile for compiling the C++ code to produce python extension module of Quda
# Borrowed from https://stackoverflow.com/questions/8025766/makefile-auto-dependency-generation
CXX = mpic++
CXXFLAGS = -std=c++17 -DQMP_COMMS -DHAVE_QIO -DMULTI_GPU
CPPFLAGS = -fPIC -MD -MP -O3 -Wall -shared
QUDA_DIR = /data/d10b/users/ylin/quda/build
CUDA_DIR = /opt/software/cuda-11.2.0

# I don't know why you need to link it, or qmp undefined symbol error
#USQCD_DIR = /home/ylin/data/quda/build/usqcd

export OMPI_CXX=g++
export OMPI_CC=gcc
 
# Automatically finding include paths for pybind and mpi4py from the current python3
QUDA_INCLUDE_PATH := -I$(QUDA_DIR)/include -I$(QUDA_DIR)/usqcd/include
CUDA_INCLUDE_PATH := -I$(CUDA_DIR)/include
PYBIND_INCLUDE_PATH := $(shell python3 -m pybind11 --includes)
MPI4PY_INCLUDE_PATH := -I$(shell python -c "import mpi4py; print(mpi4py.get_include())")
CXX_INCLUDE_PATH = $(PYBIND_INCLUDE_PATH) $(MPI4PY_INCLUDE_PATH) $(QUDA_INCLUDE_PATH) $(CUDA_INCLUDE_PATH)

LDFLAGS += -L$(QUDA_DIR)/lib -L$(QUDA_DIR)/usqcd/lib
LDLIBS += -lquda -lqio -lqmp -llime
CXXFLAGS += $(CXX_INCLUDE_PATH)

PYTHON_MODUDLE_SUFFIX := $(shell python3-config --extension-suffix)# without the suffix, python can't import it

SRC = $(wildcard *.cpp)

all: quda

quda: $(SRC:%.cpp=%.o)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@$(PYTHON_MODUDLE_SUFFIX) $^ $(LDFLAGS) $(LDLIBS) 

# This will add targets for all dependencies that will be built automatically
-include $(SRC:%.cpp=%.d)

.PHONY: clean
clean:
	rm -f *.o quda$(PYTHON_MODUDLE_SUFFIX)