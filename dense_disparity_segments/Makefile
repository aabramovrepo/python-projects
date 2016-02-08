# Alexey Abramov <abramov@physik3.gwdg.de>
# Makefile with CUDA 3.0

# $HOME $QTDIR $OPENCV $QT_MOC have to be defined before the compilation

CUDA_SDK:=$(HOME)/NVIDIA_GPU_Computing_SDK/C
CUDA_BASE:=/usr/local/cuda

CC:=g++

INCLUDE:=-I/usr/include -I$(CUDA_BASE)/include -I$(CUDA_SDK)/common/inc -I/home/abramov/workspace/oculus/Libraries/CudaSegmentation -I/home/abramov/workspace/oculus/Libraries/opticalflow_stereo/CudaDenseVision/inc -I/home/abramov/workspace/oculus/Libraries/DenseDisparity -I/usr/local/include/opencv2

FLAGS:=-O3 -MMD -Wall -c -g
LIBS:=-L/usr/local/lib -L$(CUDA_SDK)/lib -L$(CUDA_BASE)/lib64 -L/home/abramov/workspace/oculus/build/lib -lCudaSegmentation -lDenseDisparity -lCudaDenseVision -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_calib3d -lopencv_video -lopencv_gpu -lcudart -lcutil_x86_64

HEADERS:=$(wildcard *.h)
OBJECTS:=$(patsubst %.cpp,%.o,$(wildcard *.cpp))
TARGET:=denseStereo

all: $(OBJECTS)
	$(CC) $(OBJECTS) $(LIBS) -o $(TARGET)

%.o: %.cpp
	$(CC) $(INCLUDE) $(FLAGS) $< -o $@

%.o: %.c
	gcc $(INCLUDE) $(FLAGS) $< -o $@

include $(wildcard *.d)

clean:
	rm -f $(OBJECTS) *.d $(TARGET)
