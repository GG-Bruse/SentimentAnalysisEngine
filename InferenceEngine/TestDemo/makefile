CC=g++

LINKER=g++

INCLUDES= -Iinclude -Ithirdparty -I/usr/local/cuda/include -I/data3/a100_project/TensorRT-8.5.2.2/include/

CPPFLAGS= $(INCLUDES) -O2 -g -pipe -W -Wall -std=c++17 -rdynamic -fPIC -m64 -msse -msse2 -pthread

CPPFLAGS+= -Wno-sign-compare -Wno-unused-local-typedefs -Wno-unused-parameter \
			-Wno-deprecated-declarations -Winit-self

LDFLAGS= -Llib -lmkl_rt -liniparser -lnvinfer_plugin -lnvinfer -lemotion_classification_engine -lutf8proc -licuuc -licuio -licudata -licui18n -ltvm_runtime -ldnnl -L/usr/local/cuda/lib64 -lcuda -lcudart

HEADERS= $(wildcard ./include/*.h)
HEADERS= $(wildcard /usr/local/cuda/include/*.h)
HEADERS= $(wildcard /data3/a100_project/TensorRT-8.5.2.2/include/*.h)

SOURCE= $(wildcard ./src/*.cxx)
OBJS= $(patsubst %.cxx,%.o, $(SOURCE))
TARGET=test_demo

all: $(OBJS)
	$(LINKER) -o $(TARGET) $^ $(LDFLAGS)

%.o: %.cxx ${HEADERS}
	$(CC) $(CPPFLAGS) $< -c -o $@

.PNONY:clean
clean:
	rm $(OBJS) *.o -f
	rm $(TARGET) -f