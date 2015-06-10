CFLAGS=-lOpenCL
OBJS=device_info run_cpu run_gpu

# Debug/Profiling enable
ifdef DEBUG
CFLAGS+=-DDEBUG_MODE
endif
ifdef PROF
CFLAGS+=-DPROF_MODE
endif

# rules
all: $(OBJS)
clean:
	rm -f $(OBJS)

device_info: device_info.c
	gcc -o $@ $^ $(CFLAGS)

run_cpu: cpu_reference.c
	gcc -o $@ $^ $(CFLAGS)

run_gpu: gpu_accelerated.c
	gcc -o $@ $^ $(CFLAGS)
