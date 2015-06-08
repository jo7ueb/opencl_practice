CFLAGS=-lOpenCL
OBJS=device_info run_cpu run_gpu

all: $(OBJS)
clean:
	rm -f $(OBJS)

device_info: device_info.c
	gcc -o $@ $^ $(CFLAGS)

run_cpu: cpu_reference.c
	gcc -o $@ $^ $(CFLAGS)

run_gpu: gpu_accelerated.c
	gcc -o $@ $^ $(CFLAGS)
