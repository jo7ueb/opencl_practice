#include <stdio.h>
#include <CL/cl.h>

#define N 1024

main() {
    cl_uint num_platforms;
    cl_platform_id *platform;
    cl_uint idx_platform;

    clGetPlatformIDs(0, NULL, &num_platforms);
    platform = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platform, NULL);

    printf("OpenCL # of platform: %d\n", num_platforms);
    for (idx_platform=0; idx_platform < num_platforms; ++idx_platform) {
        cl_device_id *device;
        cl_uint num_devices;
        cl_uint idx_device;
        // platform/device info inquiry
        cl_uint p_uint;
        cl_ulong p_ulong;
        cl_device_type p_type;
        char buf[N];
        size_t p_size[3];

        printf("[Platform %d]\n", idx_platform);

        clGetDeviceIDs(platform[idx_platform], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        device = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
        clGetDeviceIDs(platform[idx_platform], CL_DEVICE_TYPE_ALL, num_devices, device, NULL);

        // platform info
        printf("# of devices: %d\n", num_devices);
        clGetPlatformInfo(platform[idx_platform], CL_PLATFORM_NAME,
                          N, buf, NULL);
        printf("PLATFORM_NAME: %s\n", buf);
        clGetPlatformInfo(platform[idx_platform], CL_PLATFORM_VENDOR,
                          N, buf, NULL);
        printf("PLATFORM_VENDOR: %s\n", buf);
        printf("\n");
        for (idx_device=0; idx_device < num_devices; ++idx_device) {
            printf("[Device #%d]\n", idx_device);

            // device info
            printf("Device info\n");
            clGetDeviceInfo(device[idx_device], CL_DEVICE_VENDOR,
                            N, buf, NULL);
            printf("\tDEVICE_VENDOR: %s\n", buf);
            clGetDeviceInfo(device[idx_device], CL_DEVICE_NAME, 
                            N, buf, NULL);
            printf("\tDEVICE_NAME: %s\n", buf);
            clGetDeviceInfo(device[idx_device], CL_DEVICE_TYPE,
                            sizeof(cl_device_type), &p_type, NULL);
            printf("\tDEVICE_TYPE: %d\n", (int)p_type);
            clGetDeviceInfo(device[idx_device], CL_DEVICE_MAX_CLOCK_FREQUENCY,
                            sizeof(cl_uint), &p_uint, NULL);
            printf("\tDEVICE_MAX_CLOCK_FREQUENCY: %d\n", (int)p_uint);


            // memory size
            printf("Memory info\n");
            clGetDeviceInfo(device[idx_device], CL_DEVICE_GLOBAL_MEM_SIZE,
                            sizeof(cl_ulong), &p_ulong, NULL);
            printf("\tGLOBAL_MEM_SIZE: %lu (%lu MB)\n", p_ulong, p_ulong/(1024*1024));
            clGetDeviceInfo(device[idx_device], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                            sizeof(cl_ulong), &p_ulong, NULL);
            printf("\tGLOBAL_MEM_CACHE_SIZE: %lu (%lu kB)\n", p_ulong, p_ulong/1024);
            clGetDeviceInfo(device[idx_device], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                            sizeof(cl_uint), &p_uint, NULL);
            printf("\tGLOBAL_MEM_CACHELINE_SIZE: %u\n", p_uint);
            clGetDeviceInfo(device[idx_device], CL_DEVICE_MAX_MEM_ALLOC_SIZE, 
                            sizeof(cl_ulong), &p_ulong, NULL);
            printf("\tMAX_MEM_ALLOC_SIZE: %lu (%lu MB)\n", p_ulong, p_ulong/(1024*1024));
            clGetDeviceInfo(device[idx_device], CL_DEVICE_LOCAL_MEM_SIZE,
                            sizeof(cl_ulong), &p_ulong, NULL);
            printf("\tLOCAL_MEM_SIZE: %lu (%lu kB)\n", p_ulong, p_ulong/1024);
            clGetDeviceInfo(device[idx_device], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                            sizeof(cl_ulong), &p_ulong, NULL);
            printf("\tMAX_CONSTANT_BUFFER_SIZE: %lu (%lu kB)\n", p_ulong, p_ulong/1024);

            // computation units
            printf("Computation unit info\n");
            clGetDeviceInfo(device[idx_device], CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(cl_uint), &p_uint, NULL);
            printf("\tMAX_COMPUTE_UNITS: %u\n", p_uint);
            clGetDeviceInfo(device[idx_device], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                            sizeof(cl_uint), &p_uint, NULL);
            printf("\tMAX_WORK_GROUP_SIZE: %u\n", p_uint);
            clGetDeviceInfo(device[idx_device], CL_DEVICE_MAX_WORK_ITEM_SIZES,
                            sizeof(size_t) * 3, p_size, NULL);
            printf("\tMAX_WORK_ITEM_SIZES[0]: %lu\n", p_size[0]);
            printf("\tMAX_WORK_ITEM_SIZES[1]: %lu\n", p_size[1]);
            printf("\tMAX_WORK_ITEM_SIZES[2]: %lu\n", p_size[2]);

            // prifiling info
            printf("Profiling info\n");
            clGetDeviceInfo(device[idx_device], CL_DEVICE_PROFILING_TIMER_RESOLUTION,
                            sizeof(size_t), p_size, NULL);
            printf("\tPROFILING_TIMER_RESOLUTION: %lu\n", p_size[0]);
            printf("\n");
        }

        free(device);
    }

    free(platform);
}
