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

        printf("[Platform %d]\n", idx_platform);

        clGetDeviceIDs(platform[idx_platform], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        device = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
        clGetDeviceIDs(platform[idx_platform], CL_DEVICE_TYPE_ALL, num_devices, device, NULL);

        printf("# of devices: %d\n", num_devices);
        for (idx_device=0; idx_device < num_devices; ++idx_device) {
            cl_uint p_uint;
            cl_ulong p_ulong;
            cl_device_type p_type;
            char buf[N];
            size_t p_size[3];

            printf("[Device #%d]\n", idx_device);

            // device info
            clGetDeviceInfo(device[idx_device], CL_DEVICE_VENDOR,
                            N, buf, NULL);
            printf("\tDEVICE_VENDOR: %s\n", buf);
            clGetDeviceInfo(device[idx_device], CL_DEVICE_NAME, 
                            N, buf, NULL);
            printf("\tDEVICE_NAME: %s\n", buf);
            clGetDeviceInfo(device[idx_device], CL_DEVICE_TYPE,
                            sizeof(cl_device_type), &p_type, NULL);
            printf("\tDEVICE_TYPE: %d\n", (int)p_type);


            // memory size
            clGetDeviceInfo(device[idx_device], CL_DEVICE_GLOBAL_MEM_SIZE,
                            sizeof(cl_ulong), &p_ulong, NULL);
            printf("\tGLOBAL_MEM_SIZE: %lu\n", p_ulong);
            clGetDeviceInfo(device[idx_device], CL_DEVICE_MAX_MEM_ALLOC_SIZE, 
                            sizeof(cl_ulong), &p_ulong, NULL);
            printf("\tMAX_MEM_ALLOC_SIZE: %lu\n", p_ulong);

            // computation units
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

            printf("\n");
        }

        free(device);
    }

    free(platform);
}
