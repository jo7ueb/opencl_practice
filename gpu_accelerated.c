#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <CL/cl.h>

#define DEBUG_MODE

#define N 2
#define NON_ZERO(x)  (fabs(x) > DBL_EPSILON)

static void generate_random_matrix(double *mat, int n);
static void get_inverse(double *in, double *out, int n);
static void generate_unit_matrix(double *mat, int n);
static int search_pivot_row(double *mat, int n, int round);
static void swap_row(double *mat, int n, int source, int dest);
static void print_matrix(double *mat, int n);

int main(int argc, char **argv) {
    int n, size;
    long long int us_before, us_after;
    double *mat;
    double *inverse;
    struct timeval tv_before, tv_after;

    // size
    n = (argc == 1) ? N : atoi(argv[1]);
    size = n*n;
    mat = (double *)malloc(sizeof(double) * size);
    inverse = (double *)malloc(sizeof(double) * size);

    generate_random_matrix(mat, n);
#ifdef DEBUG_MODE
    printf("Input matrix\n");
    print_matrix(mat, n);
#endif

    // calculate inverse
    gettimeofday(&tv_before, NULL);
    get_inverse(mat, inverse, n);
    gettimeofday(&tv_after, NULL);

    // check process time
    us_before = (tv_before.tv_sec * 1000000) + tv_before.tv_usec;
    us_after = (tv_after.tv_sec * 1000000) + tv_after.tv_usec;
    printf("%d,%lld\n", n, us_after-us_before);

#ifdef DEBUG_MODE
    printf("After calc\n");
    print_matrix(mat, n);
    printf("Inverse matrix\n");
    print_matrix(inverse, n);
#endif

    free(mat);
    free(inverse);

    return 0;
}

static void generate_random_matrix(double *mat, int n) {
    int i, j;
    for (i=0; i<n; ++i)
        for (j=0; j<n; ++j)
            mat[(i*n)+j] = (double)(rand() % 100);
}

static void get_inverse(double *in, double *out, int n) {
    int i, j, k;
    FILE *fp;
    char *source;
    size_t length_source;
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_q = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem mem_in = NULL;
    cl_mem mem_out = NULL;
    cl_int ret;
    size_t global_size = (n/256) + 1;
    size_t local_size  = 256;

    // init
    generate_unit_matrix(out, n);

    // OpenCL init
    clGetPlatformIDs(1, &platform_id, NULL);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    command_q = clCreateCommandQueue(context, device_id, 0, NULL);

    // load and load kernel
    fp = fopen("sweep.cl", "r");
    fseek(fp, 0, SEEK_END);
    length_source = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    source = (char *)malloc(length_source + 1);
    fread(source, 1, length_source, fp);
    //printf("%s\n", source);
    fclose(fp);

    // build kernel
    program = clCreateProgramWithSource(context, 1, (const char **)&source, 
                                        (const size_t *)&length_source, NULL);
    if (program == NULL)
        fprintf(stderr, "Create Program failed!\n");
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        char *log;
        size_t length_log;
        fprintf(stderr, "Build Program failed! - %d\n", ret);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &length_log);
        log = (char *)malloc(length_log);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, length_log, log, NULL);
        fprintf(stderr, "%s\n", log);
        free(log);
    }
    kernel = clCreateKernel(program, "sweep", NULL);
    if (kernel == NULL)
        fprintf(stderr, "Create kernel failed!\n");

    // set kernel args
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_in);
    if (ret != CL_SUCCESS)
        fprintf(stderr, "SetKernelArg #%d failed!\n", 0);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_out);
    if (ret != CL_SUCCESS)
        fprintf(stderr, "SetKernelArg #%d failed!\n", 1);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &n);
    if (ret != CL_SUCCESS)
        fprintf(stderr, "SetKernelArg #%d failed!\n", 3);

    // memory init
    mem_in  = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(double)*n*n, in, NULL);
    mem_out = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(double)*n*n, out, NULL);

    if ((mem_in == NULL) || (mem_out == NULL))
        fprintf(stderr, "Memory alloc failed!\n");

    // calculate inverse
    for (i=0; i<n; ++i) {
        int row_pivot;
        double pivot;

        // pivot search and swap
        row_pivot = search_pivot_row(in, n, i);
        if (row_pivot != i) {
            swap_row(in, n, i, row_pivot);
            swap_row(out, n, i, row_pivot);
        }

        if (NON_ZERO(in[(i*n) + i])) {
            // normalize
            pivot = 1.0 / in[(i*n) + i];
            for (j=0; j<n; ++j) {
                const int idx = (i*n) + j;
                in[idx] *= pivot;
                out[idx] *= pivot;
            }

            // hakidashi
            ret = clSetKernelArg(kernel, 2, sizeof(int), &i);
            if (ret != CL_SUCCESS)
                fprintf(stderr, "SetKernelArg #%d failed!\n", 2);
            ret = clEnqueueNDRangeKernel(command_q, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
            if (ret != CL_SUCCESS)
                fprintf(stderr, "Execution failed!\n");
        }
    }

    // close
    clFlush(command_q);
    clFinish(command_q);
    clReleaseMemObject(mem_in);
    clReleaseMemObject(mem_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_q);
    clReleaseContext(context);
    free(source);
}

static int search_pivot_row(double *mat, int n, int round) {
    int i;
    int prow = round;
    for (i=round+1; i<n; ++i) {
        if (mat[(i * n) + round] > mat[(prow * n) + round])
            prow = i;
    }

    return prow; 
}

static void swap_row(double *mat, int n, int source, int dest) {
    int i;
    for (i=0; i<n; ++i) {
        double tmp = mat[(dest * n) + i];
        mat[(dest * n) + i] = mat[(source * n) + i];
        mat[(source * n) + i] = tmp;
    }
}

static void generate_unit_matrix(double *mat, int n) {
    int i, j;
    for(i=0; i<n; ++i)
        for (j=0; j<n; ++j)
            mat[(i*n) + j] = (i==j) ? 1.0 : 0.0;
}

static void print_matrix(double *mat, int n) {
    int i, j;

    for (i=0; i<n; ++i) {
        for (j=0; j<n; ++j) {
            printf("%lf,", mat[(i*n) + j]);
        }
        printf("\n");
    }
}
