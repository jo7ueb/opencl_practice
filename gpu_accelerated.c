#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <CL/cl.h>

#define N 2
#define NON_ZERO(x)  (fabs(x) > DBL_EPSILON)

#ifdef PROF_MODE
#define CMDQ_PARAM CL_QUEUE_PROFILING_ENABLE
#define PROF(event, type, round) show_profiling_info(event, type, round);
#else
#define CMDQ_PARAM 0
#define PROF(event, type, round)
#endif /* PROF_MODE */

typedef enum {
    HOST2DEV_1,
    HOST2DEV_2,
    KERNEL_PIVOT,
    KERNEL_SWEEP,
    DEV2HOST_1,
    DEV2HOST_2,
} prof_type;

static void generate_random_matrix(double *mat, int n);
static void get_inverse(double *in, double *out, int n);
static void generate_unit_matrix(double *mat, int n);
static int search_pivot_row(double *mat, int n, int round);
static void swap_row(double *mat, int n, int source, int dest);
static void print_matrix(double *mat, int n);
#ifdef PROF_MODE
static void show_profiling_info(cl_event *event, prof_type type, int round);
#endif /* PROF_MODE */

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
    cl_kernel kernel_pivot = NULL;
    cl_kernel kernel_sweep = NULL;
    cl_mem mem_in = NULL;
    cl_mem mem_out = NULL;
    cl_event event_trans;
    cl_int ret;
    size_t global_size = n;
    size_t local_size  = 1;

    // init
    generate_unit_matrix(out, n);

    // OpenCL init
    clGetPlatformIDs(1, &platform_id, NULL);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    command_q = clCreateCommandQueue(context, device_id, CMDQ_PARAM, NULL);

    // load and load kernel
    fp = fopen("sweep.cl", "r");
    fseek(fp, 0, SEEK_END);
    length_source = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    source = (char *)malloc(length_source + 1);
    fread(source, 1, length_source, fp);
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

    kernel_pivot = clCreateKernel(program, "search_swap_pivot", NULL);
    if (kernel_pivot == NULL)
        fprintf(stderr, "Create kernel_pivot failed!\n");
    kernel_sweep = clCreateKernel(program, "sweep", NULL);
    if (kernel_sweep == NULL)
        fprintf(stderr, "Create kernel_sweep failed!\n");

    // memory init
    mem_in  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*n*n, NULL, NULL);
    mem_out = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*n*n, NULL, NULL);
    if ((mem_in == NULL) || (mem_out == NULL))
        fprintf(stderr, "Memory alloc failed!\n");

    // kernel_pivot args
    ret = clSetKernelArg(kernel_pivot, 0, sizeof(cl_mem), &mem_in);
    if (ret != CL_SUCCESS)
        fprintf(stderr, "SetKernelArg #%d failed!\n", 0);
    ret = clSetKernelArg(kernel_pivot, 1, sizeof(cl_mem), &mem_out);
    if (ret != CL_SUCCESS)
        fprintf(stderr, "SetKernelArg #%d failed!\n", 1);
    ret = clSetKernelArg(kernel_pivot, 3, sizeof(int), &n);
    if (ret != CL_SUCCESS)
        fprintf(stderr, "SetKernelArg #%d failed!\n", 3);

    // kernel_sweep args
    ret = clSetKernelArg(kernel_sweep, 0, sizeof(cl_mem), &mem_in);
    if (ret != CL_SUCCESS)
        fprintf(stderr, "SetKernelArg #%d failed!\n", 0);
    ret = clSetKernelArg(kernel_sweep, 1, sizeof(cl_mem), &mem_out);
    if (ret != CL_SUCCESS)
        fprintf(stderr, "SetKernelArg #%d failed!\n", 1);
    ret = clSetKernelArg(kernel_sweep, 3, sizeof(int), &n);
    if (ret != CL_SUCCESS)
        fprintf(stderr, "SetKernelArg #%d failed!\n", 3);

    // memory transfer
    ret = clEnqueueWriteBuffer(command_q, mem_in, CL_TRUE, 0, sizeof(double)*n*n, in, 0, NULL, &event_trans);
    if (ret != CL_SUCCESS)
        fprintf(stderr, "WriteBuffer(in) failed!\n");
    PROF(&event_trans, HOST2DEV_1, i);

    ret = clEnqueueWriteBuffer(command_q, mem_out, CL_TRUE, 0, sizeof(double)*n*n, out, 0, NULL, &event_trans);
    if (ret != CL_SUCCESS)
        fprintf(stderr, "WriteBuffer(out) failed!\n");
    PROF(&event_trans, HOST2DEV_2, i);

    // calculate inverse
    for (i=0; i<n; ++i) {
        int row_pivot;
        double pivot;
        cl_event event_pivot;
        cl_event event_sweep;

        // set kernel args
        ret = clSetKernelArg(kernel_pivot, 2, sizeof(int), &i);
        if (ret != CL_SUCCESS)
            fprintf(stderr, "SetKernelArg #%d failed!\n", 2);
        ret = clSetKernelArg(kernel_sweep, 2, sizeof(int), &i);
        if (ret != CL_SUCCESS)
            fprintf(stderr, "SetKernelArg #%d failed!\n", 2);

        // pivot search and swap
        ret = clEnqueueTask(command_q, kernel_pivot, 0, NULL, &event_pivot);
        if (ret != CL_SUCCESS)
            fprintf(stderr, "Execution pivot failed!\n");
        PROF(&event_pivot, KERNEL_PIVOT, i);

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
            ret = clEnqueueNDRangeKernel(command_q, kernel_sweep, 1, NULL, &global_size, &local_size, 1, &event_pivot, &event_sweep);
            if (ret != CL_SUCCESS)
                fprintf(stderr, "Execution sweep failed!\n");
            PROF(&event_sweep, KERNEL_SWEEP, i);

        }
    }

    // memory transfer
    ret = clEnqueueReadBuffer(command_q, mem_in, CL_TRUE, 0, sizeof(double)*n*n, in, 0, NULL, &event_trans);
    if (ret != CL_SUCCESS)
        fprintf(stderr, "WriteBuffer(in) failed!\n");
    PROF(&event_trans, DEV2HOST_1, i);

    ret = clEnqueueReadBuffer(command_q, mem_out, CL_TRUE, 0, sizeof(double)*n*n, out, 0, NULL, &event_trans);
    if (ret != CL_SUCCESS)
        fprintf(stderr, "WriteBuffer(out) failed!\n");
    PROF(&event_trans, DEV2HOST_2, i);

    // close
    clFlush(command_q);
    clFinish(command_q);
    clReleaseMemObject(mem_in);
    clReleaseMemObject(mem_out);
    clReleaseKernel(kernel_sweep);
    clReleaseKernel(kernel_pivot);
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

#ifdef PROF_MODE
static void show_profiling_info(cl_event *event, prof_type type, int round) {
    char *type_str;
    cl_ulong tick_qd, tick_sub, tick_start, tick_end;
    cl_ulong time_total, to_submit, to_start, to_end;
    static int flag_header = 0;

    // wait for event
    clWaitForEvents(1, event);

    // get timer count (assume 1ns/tick)
    clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &tick_qd, NULL);
    clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &tick_sub, NULL);
    clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tick_start, NULL);
    clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tick_end, NULL);

    // typestr
    switch (type) {
    case HOST2DEV_1:   type_str = "HOST2DEV_1"; break;
    case HOST2DEV_2:   type_str = "HOST2DEV_2"; break;
    case KERNEL_PIVOT: type_str = "KERNEL_PIVOT"; break;
    case KERNEL_SWEEP: type_str = "KERNEL_SWEEP"; break;
    case DEV2HOST_1:   type_str = "DEV2HOST_1"; break;
    case DEV2HOST_2:   type_str = "DEV2HOST_2"; break;
    default: type_str = "UNKNOWN";
    }

    time_total = tick_end - tick_qd;
    to_submit  = tick_sub - tick_qd;
    to_start   = tick_start - tick_sub;
    to_end     = tick_end - tick_start;

    if (flag_header == 0) {
        printf("\"@\",\"round\",\"Event type\",\"Total time (ns)\",\"SUBMIT-QUEUED (ns)\",\"START-SUBMIT (ns)\",\"END-START\"\n");
        ++flag_header;
    }
    printf("@,%d,%s,%lu,%lu,%lu,%lu\n", round, type_str, time_total, to_submit, to_start, to_end);
}
#endif /* PROF_MODE */
