#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

//#define DEBUG_MODE

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

    // init
    generate_unit_matrix(out, n);

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
            for (j=0; j<n; ++j) {
                const double mul_num = in[(j*n) + i];
                if ((i != j) && NON_ZERO(mul_num)) {
                    for (k=0; k<n; ++k) {
                        const int idx = (j*n) + k;
                        const int idx_t = (i*n) + k;
                        const double sub_num_i = in[idx_t] * mul_num;
                        const double sub_num_o = out[idx_t] * mul_num;

                        in[idx] -= sub_num_i;
                        out[idx] -= sub_num_o;
                    }
                }
            }
        }
    }

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
