__kernel void sweep(__global double *in, __global double *out, int kth, int n) {
    int row = get_global_id(0);
    int k;
    double mul_num = in[(row*n) + kth];
    if ((row < n) && (row != kth)) {
        for (k=0; k<n; ++k) {
            int idx = (row*n) + k;
            int idx_t = (kth*n) + k;
            double sub_num_i = in[idx_t] * mul_num;
            double sub_num_o = out[idx_t] * mul_num;

            in[idx] -= sub_num_i;
            out[idx] -= sub_num_o;
        }
    }
}

__kernel void search_swap_pivot(__global double *in, __global double *out, int i, int n) {
    int j;
    int pivot_row = i;
    double pivot_val;

    // search pivot row
    for (j=i+1; j<n; ++j) {
        if (in[(j*n) + i] > in[(pivot_row*n) + i]) {
            pivot_row = j;
        }
    }

    // swap pivot row
    if (i != pivot_row) {
        for (j=0; j<n; ++j) {
            double tmp_in  = in[(i*n) + j];
            double tmp_out = out[(i*n) + j];
            in[(i*n) + j]  = in[(pivot_row*n) + j];
            out[(i*n) + j] = out[(pivot_row*n) + j];
            in[(pivot_row*n) + j]  = tmp_in;
            out[(pivot_row*n) + j] = tmp_out;
        }
    }

    // normalize pivot row
    pivot_val = in[(i*n) + i];
    if (pivot_val != 0) {
        for (j=0; j<n; ++j) {
            in[(i*n) + j]  /= pivot_val;
            out[(i*n) + j] /= pivot_val;
        }
    }
}
