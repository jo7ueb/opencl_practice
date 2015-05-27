__kernel void sweep(__global double *in, __global double *out, int kth, int n) {
    int row = get_global_id(0);
    int k;
    double mul_num = in[(kth*n) + kth];
    if ((row < n) && (row != kth)) {
        for (k=0; k<n; ++k) {
            int idx = (kth*n) + k;
            int idx_t = (row*n) + k;
            double sub_num_i = in[idx_t] * mul_num;
            double sub_num_o = out[idx_t] * mul_num;

            in[idx] -= sub_num_i;
            out[idx] -= sub_num_o;
            in[idx] = 0;
            out[idx] = 0;
        }
    }
}
