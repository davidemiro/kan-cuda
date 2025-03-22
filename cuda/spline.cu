//
// Created by davide miro on 24/09/24.
//

#include <iostream>
#include <cmath>
#include <torch/extension.h>
#include <stdio.h>
#define DIMS batch_size, num_input, num_knots, degree

using namespace std;


//2d tensor idx computation
__device__ int compute_idx(int i, int j, int dim){
    return i * dim + j;
}


//compute idx b_spline_basis
__device__ int compute_idx_base(int z, int i, int j, int d,
                                      int batch_size, int num_input, int num_knots, int degree){

    int stride_degree = num_input * num_knots * batch_size;
    int stride_batch_size = num_input * num_knots;
    int stride_num_input = num_knots;



    return (d * stride_degree) + (z * stride_batch_size) + (i * stride_num_input) + j;

}

__global__ void b_spline_base(float* b_spline_basis, float* x, int batch_size, int num_input, int num_knots, int degree, float* knots) {
    /*
     * z : z-th batch element
     * i : i-th element of the input
     * j : j-th knot
     * k : degree
     */

    int z = blockIdx.x;
    int i = threadIdx.x;

    float t;
    float leftTerm = 0.0;
    float rightTerm = 0.0;
    size_t idx = 0;
    size_t idx_ = 0;

    //dynamic cache
    extern __shared__ float cache_ptr[];
    float* knots_cache = cache_ptr;




    //coalesce load to cache, using grid-stride loop to handle the case batch_size * num_input < num_knots
    for (int x = blockIdx.x * blockDim.x + threadIdx.x;
         x < num_knots;
         x += blockDim.x * gridDim.x){
        knots_cache[x] = knots[x];
    }
    __syncthreads();

    if(z >= batch_size || i >= num_input){
        return;
    }

    t = x[compute_idx(z, i, num_input)];

    for(int d = 0; d <= degree; d++) {
        for (int j = 0; j < num_knots ; j++) {

            idx = compute_idx_base(z, i, j, d, DIMS);

            if (d == 0) {
                if (knots_cache[j]<= t && t < knots_cache[j + 1]) {
                    b_spline_basis[idx] = 1.0;
                } else {
                    b_spline_basis[idx] = 0.0;
                }
            } else {

                if (knots_cache[j + d] != knots_cache[j]) {
                    idx_ = compute_idx_base(z, i, j, d - 1, DIMS);
                    leftTerm = (t - knots_cache[j]) / (knots_cache[j + d] - knots_cache[j] * b_spline_basis[idx_]);
                }
                if (knots_cache[j + d + 1] != knots_cache[j + 1]) {
                    idx_ = compute_idx_base(z, i, j + 1, d - 1, DIMS);
                    rightTerm = (knots_cache[j + d + 1] - t) / (knots_cache[j + d + 1] - knots_cache[j + 1]) * b_spline_basis[idx_];
                }
                b_spline_basis[idx] = leftTerm + rightTerm;
            }
        }
    }
}

__device__ float spline(float* cps, float* b_spline_basis, int z, int i, int j, int batch_size, int num_input, int num_knots, int degree) {
    /*
     * z : z-th batch element
     * i : i-th element of the input
     * j : j-th activation function
     * k : k-th knot
     * d : degree
     */

    float result = 0.0;
    size_t idx = compute_idx_base(z, i, j, degree, DIMS);

    for(int k = 0; k < num_knots; k++){
        result = result + (cps[compute_idx(num_knots, i, k)] * b_spline_basis[idx]);
    }

    return result;
}


