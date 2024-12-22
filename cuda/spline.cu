//
// Created by davide miro on 24/09/24.
//

#include <iostream>
#include <cmath>
#include <torch/extension.h>
#define DIMS batch_size, num_input, num_knots, degree

using namespace std;


//2d tensor idx computation
__device__ size_t compute_idx(int dim, int i, int j){
    return i + (dim * j) + j;
}


//compute idx b_spline_basis
__device__ size_t compute_idx_base(int z, int i, int j, int d,
                                      int batch_size, int num_input, int num_knots, int degree){

    int stride_num_knots = d;
    int stride_num_input = stride_num_knots * num_knots;
    int stride_batch_size = stride_num_input * num_input;


    return (z * stride_batch_size) + (i * stride_num_input) + (j * stride_num_knots) + d;

}

__global__ void b_spline_base(float* b_spline_basis, float* x, int batch_size, int num_input, int num_knots, int degree, float* knots) {
    /*
     * z : z-th batch element
     * i : i-th element of the input
     * j : j-th knot
     * k : degree
     */

    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.y;
    float t;
    float leftTerm = 0.0;
    float rightTerm = 0.0;
    size_t idx = 0;
    size_t idx_ = 0;

    if(z >= batch_size || i >= num_input){
        return;
    }

    //[batch_size, num_input, num_activations, num_knots, degree]
    for(int d = 0; d < degree; d++) {
        for (int j = 0; j < num_knots; j++) {

            idx = compute_idx_base(z, i, j, d, DIMS);
            t = x[compute_idx(num_input, z, i)];
            if (d == 0) {
                // Base case: piecewise constant function (degree 0)
                if (knots[compute_idx(num_knots, i, j)]<= t && t < knots[compute_idx(num_knots, i, j + 1)]) {
                    b_spline_basis[idx] = 1.0;
                } else {
                    b_spline_basis[idx] = 0.0;
                }
            } else {

                // Check the left term (avoid division by zero)
                if (knots[compute_idx(num_knots, i, j + d)] != knots[compute_idx(num_knots, i, j)]) {
                    idx_ = compute_idx_base(z, i, j, d - 1, DIMS);
                    leftTerm = (t - knots[compute_idx(num_knots,i,j)]) / (knots[compute_idx(num_knots, i, j + d)] - knots[compute_idx(num_knots, i, j)] * b_spline_basis[idx_]);
                }

                // Check the right term (avoid division by zero)
                //TODO: fix the error j + d  + 1 > num_knots
                if (knots[compute_idx(num_knots, i, j + d + 1)] != knots[compute_idx(num_knots, i, j + 1)]) {
                    idx_ = compute_idx_base(z, i, j + 1, d - 1, DIMS);
                    rightTerm = (knots[compute_idx(num_knots, i, j + d + 1)] - t) / (knots[compute_idx(num_knots, i, j + d + 1)] - knots[compute_idx(num_knots, i, j + 1)]) * b_spline_basis[idx_];
                }
                b_spline_basis[idx] = leftTerm + rightTerm;
            }
        }
    }
}

__global__ void spline(float* result, float* cps, float* b_spline_basis, int z, int i, int j, int batch_size, int num_input, int num_knots, int degree) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    /*
     * z : z-th batch element
     * i : i-th element of the input
     * j : j-th activation function
     * k : k-th knot
     * d : degree
     */

    size_t idx = compute_idx_base(z, i, j, degree - 1, DIMS);
    float mul = 0.0;
    if (k < num_knots) {

        mul = cps[compute_idx(num_knots, i, k)] * b_spline_basis[idx];
        atomicAdd(result, mul);
    }
}

