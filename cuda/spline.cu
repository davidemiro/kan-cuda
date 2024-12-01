//
// Created by davide miro on 24/09/24.
//

#include <iostream>
#include <cmath>
#include <torch/extension.h>

using namespace std;


//2d tensor offset computation
__device__ size_t compute_offset(int dim, int i, int j){
    return dim * j + i;
}


//compute offset b_spline_basis
__device__ size_t compute_offset_base(size_t* dims, size_t* ids, int num_dims){
    size_t offset = 0;
    size_t multiplier = 1;
    for(int i = num_dims - 1; i >= 0; i--){
        offset += ids[i]*multiplier;
        multiplier *=dims[i];
    }

    return offset;

}

__global__ void b_spline_base(torch::Tensor b_spline_basis, torch::Tensor x, int batch_size, int num_input, int num_knots, int degree,torch::Tensor knots, int num_dims, size_t* dims, size_t* ids) {
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

    if(z >= batch_size || i >= num_input){
        return;
    }

    //[batch_size, num_input, num_activations, num_knots, degree]
    dims[0] = batch_size;
    dims[1] = num_input;
    dims[2] = num_knots;
    dims[3] = degree;

    ids[0] = z;
    ids[1] = i;

    for(int d = 0; d < degree; d++) {
        ids[3] = d;
        for (int j = 0; j < num_knots; j++) {
            ids[2] = j;

            idx = compute_offset(dims, ids, num_dims);
            t = x[compute_offset(num_input,z,i)]
            if (d == 0) {
                // Base case: piecewise constant function (degree 0)
                if (knots[compute_offset(num_knots,i,j)]<= t && t < knots[compute_offset(num_knots,i,j + 1)]) {
                    b_spline_basis[idx] = 1.0;
                } else {
                    b_spline_basis[idx] = 0.0;
                }
            } else {

                // Check the left term (avoid division by zero)
                if (knots[compute_offset(num_knots,i,j + d)] != knots[compute_offset(num_knots,i,j)]) {
                    ids[3] = d - 1;
                    idx = compute_offset(dims, ids, num_dims);
                    leftTerm = (t - knots[compute_offset(num_knots,i,j)]) / (knots[compute_offset(num_knots,i,j + d)] - knots.index[compute_offset(num_knots,i,j)] * b_spline_basis[idx]);
                    ids[3] = d;
                }

                // Check the right term (avoid division by zero)
                //TODO: fix the error j + d  + 1 > num_knots
                if (knots[compute_offset(num_knots,i,j + d + 1)] != knots[compute_offset(num_knots,i,j + 1)]) {
                    ids[2] = j + 1;
                    ids[3] = d - 1;
                    idx = compute_offset(dims, ids, num_dims);
                    rightTerm = (knots[compute_offset(num_knots,i,j + d + 1)] - t) / (knots[compute_offset(num_knots,i,j + d + 1)] - knots[compute_offset(num_knots,i,j + 1)]) * b_spline_basis[idx];
                    ids[2] = j;
                    ids[3] = d;
                }

                idx = compute_offset(dims, ids, num_dims);
                b_spline_basis[idx] = leftTerm + rightTerm;
            }
        }
    }
}

__global__ void spline(float* result, torch::Tensor cps, torch::Tensor b_spline_basis, int z, int i, int j, int d, int num_knots, int num_dims, size_t* dims, size_t* ids) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    /*
     * z : z-th batch element
     * i : i-th element of the input
     * j : j-th activation function
     * k : k-th knot
     * d : degree
     */
    if (k < num_knots) {
        //TODO: make this operation atomic
        *result += cps.index({j,k}).item<float>() * b_spline_basis.index({z,i,k,d - 1}).item<float>();
    }
}

