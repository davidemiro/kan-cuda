//
// Created by davide miro on 24/09/24.
//

#include <iostream>
#include <cmath>
#include <torch/extension.h>
#include <ATen/ATen.h>

using namespace std;

__global__ void b_spline_base(float**** b_spline_basis, float** x, int batch_size, int num_input, int num_activations, int degree,const float* knots) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.y;
    float t;
    double leftTerm = 0.0;
    double rightTerm = 0.0;

    if(z >= batch_size || i >= num_input){
        return;
    }

    for(int d = 0; d < degree; d++) {
        for (int j = 0; j < num_activations; i++) {

            t = x[z][i];
            if (d == 0) {
                // Base case: piecewise constant function (degree 0)
                if (knots[j] <= t && t < knots[j + 1]) {
                    b_spline_basis[z][i][j][d] = 1.0;
                } else {
                    b_spline_basis[z][i][j][d] = 0.0;
                }
            } else {


                // Check the left term (avoid division by zero)
                if (knots[j + d] != knots[j]) {
                    leftTerm = (t - knots[j]) / (knots[j + d] - knots[j]) * b_spline_basis[z][i][j][d - 1];
                }

                // Check the right term (avoid division by zero)
                if (knots[j + d + 1] != knots[j + 1]) {
                    rightTerm = (knots[j + d + 1] - t) / (knots[j + d + 1] - knots[j + 1]) *
                                b_spline_basis[z][i][j + 1][d - 1];
                }

                b_spline_basis[z][i][j][d] = leftTerm + rightTerm;
            }

        }

    }
}

__global__ void spline(float* result, const float** cps, const float* knots, const float**** b_spline_basis, int i, int k){
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.x * blockDim.x + threadIdx.y;
    /*
     * z : z-th batch element
     * i : i-th element of the input
     * j : j-th activation function
     * k : degree
     */
    atomicAdd(&result, cps[j] * b_spline_basis[z][i][j][k]);
}

