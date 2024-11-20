//
// Created by davide miro on 24/09/24.
//

#include <iostream>
#include <cmath>
#include <torch/extension.h>
#include <ATen/ATen.h>

using namespace std;

__global__ void b_spline_base(float**** b_spline_basis, float** x, int batch_size, int num_input, int num_knots, int degree,float** knots) {
    /*
     * z : z-th batch element
     * i : i-th element of the input
     * j : j-th knot
     * k : degree
     */

    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.y;
    float t;
    double leftTerm = 0.0;
    double rightTerm = 0.0;

    if(z >= batch_size || i >= num_input){
        return;
    }

    for(int d = 0; d < degree; d++) {
        for (int j = 0; j < num_knots; j++) {

            t = x[z][i];
            if (d == 0) {
                // Base case: piecewise constant function (degree 0)
                if (knots[i][j] <= t && t < knots[i][j + 1]) {
                    b_spline_basis[z][i][j][d] = 1.0;
                } else {
                    b_spline_basis[z][i][j][d] = 0.0;
                }
            } else {


                // Check the left term (avoid division by zero)
                if (knots[i][j + d] != knots[i][j]) {
                    leftTerm = (t - knots[i][j]) / (knots[i][j + d] - knots[i][j]) * b_spline_basis[z][i][j][d - 1];
                }

                // Check the right term (avoid division by zero)
                if (knots[i][j + d + 1] != knots[i][j + 1]) {
                    rightTerm = (knots[i][j + d + 1] - t) / (knots[i][j + d + 1] - knots[i][j + 1]) * b_spline_basis[z][i][j + 1][d - 1];
                }

                b_spline_basis[z][i][j][d] = leftTerm + rightTerm;
            }

        }

    }
}

__global__ void spline(float* result, float** cps, float**** b_spline_basis, int z, int i, int j, int d, int num_knots) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    /*
     * z : z-th batch element
     * i : i-th element of the input
     * j : j-th activation function
     * k : k-th knot
     * d : degree
     */
    if (k < num_knots) {
        atomicAdd(&result, cps[j][k] * b_spline_basis[z][i][k][d]);
    }
}

