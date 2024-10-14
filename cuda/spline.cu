//
// Created by davide miro on 24/09/24.
//

#include <iostream>
#include <cmath>
#include <torch/extension.h>
#include <ATen/ATen.h>

using namespace std;

__device__ void bSplineBasis(float*** bSplineBasis, float** x, int num_input, int num_activations, int degree,const float* knots) {
    int z = threadIdx.x;
    float t;
    double leftTerm = 0.0;
    double rightTerm = 0.0;

    for(int i = 0; i < num_input; i++){ //TODO: we can parallelize this dimension ?
        for(int j = 0; j < num_activations; i++){
            for(int d = 0; d < degree; d++ ){
                t = x[z][i];
                if (d == 0) {
                    // Base case: piecewise constant function (degree 0)
                    if (knots[j] <= t && t < knots[j + 1]) {
                        bSplineBasis[z][i][j][d] = 1.0;
                    } else {
                        bSplineBasis[z][i][j][d] = 0.0;
                    }
                } else {


                    // Check the left term (avoid division by zero)
                    if (knots[j + d] != knots[j]) {
                        leftTerm = (t - knots[j]) / (knots[j + d] - knots[j]) * bSplineBasis[z][i][j][d - 1];
                    }

                    // Check the right term (avoid division by zero)
                    if (knots[j + d + 1] != knots[j + 1]) {
                        rightTerm = (knots[j + d + 1] - t) / (knots[j + d + 1] - knots[j + 1]) *
                                    bSplineBasis[z][i][j + 1][d - 1];
                    }

                    bSplineBasis[z][i][j][d] = leftTerm + rightTerm;
                }
            }
        }
    }



}

//TODO develop in CUDA
__global__ void b_spline(float* result, const float* cps, const float* knots, const float*** bSplineBasis, int j, int k){
    int j = threadIdx.x;
    /*
     * z : z-th batch element TODO
     * i : i-th element of the input
     * j : j-th activation function
     * k : degree
     */
    atomicAdd(&result, cps[i] * bSplineBasis[i][j][k];);
}

