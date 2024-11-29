//
// Created by davide miro on 24/09/24.
//

#include <iostream>
#include <cmath>
#include <torch/extension.h>

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

            t = x.index({z,i}).item<float>();
            if (d == 0) {
                // Base case: piecewise constant function (degree 0)
                if (knots.index({i,j}).item<float>() <= t && t < knots.index({i,j + 1}).item<float>()) {
                    b_spline_basis.index_put_({z,i,j,d}, 1.0);
                } else {
                    b_spline_basis.index_put_({z,i,j,d}, 0.0);
                }
            } else {

                // Check the left term (avoid division by zero)
                if (knots.index({i,j + d}).item<float>() != knots.index({i,j}).item<float>()) {
                    leftTerm = (t - knots.index({i,j}).item<float>()) / (knots.index({i,j + d}).item<float>() - knots.index({i,j}).item<float>() * b_spline_basis.index({z,i,j,d - 1}).item<float>());
                }

                // Check the right term (avoid division by zero)
                //TODO: fix the error j + d  + 1 > num_knots
                if (knots.index({i,j + d + 1}).item<float>() != knots.index({i,j + 1}).item<float>()) {
                    rightTerm = (knots.index({i,j + d + 1}).item<float>() - t) / (knots.index({i,j + d + 1}).item<float>() - knots.index({i,j + 1}).item<float>()) * b_spline_basis.index({z,i,j + 1,d - 1}).item<float>();
                }

                b_spline_basis.index_put_({z,i,j,d}, leftTerm + rightTerm);
            }
        }
    }
}

__global__ void spline(float* result, torch::Tensor cps, torch::Tensor b_spline_basis, int z, int i, int j, int d, int num_knots) {
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

