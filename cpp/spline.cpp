//
// Created by davide miro on 24/09/24.
//

#include <iostream>
#include <cmath>
#include <torch/extension.h>


using namespace std;

void b_spline_base(float**** b_spline_basis, float** x, int batch_size, int num_input, int num_knots, int degree,float** knots) {
    /*
     * z : z-th batch element
     * i : i-th element of the input
     * j : j-th knot
     * k : degree
     */


    float t;
    double leftTerm = 0.0;
    double rightTerm = 0.0;

    for(int z = 0; z < batch_size; z++) {
        for (int i = 0; i < num_input; i++) {
            for (int d = 0; d < degree; d++) {
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
                            leftTerm = (t - knots[i][j]) / (knots[i][j + d] - knots[i][j]) *
                                       b_spline_basis[z][i][j][d - 1];
                        }

                        // Check the right term (avoid division by zero)
                        if (knots[i][j + d + 1] != knots[i][j + 1]) {
                            rightTerm = (knots[i][j + d + 1] - t) / (knots[i][j + d + 1] - knots[i][j + 1]) *
                                        b_spline_basis[z][i][j + 1][d - 1];
                        }

                        b_spline_basis[z][i][j][d] = leftTerm + rightTerm;
                    }
                }
            }
        }
    }
}

float spline(float** cps, float**** b_spline_basis, int z, int i, int j, int d, int num_knots) {
    /*
     * z : z-th batch element
     * i : i-th element of the input
     * j : j-th activation function
     * k : k-th knot
     * d : degree
     */
    float result = 0.0;
    for(int k = 0; k < num_knots; k++){
        result += cps[j][k] * b_spline_basis[z][i][k][d];
    }

    return result;
}