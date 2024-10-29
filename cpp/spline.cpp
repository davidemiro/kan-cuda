//
// Created by davide miro on 24/09/24.
//

#include <iostream>
#include <cmath>
#include <torch/extension.h>
#include <ATen/ATen.h>

using namespace std;

float b_spline_base(int i, int d, double t, const float* knots) {
    if (d == 0) {
        // Base case: piecewise constant function (degree 0)
        if (knots[i] <= t && t < knots[i + 1]) {
            return 1.0;
        } else {
            return 0.0;
        }
    } else {
        // Recursive computation of the B-spline basis function
        double leftTerm = 0.0;
        double rightTerm = 0.0;

        // Check the left term (avoid division by zero)
        if (knots[i + d] != knots[i]) {
            leftTerm = (t - knots[i]) / (knots[i + d] - knots[i]) * b_spline_base(i, d - 1, t, knots);
        }

        // Check the right term (avoid division by zero)
        if (knots[i + d + 1] != knots[i + 1]) {
            rightTerm = (knots[i + d + 1] - t) / (knots[i + d + 1] - knots[i + 1]) * b_spline_base(i + 1, d - 1, t, knots);
        }

        return leftTerm + rightTerm;
    }
}

float b_spline(float t, int n, const float* controlPoints, const float* knots, int degree){
    float result = 0.0;

    for (int i = 0; i < n; ++i) {
        float basisValue = bSplineBasis(i, degree, t, knots);
        result += controlPoints[i] * basisValue;
    }

    return result;
}