import torch
import torch
import torch.nn as nn
import numpy as np

class KolmogorovArnoldLayer(nn.Module):
    def __init__(self, input_dim, num_functions, num_knots, degree=3):
        super(KolmogorovArnoldLayer, self).__init__()

        self.input_dim = input_dim
        self.num_functions = num_functions
        self.degree = degree
        self.num_knots = num_knots

        # Create random knot vector and control points
        self.knots = np.linspace(0, 1, num_knots + degree + 1)
        self.control_points = np.random.rand(num_functions, input_dim, num_knots)

    # Function to generate cubic B-splines basis functions
    def generate_b_spline_basis(self,knots, degree, num_control_points, x_vals):
        """Generate cubic B-spline basis functions."""
        # Create B-spline basis functions for each control point
        basis_functions = []

        for i in range(len(knots) - degree - 1):
            # Create B-spline basis function with given degree and knot vector
            coeffs = np.zeros(len(knots) - degree - 1)
            coeffs[i] = 1  # Only one coefficient is 1 for a specific basis function
            #TODO: implement BSpline in CUDA and CPP
            spline = BSpline(knots, coeffs, degree)
            basis_functions.append(spline(x_vals))

        return np.array(basis_functions).T

    def univariate_function(self, i, j, x_vals):
        """Compute the j-th univariate function φ_ij."""
        basis_functions = self.generate_b_spline_basis(self.knots, self.degree, self.num_knots, x_vals)
        return np.dot(basis_functions, self.control_points[i, j, :])

    def approximate_function(self, x_vals):
        """Compute the KAN output for the input vector."""
        n = len(x_vals)
        result = np.zeros(n)

        # Loop over each function φ_i
        for i in range(self.num_functions):
            sum_inner = np.zeros(n)
            # Loop over the inputs and sum the ψ_ij functions
            for j in range(self.input_dim):
                sum_inner += self.univariate_function(i, j, x_vals[:, j])
            # Sum the final function φ_i
            result += self.univariate_function(i, 0, sum_inner)

        return result






    def forward(self, x):
        # Apply the univariate transformations (phi) to the input
        phi_outputs = [phi_fn(x) for phi_fn in self.phi]

        # Stack the outputs and sum them up
        phi_outputs = torch.cat(phi_outputs, dim=1)  # Shape: (batch_size, output_dim)

        # Apply the second layer of univariate transformations (psi)
        psi_outputs = [psi_fn(phi_outputs) for psi_fn in self.psi]

        # Stack the outputs of the psi transformations
        output = torch.cat(psi_outputs, dim=1)

        return output



