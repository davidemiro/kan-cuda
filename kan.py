import torch
import torch
import torch.nn as nn
import numpy as np
import cuda_kan
import cpp_kan

class KolmogorovArnoldLayer(nn.Module):
    def __init__(self, input_dim, num_functions, num_knots, degree=3):
        super(KolmogorovArnoldLayer, self).__init__()

        self.input_dim = input_dim
        self.num_functions = num_functions
        self.degree = degree
        self.num_knots = num_knots

        # Create random knot vector and control points
        self.knots = np.linspace(0, 1, num_knots + degree + 1)
        self.cps = np.random.rand(num_functions, input_dim, num_knots)

    def forward(self, x :torch.Tensor):

        if x.is_cuda:
            output = cuda_kan.kan_layer(x, wb, ws, knots, cps)
        else:
            output = cpp_kan.kan_layer(x, wb, ws, knots, cps)

        return output



