import torch
import torch.nn as nn
import cuda_kan
#import cpp_kan


class KolmogorovArnoldLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_knots, range_knots, degree, device):
        super(KolmogorovArnoldLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.num_knots = num_knots

        # Create random knot vector and control points
        self.knots = torch.linspace(range_knots[0], range_knots[1], num_knots).to(device)
        self.cps = torch.nn.Parameter(torch.ones(output_dim, num_knots)).to(device)

        self.wb = torch.nn.Parameter(torch.rand(input_dim, output_dim)).to(device)
        self.ws = torch.nn.Parameter(torch.rand(input_dim, output_dim)).to(device)

    def forward(self, x :torch.Tensor):
        if x.is_cuda:
            output = cuda_kan.kan_layer(x, self.wb, self.ws, self.knots, self.cps, self.degree)
        else:
            #output = cpp_kan.kan_layer(x, self.wb, self.ws, self.knots, self.cps, self.degree)
            pass

        return output









