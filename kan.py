import torch
import torch.nn as nn
#import cuda_kan
import cpp_kan


class KolmogorovArnoldLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_knots, range_knots, degree=3):
        super(KolmogorovArnoldLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.num_knots = num_knots

        # Create random knot vector and control points
        self.knots = torch.linspace(range_knots[0], range_knots[1], num_knots).expand(input_dim,num_knots)
        self.cps = torch.nn.Parameter(torch.ones(input_dim, num_knots))

        self.wb = torch.nn.Parameter(torch.rand(input_dim, output_dim))
        self.ws = torch.nn.Parameter(torch.rand(input_dim, output_dim))

    def forward(self, x :torch.Tensor):
        if x.is_cuda:
            print("IS_CUDA")
            output = None
            #output = cuda_kan.kan_layer(x, self.wb, self.ws, self.knots, self.cps, self.degree)
        else:
            print("IS_CPP")
            output = cpp_kan.kan_layer(x, self.wb, self.ws, self.knots, self.cps, self.degree)

        return output


x = torch.Tensor([1.0, 2.0, 3.0])

layer = KolmogorovArnoldLayer(3,3,3)

print(layer(x))



