import torch
import torch.nn as nn
import cuda_kan
#import cpp_kan


class KolmogorovArnoldLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_knots, range_knots, degree=3):
        super(KolmogorovArnoldLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.num_knots = num_knots

        # Create random knot vector and control points
        self.knots = torch.linspace(range_knots[0], range_knots[1], num_knots).expand(input_dim,num_knots).to("cuda")
        self.cps = torch.nn.Parameter(torch.ones(output_dim, num_knots)).to("cuda")

        self.wb = torch.nn.Parameter(torch.rand(input_dim, output_dim)).to("cuda")
        self.ws = torch.nn.Parameter(torch.rand(input_dim, output_dim)).to("cuda")

    def forward(self, x :torch.Tensor):
        if x.is_cuda:
            output = cuda_kan.kan_layer(x, self.wb, self.ws, self.knots, self.cps, self.degree)
        else:
            pass
            #output = cpp_kan.kan_layer(x, self.wb, self.ws, self.knots, self.cps, self.degree)

        return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

x = torch.rand(3,3).to("cuda")
print(x)

layer = KolmogorovArnoldLayer(3,5,5, [0,100], 3)

y = layer(x)
y.to("cpu")

print(y)






