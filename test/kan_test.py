import unittest
import torch
from kan import *
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

x = torch.rand(32,1000).to("cuda")

layer = KolmogorovArnoldLayer(1000,1004,7, [0,100], 3)

y = layer(x)
y.to("cpu")

print(y)
class KolmogorovArnoldNetworkTests(unittest.TestCase):

    def test_speed_kan(self):
        batch_size = 32
        input_dim = 1024
        output_dim = 2048
        grid = 3
        k = 3

        x = torch.rand(batch_size,input_dim)

        layer = KolmogorovArnoldLayer(input_dim=input_dim,output_dim=output_dim,num_knots=grid, knots_range=[0,100], degree=3)

        official_layer = KAN(width=[2,5,1], grid=3, k=3, seed=42, device=device)

        start = time.time();
        layer(x)
        print("Unofficial KAN {}".format(time.time() - start))

        start = time.time();
        official_layer(x)

