import unittest
import torch
from kan import *
import time
import cuda_kan

class KolmogorovArnoldNetworkTests(unittest.TestCase):

    def test_speed_kan(self):
        batch_size = 32
        input_dim = 1024
        output_dim = 2048
        grid = 4
        k = 3

        x = torch.rand(batch_size,input_dim)

        layer = KolmogorovArnoldLayer(input_dim=input_dim,output_dim=output_dim,num_knots=grid, knots_range=[0,100], degree=3)

        official_layer = KAN(width=[output_dim], grid=3, k=3, seed=42, device=device)

        start = time.time();
        layer(x)
        print("Unofficial KAN {}".format(time.time() - start))

        start = time.time();
        official_layer(x)

        print("Official KAN {}".format(time.time() - start))



if __name__ == "__main__":
    unittest.main()



