import unittest
import torch
import kan_impl
import time
import kan

class KolmogorovArnoldNetworkTests(unittest.TestCase):

    def test_speed_kan(self):
        batch_size = 256
        input_dim = 512
        output_dim = 2048
        grid = 4
        k = 3

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x = torch.rand(batch_size,input_dim)

        layer = kan_impl.KolmogorovArnoldLayer(input_dim=input_dim,output_dim=output_dim,num_knots=grid, range_knots=[0,100], degree=k)


        official_layer = kan.KAN(width=[output_dim], grid=grid, k=k, seed=42, device=device)

        start = time.time();
        layer(x)
        print("Unofficial KAN {}".format(time.time() - start))

        start = time.time();
        official_layer(x)

        print("Official KAN {}".format(time.time() - start))



if __name__ == "__main__":
    unittest.main()



