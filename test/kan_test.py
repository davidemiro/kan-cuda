import unittest
import torch
import kan_impl
import time

class KolmogorovArnoldNetworkTests(unittest.TestCase):

    def test_speed_kan(self):
        batch_size = 1024
        input_dim = 512
        output_dim = 1024
        grid = 4
        k = 3

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(device)

        x = torch.rand(batch_size,input_dim).to(device)

        layer = kan_impl.KolmogorovArnoldLayer(input_dim=input_dim,output_dim=output_dim,num_knots=grid, range_knots=[0,100], degree=k, device=device)


        official_layer = kan_impl.KAN(width=[output_dim], grid=grid, k=k, seed=42, device=device)

        start = time.time();
        layer(x)
        print("Unofficial KAN {}".format(time.time() - start))

        start = time.time();
        official_layer(x)

        print("Official KAN {}".format(time.time() - start))



if __name__ == "__main__":
    unittest.main()



