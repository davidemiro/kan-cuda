//
// Created by davide miro on 07/09/24.
//

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <ATen/ATen.h>
#include <algorithm>


#include "spline.cu"

#define MAX_DIM 1024
#define MAX_THD 1024


using namespace std;


namespace cuda_kan {

    __device__ float silu(float x) {
        return x / (1 + expf(x * -1));
    }

    __global__ void kan_activation_function(float **x, float **y, float **wb, float **ws, float **cps, float ****b_spline_basis, int k, int batch_size, int num_inputs, int num_activations, int num_knots) {

        int z = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.x * blockDim.x + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.z;

        float result = 0.0;

        if (i < num_inputs && z < batch_size && j < num_activations) {
            spline<<<1,num_knots>>>(&result, cps, b_spline_basis, z, i, j, k, num_knots);
            result = result + wb[i][j] * silu(x[z][i]) + ws[i][j];
            atomicAdd(&y[z][j], result);
        }

    }


    at::Tensor kan_layer(at::Tensor x, at::Tensor wb, at::Tensor ws, at::Tensor knots, at::Tensor cps, int degree) {
        /*
         * x : [batch_size, input_dim]
         * y : [batch_size, output_dim]
         * wb,ws: [input_dim, output_dim]
         * cps : [output_dim, num_knots]
         * knots : [input_dim, num_knots]
         */

        TORCH_CHECK(wb.size(0) < MAX_DIM); //TODO: review check
        TORCH_CHECK(knots.size(0) < MAX_DIM);
        TORCH_CHECK(cps.size(0) < MAX_DIM);


        TORCH_CHECK(x.dtype() == at::kFloat);
        TORCH_CHECK(wb.dtype() == at::kFloat);
        TORCH_CHECK(ws.dtype() == at::kFloat);
        TORCH_CHECK(wb.dtype() == at::kFloat);
        TORCH_CHECK(ws.dtype() == at::kFloat);

        TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(wb.device().type() == at::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(ws.device().type() == at::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(knots.device().type() == at::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(cps.device().type() == at::DeviceType::CUDA);

        int batch_size = x.size(0);
        int num_input = x.size(1);
        int num_activations = wb.size(0);
        int num_knots = cps.size(1);


        at::Tensor x_contig = x.contiguous();
        at::Tensor wb_contig = wb.contiguous();
        at::Tensor ws_contig = ws.contiguous();
        at::Tensor cps_contig = cps.contiguous();
        at::Tensor knots_contig = knots.contiguous();



        at::Tensor y = torch::zeros({batch_size, num_activations}, wb_contig.options());
        at::Tensor b_spline_basis = torch::empty({batch_size,num_input,num_activations,degree}, wb_contig.options());


        float **x_ptr = x_contig.data_ptr<float*>();
        float **cps_ptr = cps_contig.data_ptr<float*>();
        float **wb_ptr = wb_contig.data_ptr<float*>();
        float **ws_ptr = ws_contig.data_ptr<float*>();
        float **knots_ptr = knots_contig.data_ptr<float*>();

        float **y_ptr = y.data_ptr<float*>();
        float ****b_spline_basis_ptr = b_spline_basis.data_ptr<float***>();

        int dim = MAX_DIM / 3;
        int num_block = max(batch_size, num_input);
        dim3 threads_block(min(dim + 1,batch_size),min(dim,num_input)); // batch_size x num_input
        b_spline_base<<<num_block, threads_block>>>(b_spline_basis_ptr, x_ptr, batch_size, num_input, num_activations, degree, knots_ptr);


        num_block = max(batch_size,max(num_input,num_activations));
        dim3 threads_block_(min(dim + 1,batch_size),min(dim,num_input),min(dim,num_activations)); // batch_size x num_input x num_activations
        kan_activation_function<<<num_block, threads_block_>>>(x_ptr, y_ptr, wb_ptr, ws_ptr, cps_ptr, b_spline_basis_ptr, degree, batch_size, num_input, num_activations, num_knots);

        return y;


    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

    TORCH_LIBRARY(cuda_kan, m) {
        m.def("kan_layer(Tensor x, Tensor wb, Tensor ws, Tensor knots, Tensor cps) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(cuda_kan, CUDA, m) {
        m.impl("kan_layer", &kan_layer);
    }

}


