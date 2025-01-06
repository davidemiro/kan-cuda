//
// Created by davide miro on 07/09/24.
//

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <iomanip>
#include <stdio.h>

#include "spline.cu"

#define MAX_DIM 1024
#define MAX_THD 1024
#define DIMS batch_size, num_input, num_knots, degree


using namespace std;


namespace cuda_kan {

    __device__ float silu(float x) {
        return x / (1 + expf(x * -1));
    }


    __global__ void kan_activation_function(float* x, float* y, float* wb, float* ws, float* cps, float* b_spline_basis, int degree, int batch_size, int num_input, int num_activations, int num_knots) {

        int z = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.z * blockDim.z + threadIdx.z;

        if (i < num_input && z < batch_size && j < num_activations) {

            float result = 1.0;

            size_t x_idx = compute_idx(num_input, z, i);
            size_t y_idx = compute_idx(num_activations,z,j);
            size_t w_idx = compute_idx(num_activations, i, j);

            result = spline(cps, b_spline_basis, z, i, j, DIMS) * ws[w_idx] + silu(x[x_idx]) * wb[w_idx];
            atomicAdd(&y[y_idx], result);
        }

    }


    at::Tensor kan_layer(at::Tensor x, at::Tensor wb, at::Tensor ws, at::Tensor knots, at::Tensor cps, int64_t degree) {
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

        TORCH_CHECK(x.dtype() == torch::kFloat);
        TORCH_CHECK(wb.dtype() == torch::kFloat);
        TORCH_CHECK(ws.dtype() == torch::kFloat);

        TORCH_INTERNAL_ASSERT(x.device().type() == torch::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(wb.device().type() == torch::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(ws.device().type() == torch::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(knots.device().type() == torch::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(cps.device().type() == torch::DeviceType::CUDA);



        int batch_size = x.size(0);
        int num_input = x.size(1);
        int num_activations = wb.size(1);
        int num_knots = cps.size(1);

        torch::Tensor x_contig = x.contiguous();
        torch::Tensor wb_contig = wb.contiguous();
        torch::Tensor ws_contig = ws.contiguous();
        torch::Tensor cps_contig = cps.contiguous();
        torch::Tensor knots_contig = knots.contiguous();
        torch::Tensor y = torch::zeros({x.size(0), wb.size(1)}, x.options()).contiguous();
        torch::Tensor b_spline_basis = torch::empty({batch_size,num_input,num_knots,degree}, wb.options()).contiguous();

        float *x_ptr = x_contig.data_ptr<float>();
        float *cps_ptr = cps_contig.data_ptr<float>();
        float *wb_ptr = wb_contig.data_ptr<float>();
        float *ws_ptr = ws_contig.data_ptr<float>();
        float *knots_ptr = knots_contig.data_ptr<float>();
        float *y_ptr = y.data_ptr<float>();
        float *b_spline_basis_ptr = b_spline_basis.data_ptr<float>();

        for(int i = 0; i < 9; i++){
            printf("%d\n",x_ptr[i]);
        }


        dim3 gridDim(ceil(batch_size/32), ceil(num_input/32));
        dim3 blockDim(32,32,1);
        b_spline_base<<<gridDim, blockDim>>>(b_spline_basis_ptr, x_ptr, DIMS, knots_ptr);
        cudaDeviceSynchronize();

        /*
        dim3 threads_block(num_input,num_activations); // batch_size x num_input x num_activations
        kan_activation_function<<<batch_size, threads_block>>>(x_ptr, y_ptr, wb_ptr, ws_ptr, cps_ptr, b_spline_basis_ptr, degree, batch_size, num_input, num_activations, num_knots);
        cudaDeviceSynchronize();
        */

        return y;


    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("kan_layer", &kan_layer, "kan_layer");
    }

}


