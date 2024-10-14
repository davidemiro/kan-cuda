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


#include "spline.cu"

#define MAX_DIM 1024


using namespace std;


namespace cuda_kan {

    __device__ float silu(float x) {
        return 1 / (1 + expf(x * -1));

    }



    float **tensor_to_float_ptr(at::Tensor x) {
        // Ensure the tensor is of type float and has 2 dimensions (batch_size, length)
        TORCH_CHECK(x.scalar_type() == at::kFloat, "Tensor must be of type float");
        TORCH_CHECK(x.dim() == 2, "Tensor must be 2D");

        // Get dimensions of the tensor
        int64_t batch_size = x.size(0);
        int64_t length = x.size(1);

        // Get a pointer to the raw data
        float *data_ptr = x.data_ptr<float>();

        // Allocate memory for the array of float pointers (for each row)
        float **float_ptr = new float *[batch_size];

        // Fill the float_ptr array, each element points to a row in the tensor
        for (int64_t i = 0; i < batch_size; ++i) {
            float_ptr[i] = data_ptr + i * length;
        }

        return float_ptr;
    }

    __global__ void kan_activation_function(float **x, float **y, const float *wb, const float *ws, const float *cps, const float *knots, const float ***bSplineBasis, int k, int N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y;
        int z = blockIdx.z;
        float result = 0.0;
        if (i < N) {
            result = wb[j] * silu(x[z][i]) + ws[j] * b_spline(i, cps, knots, bSplineBasis, k);
            atomicAdd(&y[z][i], result);
        }

    }


    at::Tensor kan_layer(at::Tensor x, at::Tensor wb, at::Tensor ws, at::Tensor knots, at::Tensor cps) {

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


        at::Tensor x_contig = x.contiguous();
        at::Tensor wb_contig = wb.contiguous();
        at::Tensor ws_contig = ws.contiguous();
        at::Tensor cps_contig = cps.contiguous();
        at::Tensor knots_contig = knots.contiguous();

        at::Tensor y = torch::zeros({x.size(0), wb.size(0)}, wb_contig.options());

        float **x_ptr = tensor_to_float_ptr(x_contig);
        const float *wb_ptr = wb_contig.data_ptr<float>();
        const float *ws_ptr = ws_contig.data_ptr<float>();
        const float *cps_ptr = cps_contig.data_ptr<float>();
        const float *knots_ptr = knots_contig.data_ptr<float>();

        float **y_ptr = tensor_to_float_ptr(y);



        //TODO: k as argument of CUDA/CPP function
        int k = 3; //degree
        int num_input = x.size(1);
        int num_activations = wb.size(0);
        int num_threads = 1024; //max number of threads x bloc

        dim3 num_blocks(num_input / 1024, num_activations, x.size(0)); // num_input x num_activations x batch_size

        kan_activation_function<<<num_blocks, num_threads>>>(x_ptr, y_ptr, wb_ptr, ws_ptr, cps_ptr, knots_ptr, k,
                                                             num_activations);


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


