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


#include "cpp/spline.cpp"

#define MAX_DIM 1024


using namespace std;

__global__ void b_spline(const float* a, float* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

float silu(float x){
    return 1/(1 + expf(x*-1));

}
__global__ void kan_activation_function(float* x, float* y, float* wb, float* ws, float* cps, float* knots, int k, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int z = blockIdx.z;
    if(i < N){
        y[z][j] = y[z][j] + wb[j]*silu(x[i]) + ws[j]*b_spline(x[z][i], cps, knots, k);
    }

}


at::Tensor kan_layer(at::Tensor x, at::Tensor wb, at::Tensor ws, at::Tensor knots, at::Tensor cps){

    TORCH_CHECK(wb.sizes() == ws.sizes());
    TORCH_CHECK(wb.sizes() < MAX_DIM); //TODO: review check
    TORCH_CHECK(knots.sizes() < MAX_DIM);
    TORCH_CHECK(cps.sizes() < MAX_DIM);


    TORCH_CHECK(x.dtype() == at::kFloat);
    TORCH_CHECK(wb.dtype() == at::kFloat);
    TORCH_CHECK(ws.dtype() == at::kFloat);
    TORCH_CHECK(wb.dtype() == at::knots);
    TORCH_CHECK(ws.dtype() == at::cps);

    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(wb.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(ws.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(knots.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(cps.device().type() == at::DeviceType::CUDA);


    at::Tensor x_contig = x.contiguous();
    at::Tensor wb_contig = wb.contiguous();
    at::Tensor ws_contig = ws.contiguous();
    at::Tensor cps_contig = controlPoints.contiguous();
    at::Tensor knots_contig = knots.contiguous();

    at::Tensor y = torch::zeros({x.size(0),wb.size(0)}, wb_contig.options());

    const float* x_ptr = x_contig.data_ptr<float>();
    const float* wb_ptr = wb_contig.data_ptr<float>();
    const float* ws_ptr = ws_contig.data_ptr<float>();
    const float* cps_ptr = controlPoints_contig.data_ptr<float>();
    const float* knots_ptr = knots_contig.data_ptr<float>();

    float* y_ptr = y.data_ptr<float>();

    int N = x.size();
    int M = x.size();
    //TODO: k deve essere passato come argomento
    int k = 3

    int num_threads = 1024; //max number of threads x bloc
    dim3 num_blocks(N /1024,M, x.size(0)) // num_input x num_activations x batch_size

    kan_activation_function<<<num_blocks, num_threads>>>(x_ptr,y_ptr,wb_ptr,ws_ptr,cps_ptr,knots_ptr,k,N);


    return y;



}

TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
m.impl("kan_layer", &kan_layer);
}


