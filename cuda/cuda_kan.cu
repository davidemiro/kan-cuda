//
// Created by davide miro on 07/09/24.
//

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

#define MAX_CUDA_BLOCK 65535
#define MAX_CUDA_THREADS_X_BLOCK 1024

using namespace std;

__global__ void b_spline(const float* a, float* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

float silu(float x){
    return 1/(1 + expf(x*-1));

}
__global__ void kan_activation_function(float* x, float* y, float* wb, float ws, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    if(i < N){
        //TODO: implement b_spline
        y[i] = wb[j]*silu(x[i]) + ws[j]*b_spline(x[i])
    }

}


at::Tensor kan_activation_function(at::Tensor x, at:Tensor wb, at:Tensor ws){

    TORCH_CHECK(wb.sizes() == ws.sizes());
    TORCH_CHECK(wb.sizes() * x.size() / MAX_CUDA_THREADS_X_BLOCK < MAX_CUDA_BLOCK); //TODO: review check
    TORCH_CHECK(x.dtype() == at::kFloat);
    TORCH_CHECK(wb.dtype() == at::kFloat);
    TORCH_CHECK(ws.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(wb.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(ws.device().type() == at::DeviceType::CUDA);

    at::Tensor x_contig = x.contiguous();
    at::Tensor wb_contig = wb.contiguous();
    at::Tensor ws_contig = ws.contiguous();

    at::Tensor y = torch::empty(wb_contig.sizes(), wb_contig.options());

    const float* x_ptr = x_contig.data_ptr<float>();
    const float* wb_ptr = wb_contig.data_ptr<float>();
    const float* ws_ptr = ws_contig.data_ptr<float>();

    float* y_ptr = y.data_ptr<float>();

    int N = x.size();
    int M = x.size();

    int num_threads = 1024; //max number of threads x bloc
    dim3 num_blocks(N /1024,M) // num_input x num_activations

    kan_activation_function<<<num_blocks, num_threads>>>(x_ptr,y_ptr,wb_ptr,ws_ptr,N);


    return y;



}

TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
m.impl("kan_activation_function", &kan_activation_function);
}


