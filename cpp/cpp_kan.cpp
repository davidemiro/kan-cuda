//
// Created by davide miro on 14/09/24.
//

#include <iostream>
#include <cmath>
#include "spline.cpp"


using namespace std;

void b_spline(const float* a, float* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

float silu(float x){
    return 1/(1 + expf(x*-1));

}


void kan_activation_function(float* x, float* y, float* wb, float* ws, float* knots, float* controlPoints, int k, int N, int i, int j, int z){
    y[z][i] = wb[j]*silu(x[z][i]) + ws[j]*b_spline(x[z][i],controlPoints, knots,k);
}


at::Tensor kan_layer(at::Tensor x, at::Tensor wb, at::Tensor ws, at::Tensor knots, at::Tensor controlPoints){

    TORCH_CHECK(wb.sizes() == ws.sizes());
    TORCH_CHECK(x.dtype() == at::kFloat);
    TORCH_CHECK(wb.dtype() == at::kFloat);
    TORCH_CHECK(ws.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(wb.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(ws.device().type() == at::DeviceType::CPU);

    at::Tensor x_contig = x.contiguous();
    at::Tensor wb_contig = wb.contiguous();
    at::Tensor ws_contig = ws.contiguous();
    at::Tensor controlPoints_contig = controlPoints.contiguous();
    at::Tensor knots_contig = knots.contiguous();


    at::Tensor y = torch::empty(wb_contig.sizes(), wb_contig.options());

    const float* x_ptr = x_contig.data_ptr<float>();
    const float* wb_ptr = wb_contig.data_ptr<float>();
    const float* ws_ptr = ws_contig.data_ptr<float>();
    const float* controlPoints_ptr = controlPoints_contig.data_ptr<float>();
    const float* knots_ptr = knots_contig.data_ptr<float>();

    float* y_ptr = y.data_ptr<float>();


    int batch_size = 64 //TODO: da definire
    int N = x.size();
    int M = x.size();
    //TODO: k deve essere passato come argomento
    int k = 3;

    for(int64_t z = 0; z < batch_size(); z++) {
        for (int64_t i = 0; i < x.numel(); i++) {
            for (int64_t j = 0; j < wb.numel(); j++) {
                kan_activation_function(x_ptr, y_ptr, wb_ptr, ws_ptr, knots_ptr, controlPoints_ptr, k, N, i, j,
                                        z);
            }
        }
    }





    return y;



}

TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
m.impl("kan_activation_function", &kan_activation_function);
}
