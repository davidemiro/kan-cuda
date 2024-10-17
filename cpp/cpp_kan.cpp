//
// Created by davide miro on 14/09/24.
//

#include <iostream>
#include <cmath>
#include <torch/extension.h>
#include <ATen/ATen.h>


#include "spline.cpp"

#define MAX_DIM 1024


using namespace std;


namespace cpp_kan {
    float silu(float x) {
        return 1 / (1 + expf(x * -1));

    }


    void kan_activation_function(float **x, float **y, const float *wb, const float *ws, const float *knots,
                                 const float *cps, int k, int N, int i, int j, int z) {
        y[z][j] = y[z][j] + wb[i][j] * silu(x[z][i]) + ws[i][j] * b_spline(x[z][i], N, cps, knots, k);
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


    at::Tensor kan_layer(at::Tensor x, at::Tensor wb, at::Tensor ws, at::Tensor knots, at::Tensor cps, int degree) {
        /*
         * x : [batch_size, input_dim]
         * y : [batch_size, output_dim]
         * wb,ws: [input_dim, output_dim]
         * cps : [input_dim, num_knots]
         * knots : [num_knots]
         */

        TORCH_CHECK(wb.size(0) < MAX_DIM); //TODO: review check
        TORCH_CHECK(knots.size(0) < MAX_DIM);
        TORCH_CHECK(cps.size(0) < MAX_DIM);

        TORCH_CHECK(x.dtype() == at::kFloat);
        TORCH_CHECK(wb.dtype() == at::kFloat);
        TORCH_CHECK(ws.dtype() == at::kFloat);

        TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(wb.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(ws.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(knots.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(cps.device().type() == at::DeviceType::CPU);

        at::Tensor x_contig = x.contiguous();
        at::Tensor wb_contig = wb.contiguous();
        at::Tensor ws_contig = ws.contiguous();
        at::Tensor cps_contig = cps.contiguous();
        at::Tensor knots_contig = knots.contiguous();


        at::Tensor y = torch::zeros({x.size(0), wb.size(0)}, x_contig.options());

        float **x_ptr = tensor_to_float_ptr(x_contig);
        const float *wb_ptr = wb_contig.data_ptr<float>();
        const float *ws_ptr = ws_contig.data_ptr<float>();
        const float *cps_ptr = cps_contig.data_ptr<float>();
        const float *knots_ptr = knots_contig.data_ptr<float>();

        float **y_ptr = tensor_to_float_ptr(y);


        int num_cps = cps.size(0);

        for (int64_t z = 0; z < x.size(0); z++) {
            for (int64_t i = 0; i < x.size(1); i++) {
                for (int64_t j = 0; j < wb.size(0); j++) {
                    kan_activation_function(x_ptr, y_ptr, wb_ptr, ws_ptr, knots_ptr, cps_ptr, k, num_cps, i, j, z);
                }
            }
        }


        return y;


    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

    TORCH_LIBRARY(cpp_kan, m) {
        m.def("kan_layer(Tensor x, Tensor wb, Tensor ws, Tensor knots, Tensor cps) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(cpp_kan, CPU, m) {
        m.impl("kan_layer", &kan_layer);
    }
}