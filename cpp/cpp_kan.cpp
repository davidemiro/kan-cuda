//
// Created by davide miro on 14/09/24.
//

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <iomanip>
#include <cmath>


#include "spline.cpp"

#define MAX_DIM 1024
#define MAX_THD 1024


using namespace std;


namespace cpp_kan {
    float silu(float x) {
        return 1 / (1 + expf(x * -1));

    }


    void kan_activation_function(float **x, float **y, float **wb, float **ws, float **cps, float ****b_spline_basis, int k, int batch_size, int num_inputs, int num_activations, int num_knots) {

        for(int z = 0; z < batch_size; z++){
            for(int i = 0; i < num_inputs; i++) {
                for(int j = 0; j < num_activations; j++){
                    y[z][j] = y[z][j] + spline(cps, b_spline_basis, z, i, j, k, num_knots) + wb[i][j] * silu(x[z][i]) + ws[i][j];
                }
            }
        }

    }


    torch::Tensor kan_layer(torch::Tensor x, torch::Tensor wb, torch::Tensor ws, torch::Tensor knots, torch::Tensor cps, int64_t degree) {
        /*
         * x : [batch_size, input_dim]
         * y : [batch_size, output_dim]
         * wb,ws: [input_dim, output_dim]
         * cps : [input_dim, num_knots]
         * knots : [input_dim, num_knots]
         */

        TORCH_CHECK(wb.size(0) < MAX_DIM); //TODO: review check
        TORCH_CHECK(knots.size(0) < MAX_DIM);
        TORCH_CHECK(cps.size(0) < MAX_DIM);

        TORCH_CHECK(x.dtype() == torch::kFloat);
        TORCH_CHECK(wb.dtype() == torch::kFloat);
        TORCH_CHECK(ws.dtype() == torch::kFloat);

        TORCH_INTERNAL_ASSERT(x.device().type() == torch::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(wb.device().type() == torch::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(ws.device().type() == torch::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(knots.device().type() == torch::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(cps.device().type() == torch::DeviceType::CPU);

        int batch_size = x.size(0);
        int num_input = x.size(1);
        int num_activations = wb.size(0);
        int num_knots = cps.size(1);

        torch::Tensor x_contig = x.contiguous();
        torch::Tensor wb_contig = wb.contiguous();
        torch::Tensor ws_contig = ws.contiguous();
        torch::Tensor cps_contig = cps.contiguous();
        torch::Tensor knots_contig = knots.contiguous();


        torch::Tensor y = torch::zeros({x.size(0), wb.size(0)}, x_contig.options());
        torch::Tensor b_spline_basis = torch::empty({batch_size,num_input,num_activations,degree}, wb_contig.options());


        float **x_ptr = (float**) x_contig.data_ptr<float>();
        float **cps_ptr = (float**) cps_contig.data_ptr<float>();
        float **wb_ptr = (float**) wb_contig.data_ptr<float>();
        float **ws_ptr = (float**) ws_contig.data_ptr<float>();
        float **knots_ptr = (float**) knots_contig.data_ptr<float>();

        float **y_ptr = (float**) y.data_ptr<float>();
        float ****b_spline_basis_ptr = (float****) b_spline_basis.data_ptr<float>();


        b_spline_base(b_spline_basis_ptr, x_ptr, batch_size, num_input, num_activations, degree, knots_ptr);

        kan_activation_function(x_ptr, y_ptr, wb_ptr, ws_ptr, cps_ptr, b_spline_basis_ptr, degree, batch_size, num_input, num_activations, num_knots);


        return x;



    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("kan_layer", &kan_layer, "kan_layer");
    }

}