//
// Created by davide miro on 14/09/24.
//

#include <iostream>
#include <cmath>
#include <torch/extension.h>
#include <ATen/ATen.h>



#define MAX_DIM 1024
#define MAX_THD 1024


using namespace std;


namespace cpp_kan {
    float silu(float x) {
        return 1 / (1 + expf(x * -1));

    }

    void b_spline_base(float**** b_spline_basis, float** x, int batch_size, int num_input, int num_knots, int degree,float** knots) {
        /*
         * z : z-th batch element
         * i : i-th element of the input
         * j : j-th knot
         * k : degree
         */


        float t;
        double leftTerm = 0.0;
        double rightTerm = 0.0;

        for(int z = 0; z < batch_size; z++) {
            for (int i = 0; i < num_input; i++) {
                for (int d = 0; d < degree; d++) {
                    for (int j = 0; j < num_knots; j++) {

                        t = x[z][i];
                        if (d == 0) {
                            // Base case: piecewise constant function (degree 0)
                            if (knots[i][j] <= t && t < knots[i][j + 1]) {
                                b_spline_basis[z][i][j][d] = 1.0;
                            } else {
                                b_spline_basis[z][i][j][d] = 0.0;
                            }
                        } else {


                            // Check the left term (avoid division by zero)
                            if (knots[i][j + d] != knots[i][j]) {
                                leftTerm = (t - knots[i][j]) / (knots[i][j + d] - knots[i][j]) *
                                           b_spline_basis[z][i][j][d - 1];
                            }

                            // Check the right term (avoid division by zero)
                            if (knots[i][j + d + 1] != knots[i][j + 1]) {
                                rightTerm = (knots[i][j + d + 1] - t) / (knots[i][j + d + 1] - knots[i][j + 1]) *
                                            b_spline_basis[z][i][j + 1][d - 1];
                            }

                            b_spline_basis[z][i][j][d] = leftTerm + rightTerm;
                        }
                    }
                }
            }
        }
    }

    float spline(float** cps, float**** b_spline_basis, int z, int i, int j, int d, int num_knots) {
        /*
         * z : z-th batch element
         * i : i-th element of the input
         * j : j-th activation function
         * k : k-th knot
         * d : degree
         */
        float result = 0.0;
        for(int k = 0; k < num_knots; k++){
            result += cps[j][k] * b_spline_basis[z][i][k][d];
        }

        return result;
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




    at::Tensor kan_layer(at::Tensor x, at::Tensor wb, at::Tensor ws, at::Tensor knots, at::Tensor cps, int64_t degree) {
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

        TORCH_CHECK(x.dtype() == at::kFloat);
        TORCH_CHECK(wb.dtype() == at::kFloat);
        TORCH_CHECK(ws.dtype() == at::kFloat);

        TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(wb.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(ws.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(knots.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(cps.device().type() == at::DeviceType::CPU);

        int batch_size = x.size(0);
        int num_input = x.size(1);
        int num_activations = wb.size(0);
        int num_knots = cps.size(1);

        at::Tensor x_contig = x.contiguous();
        at::Tensor wb_contig = wb.contiguous();
        at::Tensor ws_contig = ws.contiguous();
        at::Tensor cps_contig = cps.contiguous();
        at::Tensor knots_contig = knots.contiguous();


        at::Tensor y = torch::zeros({x.size(0), wb.size(0)}, x_contig.options());
        at::Tensor b_spline_basis = torch::empty({batch_size,num_input,num_activations,degree}, wb_contig.options());

        float **x_ptr = x_contig.data_ptr<float*>();
        float **cps_ptr = cps_contig.data_ptr<float*>();
        float **wb_ptr = wb_contig.data_ptr<float*>();
        float **ws_ptr = ws_contig.data_ptr<float*>();
        float **knots_ptr = knots_contig.data_ptr<float*>();

        float **y_ptr = y.data_ptr<float*>();
        float ****b_spline_basis_ptr = b_spline_basis.data_ptr<float***>();

        b_spline_base(b_spline_basis_ptr, x_ptr, batch_size, num_input, num_activations, degree, knots_ptr);

        kan_activation_function(x_ptr, y_ptr, wb_ptr, ws_ptr, cps_ptr, b_spline_basis_ptr, degree, batch_size, num_input, num_activations, num_knots);

        return y;



    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

    TORCH_LIBRARY(cpp_kan, m) {
        m.def("kan_layer(Tensor x, Tensor wb, Tensor ws, Tensor knots, Tensor cps, int64_t degree) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(cpp_kan, CPU, m) {
        m.impl("kan_layer", &kan_layer);
    }
}