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
#define CHUNK 32
#define DIMS batch_size, num_input, num_knots, degree


using namespace std;


namespace cuda_kan {

    __device__ float silu(float x) {
        return x / (1 + expf(x * -1));
    }


    __global__ void kan_activation_function_chunk(float* x, float* y, float* wb, float* ws, float* cps, float* b_spline_basis, int degree, int batch_size, int num_input, int num_activations, int num_knots){
        int z = blockIdx.x;
        int i = threadIdx.x;
        int  w_idx, stride;

        float result = 0.0;
        extern __shared__ float* cache_ptr;
        float* x_l;
        float* bsp_l;

        printf("blockIdx: %d blockDimx: %d blockIdy: %d blockDimy: %d threadIdx: %d",blockIdx.x, blockDim.x, blockIdx.y, blockDim.y, threadIdx.x);
        printf("num_input: %d", num_input);

        if(threadIdx.x + blockIdx.y * blockDim.y < num_input) {

            printf("C");
            bsp_l = cache_ptr;
            x_l = &bsp_l[num_knots * num_input];

            //load b_spline_ptr(1, 1, CHUNK, num_knots)
            for (int j = threadIdx.x; j < num_knots; j += CHUNK) {
                bsp_l[compute_idx(i,j, num_knots)] = b_spline_basis[compute_idx_base(z, i, j, degree, DIMS)];
            }
            printf("D");
            __syncthreads();

            printf("E");
            //load x(CHUNK)
            x_l[i] = x[compute_idx(z, i + blockIdx.y * CHUNK, num_input)];
            __syncthreads();


            printf("F");
            for (int j = 0; j < num_activations; j += CHUNK) {

                stride = fminf(CHUNK, num_activations - j);
                for (int k = 0; k < stride; k++) {
                    w_idx = compute_idx(i, j + k, num_activations);
                    result = spline(cps, bsp_l, 0, i, j + k, 0, num_input, num_knots, 0) * ws[w_idx] + silu(x_l[i]) * wb[w_idx];
                    atomicAdd(&y[compute_idx(z,j + k, num_activations)], result);
                }
            }
        }
    }

    __global__ void kan_activation_function(float* x, float* y, float* wb, float* ws, float* cps, float* b_spline_basis, int degree, int batch_size, int num_input, int num_activations, int num_knots) {

        int z = blockIdx.x;
        int i = threadIdx.x;
        int j = threadIdx.y;

        if (i < num_input && z < batch_size && j < num_activations) {



            float result = 1.0;

            size_t x_idx = compute_idx(z, i, num_input);
            size_t y_idx = compute_idx(z,j, num_activations);
            size_t w_idx = compute_idx(i, j, num_activations);

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
         * knots : [num_knots]
         */

        TORCH_CHECK(wb.size(0) <= MAX_DIM); //TODO: review check
        TORCH_CHECK(knots.size(0) <= MAX_DIM);
        TORCH_CHECK(cps.size(0) <= MAX_DIM);

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
        torch::Tensor b_spline_basis = torch::empty({degree + 1,batch_size,num_input,num_knots}, wb.options()).contiguous();

        float *x_ptr = x_contig.data_ptr<float>();
        float *cps_ptr = cps_contig.data_ptr<float>();
        float *wb_ptr = wb_contig.data_ptr<float>();
        float *ws_ptr = ws_contig.data_ptr<float>();
        float *knots_ptr = knots_contig.data_ptr<float>();
        float *y_ptr = y.data_ptr<float>();
        float *b_spline_basis_ptr = b_spline_basis.data_ptr<float>();


        int cache_size = num_input * num_knots * sizeof(float);
        b_spline_base<<<batch_size,num_input,cache_size>>>(b_spline_basis_ptr, x_ptr, DIMS, knots_ptr);
        cudaDeviceSynchronize();


        if (num_input <= CHUNK && num_activations <= CHUNK){
            dim3 thread_blocks(num_input, num_activations);
            kan_activation_function<<<batch_size,thread_blocks>>>(x_ptr, y_ptr, wb_ptr, ws_ptr, cps_ptr, b_spline_basis_ptr, degree, batch_size, num_input, num_activations, num_knots);

        }else {
            cache_size = CHUNK * CHUNK * num_knots * sizeof(float);
            dim3 grid_blocks(batch_size, ceil(num_input / CHUNK));
            printf("B");
            printf(grid_blocks);
            kan_activation_function_chunk<<<grid_blocks, CHUNK, cache_size>>>(x_ptr, y_ptr, wb_ptr, ws_ptr, cps_ptr,
                                                                              b_spline_basis_ptr, degree, batch_size,
                                                                              num_input, num_activations, num_knots);
        }

        cudaDeviceSynchronize();
        return y;


    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("kan_layer", &kan_layer, "kan_layer");
    }

}


