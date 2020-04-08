#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <algorithm>

#include "deepspeech.cu"
#include "cuda_wrappers.cpp"
#include "configuration.h"
#define GPU1_ID 0
#define GPU2_ID 1
#define GPU3_ID 2
#define ENABLE_PEER_ACCESS 1

#define GEMM_BATCH 10

#define WEIGHTS_SIZE 33554432
#define BIASES_SIZE 32768

#define WEIGHTS2_SIZE 33554432
#define BIASES2_SIZE 32768

#define X_SIZE 13107200
#define Y_SIZE 6553600

#define NB_TIMES 1000
bool first_execution = true;
using namespace std;
void deepspeech2(float *buf_Weights_cpu, float *buf_biases_cpu,
                 float *buf_Weights2_cpu, float *buf_biases2_cpu,
                 float *buf_x_cpu, float *buf_y_cpu,
                 float *time_start, float *time_end)
                 {

/** GPU1 INITIALIZATIONS */
    wrapper_cuda_set_device(GPU1_ID);
    if (first_execution && ENABLE_PEER_ACCESS){
      wrapper_cuda_device_enable_peer_access(GPU2_ID, 0);
      wrapper_cuda_device_enable_peer_access(GPU3_ID, 0);
    }
    float *buf_x_gpu1, *buf_weights_gpu1, *buf_biases_gpu1, *buf_tmp_gpu1, *buf_weights_T_gpu1, *buf_h_gpu1, *buf_c_gpu1;
    buf_x_gpu1 = (float *) wrapper_cuda_malloc((uint64_t)13107200);
    buf_weights_gpu1 = (float *) wrapper_cuda_malloc((uint64_t)33554432);
    buf_biases_gpu1 = (float *) wrapper_cuda_malloc((uint64_t)32768);
    buf_tmp_gpu1 = (float *) wrapper_cuda_malloc((uint64_t)52428800);
    buf_weights_T_gpu1 = (float *) wrapper_cuda_malloc((uint64_t)33554432);
    buf_h_gpu1 = (float *) wrapper_cuda_malloc((uint64_t)66191360);
    buf_c_gpu1 = (float *) wrapper_cuda_malloc((uint64_t)52953088);

    wrapper_cuda_memcpy_to_device(buf_weights_gpu1, buf_Weights_cpu, (uint64_t)33554432);
    wrapper_cuda_memcpy_to_device(buf_biases_gpu1, buf_biases_cpu, (uint64_t)32768);

/** GPU2 INITIALIZATIONS */
    wrapper_cuda_set_device(GPU2_ID);
    if(first_execution && ENABLE_PEER_ACCESS){
      wrapper_cuda_device_enable_peer_access(GPU1_ID, 0);
      wrapper_cuda_device_enable_peer_access(GPU3_ID, 0);
    }
    float *buf_weights_gpu2, *buf_biases_gpu2, *buf_tmp_gpu2, *buf_weights_T_gpu2, *buf_h_gpu2, *buf_c_gpu2, *buf_weights2_gpu2, *buf_biases2_gpu2, *buf_weights2_T_gpu2;
    buf_weights_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)33554432);
    buf_biases_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)32768);
    buf_tmp_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)52428800);

    buf_weights_T_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)33554432);
    buf_h_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)66191360);
    buf_c_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)52953088);

    buf_weights2_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)33554432);
    buf_biases2_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)32768);
    buf_weights2_T_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)33554432);

    wrapper_cuda_memcpy_to_device(buf_weights_gpu2, buf_Weights_cpu, (uint64_t)33554432);
    wrapper_cuda_memcpy_to_device(buf_biases_gpu2, buf_biases_cpu, (uint64_t)32768);
    wrapper_cuda_memcpy_to_device(buf_weights2_gpu2, buf_Weights2_cpu, (uint64_t)33554432);
    wrapper_cuda_memcpy_to_device(buf_biases2_gpu2, buf_biases2_cpu, (uint64_t)32768);

/** GPU3 INITIALIZATIONS */
    wrapper_cuda_set_device(GPU3_ID);
    float *buf_y_gpu = (float *) wrapper_cuda_malloc((uint64_t)6553600);

    if(first_execution && ENABLE_PEER_ACCESS){
      wrapper_cuda_device_enable_peer_access(GPU1_ID, 0);
      wrapper_cuda_device_enable_peer_access(GPU2_ID, 0);
    }

    float *buf_weights2_gpu3, *buf_biases2_gpu3, *buf_tmp2_gpu3, *buf_weights2_T_gpu3, *buf_h2_gpu3, *buf_c2_gpu3;
    buf_weights2_gpu3 = (float *) wrapper_cuda_malloc((uint64_t)33554432);
    buf_biases2_gpu3 = (float *) wrapper_cuda_malloc((uint64_t)32768);
    buf_tmp2_gpu3 = (float *) wrapper_cuda_malloc((uint64_t)26214400);

    buf_weights2_T_gpu3 = (float *) wrapper_cuda_malloc((uint64_t)33554432);
    buf_h2_gpu3 = (float *) wrapper_cuda_malloc((uint64_t)33423360);
    buf_c2_gpu3 = (float *) wrapper_cuda_malloc((uint64_t)26738688);

    wrapper_cuda_memcpy_to_device(buf_weights2_gpu3, buf_Weights2_cpu, (uint64_t)33554432);
    wrapper_cuda_memcpy_to_device(buf_biases2_gpu3, buf_biases2_cpu, (uint64_t)32768);

    time_start[0] = get_time(0);

    wrapper_cuda_set_device(GPU1_ID);
    wrapper_cuda_memcpy_to_device(buf_x_gpu1, buf_x_cpu, (uint64_t)13107200);

    _kernel_0_wrapper(buf_weights_gpu1, buf_weights_T_gpu1);
    _kernel_1_wrapper(buf_h_gpu1);
    _kernel_2_wrapper(buf_c_gpu1);
    _kernel_3_wrapper(buf_h_gpu1, buf_x_gpu1);

    wrapper_cuda_set_device(GPU2_ID);
    _kernel_4_wrapper(buf_weights_gpu2, buf_weights_T_gpu2);
    _kernel_5_wrapper(buf_weights2_T_gpu2, buf_weights2_gpu2);
    _kernel_6_wrapper(buf_h_gpu2);
    _kernel_7_wrapper(buf_c_gpu2);
    _kernel_8_wrapper(buf_h_gpu2);
    _kernel_9_wrapper(buf_c_gpu2);

    wrapper_cuda_set_device(GPU3_ID);
    _kernel_10_wrapper(buf_weights2_gpu3, buf_weights2_T_gpu3);
    _kernel_11_wrapper(buf_h2_gpu3);
    _kernel_12_wrapper(buf_c2_gpu3);

    for(int c1=0; c1<13; c1++){
        for(int c3  =max(c1-4, 0); c3 < min(c1, 8) - max(c1-4, 0) +1; c3++){
            // std::cout << "s0 = " << c1 << " , l = " << c3 << std::endl;
            if(c3 < 3){
                wrapper_cuda_set_device(GPU1_ID);
                for (int c5 = 0; c5 < 2; c5++) {
                    wrapper_cublas_sgemm(buf_h_gpu1, buf_weights_T_gpu1, buf_tmp_gpu1, (uint64_t)640, (uint64_t)2048, (uint64_t)512, 1.000000f, 0.000000f, (uint64_t)0, (uint64_t)0, (uint64_t)0, uint64_t(((((c3*101) + ((((c1 - c3)*2) + c5)*10))*32768) + 32768)), uint64_t((c3*2097152)), uint64_t(((((c1 - c3)*2) + c5)*1310720)), (uint32_t)0, (uint32_t)0);
                    for (int c7 = 0; c7 < 10; c7++) {
                        wrapper_cublas_sgemm(buf_h_gpu1, buf_weights_T_gpu1, buf_tmp_gpu1, (uint64_t)64, (uint64_t)2048, (uint64_t)512, 1.000000f, 1.000000f, (uint64_t)0, (uint64_t)0, (uint64_t)0, uint64_t(((((c3*101) + (((((c1 - c3)*10) + c7)*2) + c5))*32768) + 3309568)), uint64_t(((c3*2097152) + 1048576)), uint64_t(((((((c1 - c3)*10) + c7)*2) + c5)*131072)), (uint32_t)0, (uint32_t)0);
                        _kernel_13_wrapper(c1, c3, c5, c7, buf_biases_gpu1, buf_c_gpu1, buf_h_gpu1, buf_tmp_gpu1);
                    }
                }
            }
            wrapper_cuda_stream_synchronize(0);
            // Memcpy GPU0 to GPU1
            #define offset1 ((c3 + 1) * (SEQ_LENGTH + 1) * BATCH_SIZE * FEATURE_SIZE + (2 * c1 * GEMM_BATCH + 1) * BATCH_SIZE * FEATURE_SIZE)
            #define count1 (2 * GEMM_BATCH * BATCH_SIZE * FEATURE_SIZE)
            if (c3 == 2 && ENABLE_PEER_ACCESS)
                wrapper_cuda_memcpy_peer((buf_h_gpu2 + offset1), GPU2_ID, (buf_h_gpu1 + offset1), GPU1_ID, (uint64_t) count1);
            //

            if (((c3 < 6) && (2 < c3))) {
                wrapper_cuda_set_device(GPU2_ID);
            }
            if ((c3 == 3))
              for (int c5 = 0; c5 < 2; c5++) {
                  wrapper_cublas_sgemm(buf_h_gpu2, buf_weights_T_gpu2, buf_tmp_gpu2, (uint64_t)640, (uint64_t)2048, (uint64_t)512, 1.000000f, 0.000000f, (uint64_t)0, (uint64_t)0, (uint64_t)0, uint64_t(((((c1*2) + c5)*327680) + 7995392)), (uint64_t)6291456, uint64_t(((((c1*2) + c5)*1310720) + -7864320)), (uint32_t)0, (uint32_t)0);

                  for (int c7 = 0; c7 < 10; c7++) {
                      wrapper_cublas_sgemm(buf_h_gpu2, buf_weights_T_gpu2, buf_tmp_gpu2, (uint64_t)64, (uint64_t)2048, (uint64_t)512, 1.000000f, 1.000000f, (uint64_t)0, (uint64_t)0, (uint64_t)0, uint64_t(((((((c1*10) + c7)*2) + c5)*32768) + 11272192)), (uint64_t)7340032, uint64_t(((((((c1*10) + c7)*2) + c5)*131072) + -7864320)), (uint32_t)0, (uint32_t)0);
                      _kernel_14_wrapper(c1, c3, c5, c7, buf_biases_gpu2, buf_c_gpu2, buf_h_gpu2, buf_tmp_gpu2);
                  }
              }

            wrapper_cuda_stream_synchronize(0);
            if ((c3 == 4))
              _kernel_15_wrapper(c1, c3, buf_h_gpu2);
            if ((c3 == 5)) {
                wrapper_cublas_sgemm(buf_h_gpu2, buf_weights2_T_gpu2, buf_tmp_gpu2, (uint64_t)640, (uint64_t)2048, (uint64_t)512, 1.000000f, 0.000000f, (uint64_t)0, (uint64_t)0, (uint64_t)0, uint64_t(((c1*327680) + -1605632)), (uint64_t)0, uint64_t(((c1*1310720) + -6553600)), (uint32_t)0, (uint32_t)0);
            }
            if ((c3 == 5))
              for (int c5 = 0; c5 < 10; c5++) {
                  wrapper_cublas_sgemm(buf_h_gpu2, buf_weights2_T_gpu2, buf_tmp_gpu2, (uint64_t)64, (uint64_t)2048, (uint64_t)512, 1.000000f, 1.000000f, (uint64_t)0, (uint64_t)0, (uint64_t)0, uint64_t(((((c1*10) + c5)*32768) + 32768)), (uint64_t)1048576, uint64_t(((((c1*10) + c5)*131072) + -6553600)), (uint32_t)0, (uint32_t)0);

                  _kernel_16_wrapper(c1, c3, c5, buf_biases2_gpu2, buf_c_gpu2, buf_h_gpu2, buf_tmp_gpu2);
              }
            wrapper_cuda_stream_synchronize(0);

            // Memcpy GPU0 to GPU1
            #define offset2 ((c3 + 1 - c1) * (SEQ_LENGTH + 1) * BATCH_SIZE * FEATURE_SIZE + (c1 * GEMM_BATCH + 1) * BATCH_SIZE * FEATURE_SIZE)
            #define offset3 ((c3 + 1 - c1) * (SEQ_LENGTH/2 + 1) * BATCH_SIZE * FEATURE_SIZE + (c1 * GEMM_BATCH + 1) * BATCH_SIZE * FEATURE_SIZE)
            #define count2_3 (GEMM_BATCH * BATCH_SIZE * FEATURE_SIZE)
            if (c3 == 5 && ENABLE_PEER_ACCESS)
                wrapper_cuda_memcpy_peer((buf_h2_gpu3 + offset3), GPU3_ID, (buf_h_gpu2 + offset2), GPU2_ID, (uint64_t) count2_3);

            //

            if ((5 < c3)) {
                wrapper_cuda_set_device(GPU3_ID);
                wrapper_cublas_sgemm(buf_h2_gpu3, buf_weights2_T_gpu3, buf_tmp2_gpu3, (uint64_t)640, (uint64_t)2048, (uint64_t)512, 1.000000f, 0.000000f, (uint64_t)0, (uint64_t)0, (uint64_t)0, uint64_t(((((c3*51) + ((c1 - c3)*10))*32768) + -8323072)), uint64_t(((c3*2097152) + -10485760)), uint64_t(((c1 - c3)*1310720)), (uint32_t)0, (uint32_t)0);

                for (int c5 = 0; c5 < 10; c5++) {
                    wrapper_cublas_sgemm(buf_h2_gpu3, buf_weights2_T_gpu3, buf_tmp2_gpu3, (uint64_t)64, (uint64_t)2048, (uint64_t)512, 1.000000f, 1.000000f, (uint64_t)0, (uint64_t)0, (uint64_t)0, uint64_t(((((c3*51) + (((c1 - c3)*10) + c5))*32768) + -6684672)), uint64_t(((c3*2097152) + -9437184)), uint64_t(((((c1 - c3)*10) + c5)*131072)), (uint32_t)0, (uint32_t)0);
                    _kernel_17_wrapper(c1, c3, c5, buf_biases2_gpu3, buf_c2_gpu3, buf_h2_gpu3, buf_tmp2_gpu3);
                }
            }
            wrapper_cuda_stream_synchronize(0);
        }
    }
    wrapper_cuda_set_device(GPU3_ID);
    _kernel_18_wrapper(buf_y_gpu, buf_h2_gpu3);
    wrapper_cuda_stream_synchronize(0);
    wrapper_cuda_memcpy_to_host(buf_y_cpu, buf_y_gpu, (uint64_t)6553600);

    //// Wait for all threads
    wrapper_cuda_set_device(GPU3_ID);
    wrapper_cuda_stream_synchronize(0);

    wrapper_cuda_set_device(GPU2_ID);
    wrapper_cuda_stream_synchronize(0);

    wrapper_cuda_set_device(GPU1_ID);
    wrapper_cuda_stream_synchronize(0);

/////////////// Free buffers
    wrapper_cuda_set_device(GPU1_ID);
    wrapper_cuda_free(buf_x_gpu1);
    wrapper_cuda_free(buf_weights_gpu1);
    wrapper_cuda_free(buf_biases_gpu1);
    wrapper_cuda_free(buf_tmp_gpu1);
    wrapper_cuda_free(buf_weights_T_gpu1);
    wrapper_cuda_free(buf_h_gpu1);
    wrapper_cuda_free(buf_c_gpu1);

    wrapper_cuda_set_device(GPU2_ID);
    wrapper_cuda_free(buf_weights_gpu2);
    wrapper_cuda_free(buf_biases_gpu2);
    wrapper_cuda_free(buf_tmp_gpu2);
    wrapper_cuda_free(buf_weights_T_gpu2);
    wrapper_cuda_free(buf_h_gpu2);
    wrapper_cuda_free(buf_c_gpu2);
    wrapper_cuda_free(buf_weights2_gpu2);
    wrapper_cuda_free(buf_biases2_gpu2);
    wrapper_cuda_free(buf_weights2_T_gpu2);

    wrapper_cuda_set_device(GPU3_ID);
    wrapper_cuda_free(buf_weights2_gpu3);
    wrapper_cuda_free(buf_biases2_gpu3);
    wrapper_cuda_free(buf_tmp2_gpu3);
    wrapper_cuda_free(buf_weights2_T_gpu3);
    wrapper_cuda_free(buf_h2_gpu3);
    wrapper_cuda_free(buf_c2_gpu3);
    wrapper_cuda_free(buf_y_gpu);

    time_end[0] = get_time(0);
    first_execution=false;
}

int main(){
    std::cout<< "Hello world " << std::endl;
    float *buf_Weights_cpu, *buf_biases_cpu,
          *buf_Weights2_cpu, *buf_biases2_cpu,
          *buf_x_cpu, *buf_y_cpu,
          *time_start, *time_end;
    buf_Weights_cpu = (float *)malloc(WEIGHTS_SIZE * sizeof(float));
    buf_biases_cpu = (float *)malloc(BIASES_SIZE * sizeof(float));

    buf_Weights2_cpu = (float *)malloc(WEIGHTS2_SIZE * sizeof(float));
    buf_biases2_cpu = (float *)malloc(BIASES2_SIZE * sizeof(float));

    buf_x_cpu = (float *)malloc(X_SIZE * sizeof(float));
    buf_y_cpu = (float *)malloc(Y_SIZE * sizeof(float));

    time_start = (float *)malloc(sizeof(float));
    time_end = (float *)malloc(sizeof(float));

    std::srand(0);
    for (int i = 0; i < WEIGHTS_SIZE; i++)
      buf_Weights_cpu[i] = (std::rand() % 200 - 100) / 100.;
    for (int i = 0; i < BIASES_SIZE; i++)
      buf_biases_cpu[i] = (std::rand() % 200 - 100) / 100.;

    for (int i = 0; i < WEIGHTS2_SIZE; i++)
      buf_Weights2_cpu[i] = (std::rand() % 200 - 100) / 100.;
    for (int i = 0; i < BIASES2_SIZE; i++)
      buf_biases2_cpu[i] = (std::rand() % 200 - 100) / 100.;

    for (int i = 0; i < X_SIZE; i++)
      buf_x_cpu[i] = (std::rand() % 200 - 100) / 100.;

    for (int j = 0; j < 10; j++)
      std::cout << buf_y_cpu[j] << ", ";
    float time_mean;
    float time_std;
    for (int i = 0; i<NB_TIMES; i++){
      deepspeech2(buf_Weights_cpu, buf_biases_cpu,
                  buf_Weights2_cpu, buf_biases2_cpu,
                  buf_x_cpu, buf_y_cpu,
                  time_start, time_end);
      std::cout << i << " time : "<<(time_end[0] - time_start[0])<< std::endl;
      time_mean += (time_end[0] - time_start[0]);
      time_std += (time_end[0] - time_start[0])* (time_end[0] - time_start[0]);
      for (int j = 0; j < 10; j++)
        std::cout << buf_y_cpu[j] << ", ";
      std::cout << std::endl;
    }
    time_mean /= NB_TIMES;
    time_std /= NB_TIMES;
    time_std = time_std - time_mean * time_mean;
    std::cout << "Results " << std::endl;
    std::cout << "-----------------------------" << std::endl;
    std::cout << "Time mean : " << time_mean << " / Time variance : " << time_std << std::endl;

    return 0;
}
