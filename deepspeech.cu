#include <stdint.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "configuration.h"
static __global__ void _kernel_0(float *buf_weights_gpu1, float *buf_weights_T_gpu1)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);
	for (int32_t c9 = 0; (c9 < NUM_LAYERS); (c9 += 1))
	{
		for (int32_t c11 = 0; (c11 <= 1); (c11 += 1))
		{
			buf_weights_T_gpu1[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + (c11 * ((1 * (4 * FEATURE_SIZE)) * FEATURE_SIZE))) + (c9 * (((1 * 4 * FEATURE_SIZE) * FEATURE_SIZE) * 2)))] = buf_weights_gpu1[((((0 + (((32 * __by__) + __ty__) * 1)) + (((32 * __bx__) + __tx__) * (1 * FEATURE_SIZE))) + (c11 * ((1 * FEATURE_SIZE) * 4 * FEATURE_SIZE))) + (c9 * (((1 * FEATURE_SIZE) * 4 * FEATURE_SIZE) * 2)))];
		};
	};
};
extern "C" int32_t* _kernel_0_wrapper(float *buf_weights_gpu1, float *buf_weights_T_gpu1)
{
	{
		dim3 blocks((63 + 1), (15 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_0<<<blocks, threads>>>(buf_weights_gpu1, buf_weights_T_gpu1);
	};
	return 0;
};
static __global__ void _kernel_1(float *buf_h_gpu1)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);
	for (int32_t c9 = 0; (c9 < NUM_LAYERS); (c9 += 1))
	{
		buf_h_gpu1[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (0 * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + ((c9 + 1) * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))] = 0;
	};
};
extern "C" int32_t* _kernel_1_wrapper(float *buf_h_gpu1)
{
	{
		dim3 blocks((15 + 1), (1 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_1<<<blocks, threads>>>(buf_h_gpu1);
	};
	return 0;
};
static __global__ void _kernel_2(float *buf_c_gpu1)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);
	for (int32_t c9 = 0; (c9 <= 3); (c9 += 1))
	{
		buf_c_gpu1[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (0 * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + (c9 * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))] = 0;
	};
};
extern "C" int32_t* _kernel_2_wrapper(float *buf_c_gpu1)
{
	{
		dim3 blocks((15 + 1), (1 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_2<<<blocks, threads>>>(buf_c_gpu1);
	};
	return 0;
};
static __global__ void _kernel_3(float *buf_h_gpu1, float *buf_x_gpu1)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);
	for (int32_t c9 = 0; (c9 < SEQ_LENGTH/2); (c9 += 1))
	{
		for (int32_t c11 = 0; (c11 <= 1); (c11 += 1))
		{
			buf_h_gpu1[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + ((((2 * c9) + c11) + 1) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + (0 * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))] = buf_x_gpu1[(((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (((2 * c9) + c11) * ((1 * FEATURE_SIZE) * BATCH_SIZE)))];
		};
	};
};
extern "C" int32_t* _kernel_3_wrapper(float *buf_h_gpu1, float *buf_x_gpu1)
{
	{
		dim3 blocks((15 + 1), (1 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_3<<<blocks, threads>>>(buf_h_gpu1, buf_x_gpu1);
	};
	return 0;
};
static __global__ void _kernel_4(float *buf_weights_gpu2, float *buf_weights_T_gpu2)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);
	for (int32_t c9 = 0; (c9 < NUM_LAYERS); (c9 += 1))
	{
		for (int32_t c11 = 0; (c11 <= 1); (c11 += 1))
		{
			buf_weights_T_gpu2[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * 4 * FEATURE_SIZE))) + (c11 * ((1 * 4 * FEATURE_SIZE) * FEATURE_SIZE))) + (c9 * (((1 * 4 * FEATURE_SIZE) * FEATURE_SIZE) * 2)))] = buf_weights_gpu2[((((0 + (((32 * __by__) + __ty__) * 1)) + (((32 * __bx__) + __tx__) * (1 * FEATURE_SIZE))) + (c11 * ((1 * FEATURE_SIZE) * 4 * FEATURE_SIZE))) + (c9 * (((1 * FEATURE_SIZE) * 4 * FEATURE_SIZE) * 2)))];
		};
	};
};
extern "C" int32_t* _kernel_4_wrapper(float *buf_weights_gpu2, float *buf_weights_T_gpu2)
{
	{
		dim3 blocks((63 + 1), (15 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_4<<<blocks, threads>>>(buf_weights_gpu2, buf_weights_T_gpu2);
	};
	return 0;
};
static __global__ void _kernel_5(float *buf_weights2_T_gpu2, float *buf_weights2_gpu2)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);
	for (int32_t c9 = 0; (c9 < NUM_LAYERS); (c9 += 1))
	{
		for (int32_t c11 = 0; (c11 <= 1); (c11 += 1))
		{
			buf_weights2_T_gpu2[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * 4 * FEATURE_SIZE))) + (c11 * ((1 * 4 * FEATURE_SIZE) * FEATURE_SIZE))) + (c9 * (((1 * 4 * FEATURE_SIZE) * FEATURE_SIZE) * 2)))] = buf_weights2_gpu2[((((0 + (((32 * __by__) + __ty__) * 1)) + (((32 * __bx__) + __tx__) * (1 * FEATURE_SIZE))) + (c11 * ((1 * FEATURE_SIZE) * 4 * FEATURE_SIZE))) + (c9 * (((1 * FEATURE_SIZE) * 4 * FEATURE_SIZE) * 2)))];
		};
	};
};
extern "C" int32_t* _kernel_5_wrapper(float *buf_weights2_T_gpu2, float *buf_weights2_gpu2)
{
	{
		dim3 blocks((63 + 1), (15 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_5<<<blocks, threads>>>(buf_weights2_T_gpu2, buf_weights2_gpu2);
	};
	return 0;
};
static __global__ void _kernel_6(float *buf_h_gpu2)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);
	for (int32_t c9 = 0; (c9 < NUM_LAYERS); (c9 += 1))
	{
		buf_h_gpu2[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (0 * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + ((c9 + 1) * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))] = 0;
	};
};
extern "C" int32_t* _kernel_6_wrapper(float *buf_h_gpu2)
{
	{
		dim3 blocks((15 + 1), (1 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_6<<<blocks, threads>>>(buf_h_gpu2);
	};
	return 0;
};
static __global__ void _kernel_7(float *buf_c_gpu2)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);
	for (int32_t c9 = 0; (c9 < NUM_LAYERS); (c9 += 1))
	{
		buf_c_gpu2[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (0 * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + (c9 * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))] = 0;
	};
};
extern "C" int32_t* _kernel_7_wrapper(float *buf_c_gpu2)
{
	{
		dim3 blocks((15 + 1), (1 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_7<<<blocks, threads>>>(buf_c_gpu2);
	};
	return 0;
};
static __global__ void _kernel_8(float *buf_h_gpu2)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);
	for (int32_t c9 = 0; (c9 < NUM_LAYERS); (c9 += 1))
	{
		buf_h_gpu2[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (0 * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + ((c9 + 1) * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))] = 0;
	};
};
extern "C" int32_t* _kernel_8_wrapper(float *buf_h_gpu2)
{
	{
		dim3 blocks((15 + 1), (1 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_8<<<blocks, threads>>>(buf_h_gpu2);
	};
	return 0;
};
static __global__ void _kernel_9(float *buf_c_gpu2)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);
	for (int32_t c9 = 0; (c9 < NUM_LAYERS); (c9 += 1))
	{
		buf_c_gpu2[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (0 * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + (c9 * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))] = 0;
	};
};
extern "C" int32_t* _kernel_9_wrapper(float *buf_c_gpu2)
{
	{
		dim3 blocks((15 + 1), (1 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_9<<<blocks, threads>>>(buf_c_gpu2);
	};
	return 0;
};
static __global__ void _kernel_10(float *buf_weights2_gpu3, float *buf_weights2_T_gpu3)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);
	for (int32_t c9 = 0; (c9 < NUM_LAYERS); (c9 += 1))
	{
		for (int32_t c11 = 0; (c11 <= 1); (c11 += 1))
		{
			buf_weights2_T_gpu3[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * 4 * FEATURE_SIZE))) + (c11 * ((1 * 4 * FEATURE_SIZE) * FEATURE_SIZE))) + (c9 * (((1 * 4 * FEATURE_SIZE) * FEATURE_SIZE) * 2)))] = buf_weights2_gpu3[((((0 + (((32 * __by__) + __ty__) * 1)) + (((32 * __bx__) + __tx__) * (1 * FEATURE_SIZE))) + (c11 * ((1 * FEATURE_SIZE) * 4 * FEATURE_SIZE))) + (c9 * (((1 * FEATURE_SIZE) * 4 * FEATURE_SIZE) * 2)))];
		};
	};
};
extern "C" int32_t* _kernel_10_wrapper(float *buf_weights2_gpu3, float *buf_weights2_T_gpu3)
{
	{
		dim3 blocks((63 + 1), (15 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_10<<<blocks, threads>>>(buf_weights2_gpu3, buf_weights2_T_gpu3);
	};
	return 0;
};
static __global__ void _kernel_11(float *buf_h2_gpu3)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);
	for (int32_t c9 = 0; (c9 < NUM_LAYERS); (c9 += 1))
	{
		buf_h2_gpu3[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (0 * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + ((c9 + 1) * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH/2 + 1))))] = 0;
	};
};
extern "C" int32_t* _kernel_11_wrapper(float *buf_h2_gpu3)
{
	{
		dim3 blocks((15 + 1), (1 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_11<<<blocks, threads>>>(buf_h2_gpu3);
	};
	return 0;
};
static __global__ void _kernel_12(float *buf_c2_gpu3)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);
	for (int32_t c9 = 0; (c9 < NUM_LAYERS); (c9 += 1))
	{
		buf_c2_gpu3[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (0 * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + (c9 * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH/2 + 1))))] = 0;
	};
};
extern "C" int32_t* _kernel_12_wrapper(float *buf_c2_gpu3)
{
	{
		dim3 blocks((15 + 1), (1 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_12<<<blocks, threads>>>(buf_c2_gpu3);
	};
	return 0;
};
static __global__ void _kernel_13(int32_t c1, int32_t c3, int32_t c5, int32_t c7, float *buf_biases_gpu1, float *buf_c_gpu1, float *buf_h_gpu1, float *buf_tmp_gpu1)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);

		buf_tmp_gpu1[(((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] = (1 / (1 + exp(-(buf_tmp_gpu1[(((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] + buf_biases_gpu1[((0 + (((32 * __bx__) + __tx__) * 1)) + (c3 * (1 * (4 * FEATURE_SIZE))))]))));

		buf_tmp_gpu1[(((0 + ((((32 * __bx__) + __tx__) + FEATURE_SIZE) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] = (1 / (1 + exp(-(buf_tmp_gpu1[(((0 + ((((32 * __bx__) + __tx__) + FEATURE_SIZE) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] + buf_biases_gpu1[((0 + ((((32 * __bx__) + __tx__) + FEATURE_SIZE) * 1)) + (c3 * (1 * (4 * FEATURE_SIZE))))]))));

		buf_tmp_gpu1[(((0 + ((((32 * __bx__) + __tx__) + (2 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] = tanh((buf_tmp_gpu1[(((0 + ((((32 * __bx__) + __tx__) + (2 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] + buf_biases_gpu1[((0 + ((((32 * __bx__) + __tx__) + (2 * FEATURE_SIZE)) * 1)) + (c3 * (1 * (4 * FEATURE_SIZE))))]));

		buf_tmp_gpu1[(((0 + ((((32 * __bx__) + __tx__) + (3 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] = (1 / (1 + exp(-(buf_tmp_gpu1[(((0 + ((((32 * __bx__) + __tx__) + (3 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] + buf_biases_gpu1[((0 + ((((32 * __bx__) + __tx__) + (3 * FEATURE_SIZE)) * 1)) + (c3 * (1 * (4 * FEATURE_SIZE))))]))));

		buf_c_gpu1[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) + 1) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + (c3 * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))] = ((buf_tmp_gpu1[(((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] * buf_tmp_gpu1[(((0 + ((((32 * __bx__) + __tx__) + (2 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))]) + (buf_tmp_gpu1[(((0 + ((((32 * __bx__) + __tx__) + FEATURE_SIZE) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] * buf_c_gpu1[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + (c3 * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))]));

		buf_h_gpu1[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) + 1) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + ((c3 + 1) * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))] = (tanh(buf_c_gpu1[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) + 1) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + (c3 * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))]) * buf_tmp_gpu1[(((0 + ((((32 * __bx__) + __tx__) + (3 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))]);

};
extern "C" int32_t* _kernel_13_wrapper(int32_t c1, int32_t c3, int32_t c5, int32_t c7, float *buf_biases_gpu1, float *buf_c_gpu1, float *buf_h_gpu1, float *buf_tmp_gpu1)
{
	{
		dim3 blocks((15 + 1), (1 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_13<<<blocks, threads>>>(c1, c3, c5, c7, buf_biases_gpu1, buf_c_gpu1, buf_h_gpu1, buf_tmp_gpu1);
	};
	return 0;
};
static __global__ void _kernel_14(int32_t c1, int32_t c3, int32_t c5, int32_t c7, float *buf_biases_gpu2, float *buf_c_gpu2, float *buf_h_gpu2, float *buf_tmp_gpu2)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);

		buf_tmp_gpu2[(((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] = (1 / (1 + exp(-(buf_tmp_gpu2[(((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] + buf_biases_gpu2[((0 + (((32 * __bx__) + __tx__) * 1)) + (c3 * (1 * (4 * FEATURE_SIZE))))]))));

		buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + FEATURE_SIZE) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] = (1 / (1 + exp(-(buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + FEATURE_SIZE) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] + buf_biases_gpu2[((0 + ((((32 * __bx__) + __tx__) + FEATURE_SIZE) * 1)) + (c3 * (1 * (4 * FEATURE_SIZE))))]))));

		buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + (2 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] = tanh((buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + (2 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] + buf_biases_gpu2[((0 + ((((32 * __bx__) + __tx__) + (2 * FEATURE_SIZE)) * 1)) + (c3 * (1 * (4 * FEATURE_SIZE))))]));

		buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + (3 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] = (1 / (1 + exp(-(buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + (3 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] + buf_biases_gpu2[((0 + ((((32 * __bx__) + __tx__) + (3 * FEATURE_SIZE)) * 1)) + (c3 * (1 * (4 * FEATURE_SIZE))))]))));

		buf_c_gpu2[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) + 1) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + (c3 * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))] = ((buf_tmp_gpu2[(((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] * buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + (2 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))]) + (buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + FEATURE_SIZE) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] * buf_c_gpu2[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + (c3 * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))]));

		buf_h_gpu2[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) + 1) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + ((c3 + 1) * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))] = (tanh(buf_c_gpu2[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) + 1) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + (c3 * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))]) * buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + (3 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + c5) + (2 * c7)) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))]);

};
extern "C" int32_t* _kernel_14_wrapper(int32_t c1, int32_t c3, int32_t c5, int32_t c7, float *buf_biases_gpu2, float *buf_c_gpu2, float *buf_h_gpu2, float *buf_tmp_gpu2)
{
	{
		dim3 blocks((15 + 1), (1 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_14<<<blocks, threads>>>(c1, c3, c5, c7, buf_biases_gpu2, buf_c_gpu2, buf_h_gpu2, buf_tmp_gpu2);
	};
	return 0;
};
static __global__ void _kernel_15(int32_t c1, int32_t c3, float *buf_h_gpu2)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);
	for (int32_t c13 = 0; (c13 <= 9); (c13 += 1))
	{
			buf_h_gpu2[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c13) + 1) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + (0 * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))] = max(buf_h_gpu2[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + (2 * c13)) + 1) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + (4 * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))], buf_h_gpu2[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + ((((((2 * GEMM_BATCH) * c1) - ((2 * GEMM_BATCH) * c3)) + (2 * c13)) + 2) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + (4 * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))]);
	};
};
extern "C" int32_t* _kernel_15_wrapper(int32_t c1, int32_t c3, float *buf_h_gpu2)
{
	{
		dim3 blocks((15 + 1), (1 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_15<<<blocks, threads>>>(c1, c3, buf_h_gpu2);
	};
	return 0;
};
static __global__ void _kernel_16(int32_t c1, int32_t c3, int32_t c5, float *buf_biases2_gpu2, float *buf_c_gpu2, float *buf_h_gpu2, float *buf_tmp_gpu2)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);

	buf_tmp_gpu2[(((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] = (1 / (1 + exp(-(buf_tmp_gpu2[(((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] + buf_biases2_gpu2[((0 + (((32 * __bx__) + __tx__) * 1)) + (c3 * (1 * (4 * FEATURE_SIZE))))]))));

	buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + FEATURE_SIZE) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] = (1 / (1 + exp(-(buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + FEATURE_SIZE) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] + buf_biases2_gpu2[((0 + ((((32 * __bx__) + __tx__) + FEATURE_SIZE) * 1)) + (c3 * (1 * (4 * FEATURE_SIZE))))]))));

	buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + (2 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] = tanh((buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + (2 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] + buf_biases2_gpu2[((0 + ((((32 * __bx__) + __tx__) + (2 * FEATURE_SIZE)) * 1)) + (c3 * (1 * (4 * FEATURE_SIZE))))]));

	buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + (3 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] = (1 / (1 + exp(-(buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + (3 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] + buf_biases2_gpu2[((0 + ((((32 * __bx__) + __tx__) + (3 * FEATURE_SIZE)) * 1)) + (c3 * (1 * (4 * FEATURE_SIZE))))]))));

	buf_c_gpu2[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) + 1) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + ((c3 - 5) * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))] = ((buf_tmp_gpu2[(((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] * buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + (2 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))]) + (buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + FEATURE_SIZE) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] * buf_c_gpu2[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + ((c3 - 5) * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))]));

	buf_h_gpu2[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) + 1) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + ((c3 - 4) * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))] = (tanh(buf_c_gpu2[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) + 1) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + ((c3 - 5) * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH + 1))))]) * buf_tmp_gpu2[(((0 + ((((32 * __bx__) + __tx__) + (3 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))]);
};
extern "C" int32_t* _kernel_16_wrapper(int32_t c1, int32_t c3, int32_t c5, float *buf_biases2_gpu2, float *buf_c_gpu2, float *buf_h_gpu2, float *buf_tmp_gpu2)
{
	{
		dim3 blocks((15 + 1), (1 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_16<<<blocks, threads>>>(c1, c3, c5, buf_biases2_gpu2, buf_c_gpu2, buf_h_gpu2, buf_tmp_gpu2);
	};
	return 0;
};
static __global__ void _kernel_17(int32_t c1, int32_t c3, int32_t c5, float *buf_biases2_gpu3, float *buf_c2_gpu3, float *buf_h2_gpu3, float *buf_tmp2_gpu3)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);

	buf_tmp2_gpu3[(((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] = (1 / (1 + exp(-(buf_tmp2_gpu3[(((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] + buf_biases2_gpu3[((0 + (((32 * __bx__) + __tx__) * 1)) + (c3 * (1 * (4 * FEATURE_SIZE))))]))));

	buf_tmp2_gpu3[(((0 + ((((32 * __bx__) + __tx__) + FEATURE_SIZE) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] = (1 / (1 + exp(-(buf_tmp2_gpu3[(((0 + ((((32 * __bx__) + __tx__) + FEATURE_SIZE) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] + buf_biases2_gpu3[((0 + ((((32 * __bx__) + __tx__) + FEATURE_SIZE) * 1)) + (c3 * (1 * (4 * FEATURE_SIZE))))]))));

	buf_tmp2_gpu3[(((0 + ((((32 * __bx__) + __tx__) + (2 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] = tanh((buf_tmp2_gpu3[(((0 + ((((32 * __bx__) + __tx__) + (2 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] + buf_biases2_gpu3[((0 + ((((32 * __bx__) + __tx__) + (2 * FEATURE_SIZE)) * 1)) + (c3 * (1 * (4 * FEATURE_SIZE))))]));

	buf_tmp2_gpu3[(((0 + ((((32 * __bx__) + __tx__) + (3 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] = (1 / (1 + exp(-(buf_tmp2_gpu3[(((0 + ((((32 * __bx__) + __tx__) + (3 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] + buf_biases2_gpu3[((0 + ((((32 * __bx__) + __tx__) + (3 * FEATURE_SIZE)) * 1)) + (c3 * (1 * (4 * FEATURE_SIZE))))]))));

	buf_c2_gpu3[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) + 1) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + ((c3 - 5) * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH/2 + 1))))] = ((buf_tmp2_gpu3[(((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] * buf_tmp2_gpu3[(((0 + ((((32 * __bx__) + __tx__) + (2 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))]) + (buf_tmp2_gpu3[(((0 + ((((32 * __bx__) + __tx__) + FEATURE_SIZE) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))] * buf_c2_gpu3[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + ((c3 - 5) * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH/2 + 1))))]));

	buf_h2_gpu3[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) + 1) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + ((c3 - 4) * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH/2 + 1))))] = (tanh(buf_c2_gpu3[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + (((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) + 1) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + ((c3 - 5) * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH/2 + 1))))]) * buf_tmp2_gpu3[(((0 + ((((32 * __bx__) + __tx__) + (3 * FEATURE_SIZE)) * 1)) + (((32 * __by__) + __ty__) * (1 * (4 * FEATURE_SIZE)))) + ((((GEMM_BATCH * c1) - (GEMM_BATCH * c3)) + c5) * ((1 * (4 * FEATURE_SIZE)) * BATCH_SIZE)))]);

};
extern "C" int32_t* _kernel_17_wrapper(int32_t c1, int32_t c3, int32_t c5, float *buf_biases2_gpu3, float *buf_c2_gpu3, float *buf_h2_gpu3, float *buf_tmp2_gpu3)
{
	{
		dim3 blocks((15 + 1), (1 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_17<<<blocks, threads>>>(c1, c3, c5, buf_biases2_gpu3, buf_c2_gpu3, buf_h2_gpu3, buf_tmp2_gpu3);
	};
	return 0;
};
static __global__ void _kernel_18(float *_C452_b69, float *buf_h2_gpu3)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __by__ = (blockIdx.y + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	const int32_t __ty__ = (threadIdx.y + 0);
	for (int32_t c9 = 0; (c9 < SEQ_LENGTH/2); (c9 += 1))
	{
		_C452_b69[(((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * (FEATURE_SIZE - 0)))) + (c9 * ((1 * (FEATURE_SIZE - 0)) * (BATCH_SIZE - 0))))] = buf_h2_gpu3[((((0 + (((32 * __bx__) + __tx__) * 1)) + (((32 * __by__) + __ty__) * (1 * FEATURE_SIZE))) + ((c9 + 1) * ((1 * FEATURE_SIZE) * BATCH_SIZE))) + (-1 * (((1 * FEATURE_SIZE) * BATCH_SIZE) * (SEQ_LENGTH/2 + 1))))];
	};
};
extern "C" int32_t* _kernel_18_wrapper(float *_C452_b69, float *buf_h2_gpu3)
{
	{
		dim3 blocks((15 + 1), (1 + 1), 1);
		dim3 threads((31 + 1), (31 + 1), 1);
		_kernel_18<<<blocks, threads>>>(_C452_b69, buf_h2_gpu3);
	};
	return 0;
}
