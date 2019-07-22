#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "kernel.h"
#include "device_launch_parameters.h"
#include <stdlib.h>

using namespace std;

__global__ void matrixMul_kernel(float * A, float * B, float * C, int N)
{
	int ROW = blockIdx.y * blockDim.y + threadIdx.y;
	int COL = blockIdx.x * blockDim.x + threadIdx.x;

	float tmpSum = 0;

	if (ROW < N && COL < N)
	{
		// each thread computes one elem of the block sub-matrix
		for (int i = 0; i < N; i++)
		{
			tmpSum += A[ROW * N + i] * B[i * N + COL];
		}
	}
	C[ROW * N + COL] = tmpSum;
}

void matrixMul(float * A, float * B, float * C, int N)
{
	// declare the number of blocks per grid and block size
	dim3 bs(N, N); // block size
	dim3 gs(1, 1); // grid size
	if (N*N > 512)
	{
		bs.x = 512;
		bs.y = 512;
		gs.x = ceil(double(N) / double(bs.x));
		gs.y = ceil(double(N) / double(bs.y));
	}

	matrixMul_kernel <<<gs,bs>>> (A, B, C, N); // kernel之间会 隐式wait
}
