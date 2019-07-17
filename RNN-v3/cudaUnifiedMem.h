#pragma once
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <curand.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

/*
	使用 Unified Memory ，内存可同时被 CPU、GPU访问
	简化编程难度。

	使用struct存储所需 params，避免fn中传入参数太多
*/

struct ProcPara_unimem
{
	int *a_data; // data of a
	int *a_dim; // dim(M, N). a_dim[0]=M, a_dim[1]=N

	int *b_data;
	int *b_dim;

	int *c_data;
	int *c_dim;
};

__global__ void addKernel_unimem(ProcPara_unimem* para);

void initProcPara_unimem(ProcPara_unimem* para, int arraySize);

void deInitProcPara_unimem(ProcPara_unimem* para);

void addWithCuda_unimem(ProcPara_unimem* para, unsigned int arraySize);

void uniMemMain();