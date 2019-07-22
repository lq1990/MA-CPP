#pragma once
#include <iostream>
#include <stdio.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

struct ProcPara
{
	int *h_a;
	int *h_b;
	int *h_c;

	int *d_a;
	int *d_b;
	int *d_c;
};

__global__ void addKernel(ProcPara* d_para);

void initProcPara(ProcPara** ha_para, ProcPara** da_para, int arraySize);

void deInitProcPara(ProcPara* h_para, ProcPara* d_para);

void addWithCuda(ProcPara *para, unsigned int arraySize);

void testUseStructToStoreHostDeviveParams();

class CudaScript
{
public:
	CudaScript();
	~CudaScript();

	static void showCudaInfo();

	static void cublasDemo();

	
};

