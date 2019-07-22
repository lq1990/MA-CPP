/*
	RNN version 3.

	The cores of RNN-v3 are CUDA and LSTM.

	author: LQ
*/

#include <iostream>
#include <Windows.h>

#include "IOMatlab.h"
#include "Cuda4RNN.h"
#include "MyClock.h"

// for Test
#include "gpu_raw_fns.h"

using namespace std;

void test()
{
	// ------------- train ---------------------------
	// read data
	// declare
	sces_struct* sces_s;
	cudaMallocManaged((void**)&sces_s, sizeof(sces_struct));
	// read
	int numSces = IOMatlab::read("listStructTrain",
		sces_s);
	int dataSize = sces_s->sces_data_idx_begin[numSces];

	float* id_score = sces_s->sces_id_score;
	int* mn = sces_s->sces_data_mn;
	int* idx_begin = sces_s->sces_data_idx_begin;
	printToHost(id_score, 2, numSces, "id_score");
	printToHost(mn, 2, numSces, "mn");
	printToHost(idx_begin, 1, numSces+1, "idx begin");

	float* sces_data = sces_s->sces_data;
	float* sce0_data;
	cudaMallocManaged((void**)&sce0_data, mn[0] * mn[1] * sizeof(float));
	for (int i = 0; i < mn[0]*mn[1]; i++)
	{
		sce0_data[i] = sces_data[i];
	}
	printToHost(sce0_data, mn[0], 10, "sce0 data: first 10 cols");

	cout << " ========== read data over =======\n" << endl;

	//// train
	//trainMultiThread(lossAllVec, 
	//	sces_s, 
	//	p_s, 
	//	rnn_p_s, 
	//	cache_s);

}


void showCudaInfo()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	int dev;
	for (dev = 0; dev < deviceCount; dev++)
	{
		int driver_version(0), runtime_version(0);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		if (dev == 0)
			if (deviceProp.minor = 9999 && deviceProp.major == 9999)
				printf("\n");
		printf("\nDevice%d:\"%s\"\n", dev, deviceProp.name);
		cudaDriverGetVersion(&driver_version);
		printf("CUDA驱动版本:                                   %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
		cudaRuntimeGetVersion(&runtime_version);
		printf("CUDA运行时版本:                                 %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
		printf("设备计算能力:                                   %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("Total amount of Global Memory:                  %u bytes\n", deviceProp.totalGlobalMem);
		printf("Number of SMs:                                  %d\n", deviceProp.multiProcessorCount);
		printf("Total amount of Constant Memory:                %u bytes\n", deviceProp.totalConstMem);
		printf("Total amount of Shared Memory per block:        %u bytes\n", deviceProp.sharedMemPerBlock);
		printf("Total number of registers available per block:  %d\n", deviceProp.regsPerBlock);
		printf("Warp size:                                      %d\n", deviceProp.warpSize);
		printf("Maximum number of threads per SM:               %d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);
		printf("Maximum size of each dimension of a block:      %d x %d x %d\n", deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("Maximum size of each dimension of a grid:       %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("Maximum memory pitch:                           %u bytes\n", deviceProp.memPitch);
		printf("Texture alignmemt:                              %u bytes\n", deviceProp.texturePitchAlignment);
		printf("Clock rate:                                     %.2f GHz\n", deviceProp.clockRate * 1e-6f);
		printf("Memory Clock rate:                              %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
		printf("Memory Bus Width:                               %d-bit\n", deviceProp.memoryBusWidth);
	}
}


int main()
{
	gpu_raw_warmup();
	MyClock mclock = MyClock("main");
	// ===============================================
	
	test();
	
	// test_gpu_fns();
	

	// ============================================
	mclock.stopAndShow();
	std::system("pause");
	return 0;
}
