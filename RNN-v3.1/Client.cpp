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
#include "gpu_raw_fns.h"

using namespace std;

void train()
{
	// ------------- train ---------------------------
	// read data
	// declare
	sces_struct* sces_s;
	cudaMallocManaged((void**)&sces_s, sizeof(sces_struct));
	// ========= read data from matlab-file ==============
	int numSces = IOMatlab::read("listStructTrain",
		sces_s); // num of scenarios
	int dataSize = sces_s->sces_data_idx_begin[numSces]; // total size of data
	cout << "total size: " << dataSize << endl;
	cout << " ========== read data into 'sces_s' over =======\n" << endl;
	printLast(sces_s->sces_data, dataSize, 10, "last 10 elems of data");

	// print first sce: sce0
	float* id_score = sces_s->sces_id_score;
	int* mn = sces_s->sces_data_mn;
	int* idx_begin = sces_s->sces_data_idx_begin;
	printToHost(id_score, 2, numSces, "id_score");
	printToHost(mn, 2, numSces, "mn");
	printToHost(idx_begin, 1, numSces+1, "idx begin");

	float* sces_data = sces_s->sces_data;
	float* sce0_data; // data of first sce
	cudaMallocManaged((void**)&sce0_data, mn[0] * mn[1] * sizeof(float));
	for (int i = 0; i < mn[0]*mn[1]; i++)
	{
		sce0_data[i] = sces_data[i];
	}
	printToHost(sce0_data, mn[0], 10, "sce0 data: first 10 cols");
	cout << "print sce0 over \n";

	// ===================== train ============================
	int total_epoches = 501;
	int n_features = 17;
	int n_hidden = 50;
	int n_output_classes = 10;
	float alpha = 0.1f;
	float score_min = 6.0f;
	float score_max = 8.9f;
	// decalre
	float* lossAllVec;
	params_struct* p_s;
	rnn_params_struct* rnn_p_s;
	cache_struct* cache_s;
	// malloc loss and struct
	cudaMallocManaged((void**)&lossAllVec, total_epoches*sizeof(float));
	cudaMallocManaged((void**)&rnn_p_s, sizeof(rnn_params_struct));
	cudaMallocManaged((void**)&p_s, sizeof(params_struct));
	cudaMallocManaged((void**)&cache_s, sizeof(cache_struct));
	// malloc members of struct
	cudaMallocManaged((void**)&p_s->Wxh, n_hidden*n_features * sizeof(float));
	cudaMallocManaged((void**)&p_s->Whh, n_hidden*n_hidden * sizeof(float));
	cudaMallocManaged((void**)&p_s->Why, n_hidden*n_output_classes * sizeof(float));
	cudaMallocManaged((void**)&p_s->bh, n_hidden * sizeof(float));
	cudaMallocManaged((void**)&p_s->by, n_output_classes * sizeof(float));
	cudaMallocManaged((void**)&cache_s->tmp_d_vec, n_features * sizeof(float));
	cudaMallocManaged((void**)&cache_s->W_tmp1, n_hidden*n_features * sizeof(float));
	cudaMallocManaged((void**)&cache_s->W_tmp2, n_hidden*n_features * sizeof(float));
	cudaMallocManaged((void**)&cache_s->W_tmp3, n_hidden*n_features * sizeof(float));
	// ---------- assign values ------------
	// sces_s over
	// rnn_p_s
	rnn_p_s->total_epoches = total_epoches;
	rnn_p_s->n_features = n_features;
	rnn_p_s->n_hidden = n_hidden;
	rnn_p_s->n_output_classes = n_output_classes;
	rnn_p_s->alpha = alpha;
	rnn_p_s->score_min = score_min;
	rnn_p_s->score_max = score_max;
	// p_s
	gpu_fill_rand(p_s->Wxh, rnn_p_s->n_hidden, rnn_p_s->n_features, -0.1f, 0.1f, 1);
	gpu_fill_rand(p_s->Whh, rnn_p_s->n_hidden, rnn_p_s->n_hidden, -0.1f, 0.1f, 11);
	gpu_fill_rand(p_s->Why, rnn_p_s->n_output_classes, rnn_p_s->n_hidden, -0.1f,0.1f,111);
	gpu_fill_rand(p_s->bh, rnn_p_s->n_hidden,1, -0.1f, 0.1f, 22);
	gpu_fill_rand(p_s->by, rnn_p_s->n_output_classes, 1, -0.1f, 0.1f, 222);

	trainMultiThread(lossAllVec, 
		sces_s, 
		p_s, 
		rnn_p_s, 
		cache_s);

	cout << " ========== train over ============== \n";

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
	
	//train();
	
	test_gpu_fns();
	

	// ============================================
	mclock.stopAndShow();
	std::system("pause");
	return 0;
}
