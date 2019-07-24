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
//#include "struct_params.h"
//#include "stream_demo.h"

using namespace std;

void train()
{
	// ------------- train ---------------------------
	Para* para;
	cudaMallocManaged((void**)&para, sizeof(Para));

	// ========= read data from matlab-file ==============
	IOMatlab::read("listStructTrain",
		para); // num of scenarios
	int size = para->total_size[0];
	int numSces = para->num_sces[0]; 
	cout << "total size: " << size << ", numS: " << numSces << endl;
	cout << " ========== read data into 'para' over =======\n" << endl;
	printLast(para->sces_data, size, 10, "last 10 elems of data");

	// print first sce: sce0
	float* id_score = para->sces_id_score;
	float* mn = para->sces_data_mn;
	float* idx_begin = para->sces_data_idx_begin;
	printToHost(id_score, 2, numSces, "id_score");
	printToHost(mn, 2, numSces, "mn");
	printToHost(idx_begin, 1, numSces+1, "idx begin");

	float* sces_data = para->sces_data;
	float* sce0_data; // data of first sce
	cudaMallocManaged((void**)&sce0_data, mn[0] * mn[1] * sizeof(float));
	for (int i = 0; i < mn[0]*mn[1]; i++)
	{
		sce0_data[i] = sces_data[i];
	}
	printToHost(sce0_data, mn[0], 10, "sce0 data: first 10 cols");
	cout << "print sce0 over \n";

	// ===================== train ============================

	// decalre
	float* lossAllVec;
	// malloc loss and struct
	cudaMallocManaged((void**)&lossAllVec, (int)total_epoches*sizeof(float));
	
	const int M = para->sces_data_mn[0];
	float* cache;
	cudaMallocManaged((void**)&cache, para->num_sces[0] * 2 * sizeof(float));
	const int Nmax =
		gpu_max_value(para->sces_data_mn, para->num_sces[0] * 2, cache);
	// malloc members of struct para
	initPara(para, Nmax);
	cudaDeviceSynchronize();
	cout << "initPara over" << endl;
	// ---------- assign values ------------
	
	para->total_epoches[0]		= total_epoches;
	cout << para->total_epoches[0] << endl; // 51
	para->n_features[0]			= n_features;
	cout << para->n_features[0] << endl; // 17
	para->n_hidden[0]			= n_hidden;
	cout << para->n_hidden[0] << endl; // 50
	para->n_output_classes[0]	= n_output_classes;
	cout << para->n_output_classes[0] << endl;
	para->alpha[0]				= alpha;
	para->score_min[0]			= score_min;
	para->score_max[0]			= score_max;

	cout << "para: "
		<< para->total_epoches[0] << "\n"
		<< para->n_features[0] << "\n"
		<< para->n_hidden[0] << "\n"
		<< para->n_output_classes[0] << "\n"
		<< para->alpha[0] << "\n"
		<< para->score_min[0] << "\n"
		<< para->score_max[0] << endl;
	cout << "assign over\n";

	// para, init rand val of W b
	gpu_fill_rand(para->Wxh, para->n_hidden[0], para->n_features[0], -0.1f, 0.1f, 1);
	gpu_fill_rand(para->Whh, para->n_hidden[0], para->n_hidden[0], -0.1f, 0.1f, 11);
	gpu_fill_rand(para->Why, para->n_output_classes[0], para->n_hidden[0], -0.1f,0.1f,111);
	gpu_fill_rand(para->bh, para->n_hidden[0],1, -0.1f, 0.1f, 22);
	gpu_fill_rand(para->by, para->n_output_classes[0], 1, -0.1f, 0.1f, 222);



	cout << "now train..." << endl;
	trainMultiThread(lossAllVec, 
		para);

	cout << " ========== train over ============== \n";

	/**
		epoches 101 : 93s

	*/

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
	
	train();
	
	//test_gpu_fns();
	

	// ============================================
	mclock.stopAndShow();
	std::system("pause");
	return 0;
}
