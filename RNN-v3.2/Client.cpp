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
	cudaMallocHost((void**)&para, sizeof(Para));

	// ========= read data from matlab-file ==============
	IOMatlab::read("listStructTrain",
		para); // num of scenarios
	int size = para->h_total_size[0];
	int numSces = para->h_num_sces[0]; 
	cout << "total size: " << size << ", numS: " << numSces << endl;
	cout << " ========== read data into 'para' over =======\n" << endl;
	printLast(para->h_sces_data, size, 10, "last 10 elems of data");

	// print first sce: sce0
	float* id_score = para->h_sces_id_score;
	float* mn = para->h_sces_data_mn;
	float* idx_begin = para->h_sces_data_idx_begin;
	printToHost(id_score, 2, numSces, "id_score");
	printToHost(mn, 2, numSces, "mn");
	printToHost(idx_begin, 1, numSces+1, "idx begin");

	float* sces_data = para->h_sces_data;
	float* sce0_data;
	cudaMallocHost((void**)&sce0_data, mn[0] * mn[1] * sizeof(float));
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
	cudaMalloc((void**)&lossAllVec, (int)total_epoches*sizeof(float));
	cout << "after malloc lossAllVec\n";

	const int M = para->h_sces_data_mn[0];
	cout << "M: " << M << endl;
	float* d_cache;
	cudaMallocManaged((void**)&d_cache, 22 * 2 * sizeof(float));
	gpu_max_value(para->d_sces_data_mn, 22 * 2, d_cache);
	cout << "after max value\n";
	// get val of cache
	float Nmax = d_cache[0];
	cout << "Nmax: " << Nmax << endl;


	// malloc members of struct para
	cout << "now init Para" << endl;
	initPara(para, Nmax); 
	cudaDeviceSynchronize();
	cout << "initPara over" << endl;
	// ---------- assign values ------------
	
	para->h_total_epoches[0]		= total_epoches;
	para->h_n_features[0]			= n_features;
	para->h_n_hidden[0]			= n_hidden;
	para->h_n_output_classes[0]	= n_output_classes;
	para->h_alpha[0]				= alpha;
	para->h_score_min[0]			= score_min;
	para->h_score_max[0]			= score_max;

	cout << "para: "
		<< para->h_total_epoches[0] << "\n"
		<< para->h_n_features[0] << "\n"
		<< para->h_n_hidden[0] << "\n"
		<< para->h_n_output_classes[0] << "\n"
		<< para->h_alpha[0] << "\n"
		<< para->h_score_min[0] << "\n"
		<< para->h_score_max[0] << endl;
	cout << "assign over\n";

	// para, init rand val of W b
	gpu_fill_rand(para->d_Wxh, para->h_n_hidden[0], para->h_n_features[0], -0.1f, 0.1f, 1);
	gpu_fill_rand(para->d_Whh, para->h_n_hidden[0], para->h_n_hidden[0], -0.1f, 0.1f, 11);
	gpu_fill_rand(para->d_Why, para->h_n_output_classes[0], para->h_n_hidden[0], -0.1f,0.1f,111);
	gpu_fill_rand(para->d_bh,  para->h_n_hidden[0],1, -0.1f, 0.1f, 22);
	gpu_fill_rand(para->d_by,  para->h_n_output_classes[0], 1, -0.1f, 0.1f, 222);

	cout << "fill rand of W b overn\n";
	cout << "now train..." << endl;
	trainMultiThread(lossAllVec, 
		para);

	cout << " ========== train over ============== \n";

	/**
		epoches 101 : 93s

	*/


	cudaFree(d_cache);
	cudaFree(lossAllVec);
	cudaFreeHost(sce0_data);


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

	// 21 epoches, 50 n_hidden => 53.8s 比cpu单线程的73s

	
	//test_gpu_fns();


	// ------------------ score2onehot -----------------------------
	//int idx1_targets;

	/*for (float i = 6.0f; i <= 8.91f; i+=0.1f)
	{
		float* onehot = score2onehot(i, idx1_targets, 10, 6.0, 8.9);
		cout << "i: " << i << ": ";
		printToHost(onehot, 1, 10, "onehot");

	}*/


	// find max value  :)
	/*float* d_in, *cache;
	cudaMallocManaged((void**)&d_in, 10 * sizeof(float));
	cudaMallocManaged((void**)&cache, 10 * sizeof(float));
	for (int i = 0; i < 10; i++)
	{
		float val = rand() % 10;
		d_in[i] = val;
		cout << val << " ";
	}
	cout << endl;


	gpu_max_value(d_in, 10, cache);

	cout << "cache: \n";
	cout << cache[0] << endl;
	cout << endl;*/

	// cudaMallocHost 是否可以 []，可以啊
	/*float* h_in;
	cudaMallocHost((void**)&h_in, 10 * sizeof(float));
	for (int i = 0; i < 10; i++)
	{
		cout << h_in[i] << "  ";
	}
	cout << endl;

	h_in[0] = 10;
	for (int i = 0; i < 10; i++)
	{
		cout << h_in[i] << "  ";
	}
	cout << endl;

	cudaFreeHost(h_in);*/
	
	// gpu_fill  :)
	/*Para* pa;
	cudaMallocHost((void**)&pa, sizeof(Para));
	cudaMalloc((void**)&pa->d_bh, 10 * sizeof(float));
	gpu_fill(pa->d_bh, 10, 2.1f);

	printToHost(pa->d_bh, 1, 10, "d_out");

	cudaFree(pa->d_dbh);
	cudaFreeHost(pa);*/

	// ============================================
	mclock.stopAndShow();
	std::system("pause");
	return 0;
}
