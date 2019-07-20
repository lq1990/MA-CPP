/*
	RNN version 3.

	The cores of RNN-v3 are CUDA and LSTM.

	author: LQ
*/

#include <iostream>
#include <Windows.h>

#include "IOMatlab.h"
#include "Cuda4RNN.h"
#include "CudaMap.h"
#include "MyClock.h"

// for Test
#include "gpu_fns.h"
#include "CudaUtils.h"

using namespace std;

void test()
{
	// ------------- train ---------------------------
	// read data
	device_vector<float> sces_id_score;
	device_vector<float> sces_data;
	device_vector<int> sces_data_mn;
	device_vector<int> sces_data_idx_begin;
	IOMatlab::read("listStructTrain",
		sces_id_score,
		sces_data,
		sces_data_mn,
		sces_data_idx_begin);

	cout << "data size: " << sces_data.size() << endl;
	cout << " ========== read data over =======\n" << endl;

	// train 当前使用的class。试试不用class，直接用fn。对比下速度
	Cuda4RNN* rnn;
	cudaMalloc((void**)&rnn, sizeof(Cuda4RNN));
	//rnn->alpha = device_vector<float>(1);
	//rnn->total_epoches = device_vector<int>(1);
	//rnn->n_features = device_vector<int>(1);
	//rnn->n_hidden = device_vector<int>(1);
	//rnn->n_output_classes = device_vector<int>(1);
	//rnn->score_min = device_vector<float>(1);
	//rnn->score_max = device_vector<float>(1);

	float alpha = 0.1f;
	int total_epoches = 501;
	int n_features = 17;
	int n_hidden = 50;
	int n_output_classes = 10;
	float score_min = 6.0f;
	float score_max = 8.9f;
	device_vector<float> Wxh = gpu_generate_rand(n_hidden, n_features, -0.01, 0.01f, 1);
	device_vector<float> Whh = gpu_generate_rand(n_hidden, n_hidden, -0.01, 0.01f, 11);
	device_vector<float> Why = gpu_generate_rand(n_output_classes, n_hidden, -0.01, 0.01f, 111);
	device_vector<float> bh = gpu_generate_rand(n_hidden, 1, -0.01, 0.01f, 22);
	device_vector<float> by = gpu_generate_rand(n_output_classes, 1, -0.01, 0.01f, 222);

	device_vector<float> lossAll; // init size=0
	rnn->trainMultiThread(sces_id_score,
		sces_data,
		sces_data_mn, 
		sces_data_idx_begin,
		lossAll,
		Wxh,
		Whh,
		Why,
		bh,
		by,
		alpha,
		total_epoches,
		n_features,
		n_hidden,
		n_output_classes,
		score_min,
		score_max);

	cout << "lossAll size: " << lossAll.size() << endl;
	printToHost(lossAll, lossAll.size(), 1, "lossAll on host");

	cudaFree(rnn);

	// -------------- gpu_fns --------------------------------
	//Cuda4RNN::test_gpu_fns_CudaUtils();


	// ---------------- IOMatlab 读取matlab中转置后的matDataZScore, :) ------------
	/*
	cout << "test IOMatlab \n";
	device_vector<float> sces_id_score;
	device_vector<float> sces_data;
	device_vector<int> sces_data_mn;
	device_vector<int> sces_data_idx_begin;

	IOMatlab::read("listStructTrain",
		sces_id_score,
		sces_data,
		sces_data_mn,
		sces_data_idx_begin);

	cout << "sces_id_score.size(): " << sces_id_score.size() << endl;
	cout << "sces_data.size(): " << sces_data.size() << endl;
	cout << "sces_data_mn.size(): " << sces_data_mn.size() << endl;
	cout << "sces_data_idx_begin.size(): " << sces_data_idx_begin.size() << endl;

	printToHost(sces_id_score, 2, sces_id_score.size() / 2, "sces_id_score");
	printToHost(sces_data_mn, 2, sces_data_mn.size() / 2, "mn");
	printToHost(sces_data_idx_begin, 1, sces_data_idx_begin.size(), "idx_begin");

	cout << "data first 10 \n";
	host_vector<float> h_data = sces_data;
	for (int i = 0; i < 10; i++)
	{
		cout << i << ": " << h_data[i] << endl;
	}
	*/

	// --------- SceStruct of Scenarios from IOMatlab --------------
	/*
	thrust::host_vector<SceStruct> vec =
		IOMatlab::read("listStructTrain"); // read返回device，但被host强转
	cout << "iomatlab size: " << vec.size() << endl;

	// first scenario {id, score, matDataZScore}
	SceStruct first = vec[0];
	cout << "id: " << first.id
		<< ", score: " << first.score
		<< endl;

	MyArray *matData = first.matDataZScore;
	cout << "matData size: " << matData->size << endl;

	MyArray *row0 = matData->getRowToHost(0);

	// row0 of matData
	row0->printToHost(row0->arr, row0->size, "row0", false);

	matData->printMatToHost(matData->arr,
		matData->n_rows_origin,
		matData->size / matData->n_rows_origin,
		"fist matDataZScore");
	*/

	// -------------- test Cuda4RNN --------------------
	/*Cuda4RNN cuRNN = Cuda4RNN();

	MyArray* inputs = MyArray::randn(200, 17);
	float score = 7.0f;
	MyArray* hprev = MyArray::randn(50, 1);
	device_vector<float> true_false;
	device_vector<float> log_target;
	device_vector<float> log_prediction;

	CudaMap map = cuRNN.lossFun(inputs, score, hprev,
		true_false, log_target, log_prediction);
	map.printToHost("result of lossFun");*/

	// ---------- test lossFun --------------
	/*MyArray* inputs = new MyArray();
	float score = 0;
	MyArray* hprev = new MyArray();
	device_vector<float> true_false;
	device_vector<float> log_target;
	device_vector<float> log_prediction;

	CudaMap mp = cuRNN.lossFun(inputs, score, hprev,
		true_false, log_target, log_prediction);
	mp.printToHost("res of lossFun");*/

	// show Wxh
	/*MyArray* Wxh = MyParams::Wxh;
	Wxh->printMatToHost(Wxh->arr,
		Wxh->n_rows_origin,
		Wxh->size / Wxh->n_rows_origin,
		"Wxh");*/

	// test CudaMap
	/*CudaMap map = CudaMap();
	map.put(-1, MyArray::randn(3, 2));
	map.put(2, MyArray::randn(3, 4));

	map.printToHost("test map -1");*/

	//// --------- test score2onehot ----------
	//MyArray* marr = cuRNN.score2onehot(7.0);
	//marr->printToHost(marr->arr, marr->size, "score2onehot(7.0)");

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
	gpu_warmup();
	MyClock mclock = MyClock("main");
	// ===============================================
	
	test();
	

	

	// ============================================
	mclock.stopAndShow();
	std::system("pause");
	return 0;
}
