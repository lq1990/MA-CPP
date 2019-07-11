#include "Cuda4RNN.h"



Cuda4RNN::Cuda4RNN()
{
}


Cuda4RNN::~Cuda4RNN()
{
}

void Cuda4RNN::showCudaInfo()
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

void Cuda4RNN::cublasDemo()
{
	cublasStatus_t stat;
	cublasHandle_t handle;
	cublasCreate(&handle);

	const int M = 3;
	const int N = 2;
	// var on the host
	float *h_in_A, *h_in_x, *h_y;
	h_in_A = (float*)malloc(M*N*sizeof(float));
	h_in_x = (float*)malloc(N*sizeof(float));
	h_y = (float*)malloc(M*sizeof(float));
	for (int i = 0; i < M*N; i++)
	{
		h_in_A[i] = (float)i;
		/*
			0, 3,
			1, 4,
			2, 5
		*/
	}
	for (int i = 0; i < N; i++)
	{
		h_in_x[i] = 1.0f;
	}

	// var on the device
	float *d_in_A, *d_in_x, *d_y;
	cudaMalloc((void**)&d_in_A, M*N * sizeof(float));
	cudaMalloc((void**)&d_in_x, N * sizeof(float));
	cudaMalloc((void**)&d_y, M * sizeof(float));
	cudaMemcpy(d_in_A, h_in_A, M*N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_x, h_in_x, N * sizeof(float), cudaMemcpyHostToDevice);

	// cuBLAS
	float alpha = 1.0f;
	float beta = 0.0f;
	cublasSgemv(handle,
		CUBLAS_OP_N,
		M, N,
		&alpha, 
		d_in_A, M,
		d_in_x, 1,
		&beta,
		d_y, 1); // d_y stores result, y = alpha * A * x + beta * y

	// copy back
	cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

	// show result: y = A*x
	cout << "y = A * x " << endl;
	for (int i = 0; i < M; i++)
	{
		cout << h_y[i] << "\t";
	}
	cout << endl;

	// free
	cudaFree(d_in_A);
	cudaFree(d_in_x);
	cudaFree(d_y);
	free(h_in_A);
	free(h_in_x);
	free(h_y);
	cublasDestroy(handle);
}
