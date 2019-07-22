#include "CudaScript.h"



CudaScript::CudaScript()
{
}


CudaScript::~CudaScript()
{
}

void CudaScript::showCudaInfo()
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

void CudaScript::cublasDemo()
{
	cublasStatus_t stat;
	cublasHandle_t handle; // cuBLAS handle
	cublasCreate(&handle);

	const int M = 3;
	const int N = 2;
	// var on the host
	float *h_in_A, *h_in_x, *h_y;
	h_in_A = (float*)malloc(M*N * sizeof(float));
	h_in_x = (float*)malloc(N * sizeof(float));
	h_y = (float*)malloc(M * sizeof(float));
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

__global__ void addKernel(ProcPara * d_para)
{
	int i = threadIdx.x;
	d_para->d_c[i] = d_para->d_a[i] + d_para->d_b[i];
	
}

void initProcPara(ProcPara** ha_para, ProcPara** da_para, int arraySize)
{
	// alloc struct mem，往host和device上都分配struct mem
	cudaMallocHost((void**)ha_para, sizeof(ProcPara));
	cudaMalloc((void**)da_para, sizeof(ProcPara));

	// temp ptr to ha_para, and malloc 借用临时指针变量，指向了一级指针，实现功能
	ProcPara *h_para = *ha_para;

	cudaMallocHost((void**)&h_para->h_a, arraySize * sizeof(int));
	cudaMallocHost((void**)&h_para->h_b, arraySize * sizeof(int));
	cudaMallocHost((void**)&h_para->h_c, arraySize * sizeof(int));

	cudaMalloc((void**)&h_para->d_a, arraySize * sizeof(int));
	cudaMalloc((void**)&h_para->d_b, arraySize * sizeof(int));
	cudaMalloc((void**)&h_para->d_c, arraySize * sizeof(int));

	// exchange data
	cudaMemcpy(*da_para, *ha_para, sizeof(ProcPara), cudaMemcpyHostToDevice);

}

void deInitProcPara(ProcPara* h_para, ProcPara* d_para)
{
	cudaFreeHost(h_para->h_a);
	cudaFreeHost(h_para->h_b);
	cudaFreeHost(h_para->h_c);

	cudaFree(h_para->d_a);
	cudaFree(h_para->d_b);
	cudaFree(h_para->d_c);

	// release struc mem
	cudaFreeHost(h_para);
	cudaFree(d_para);
	
}

void addWithCuda(ProcPara * h_para, ProcPara* d_para, unsigned int arraySize)
{
	cudaSetDevice(0);

	cudaMemcpy(h_para->d_a, h_para->h_a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(h_para->d_b, h_para->h_b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

	// launch kernel on the GPU
	addKernel << <1, arraySize >> > (d_para);

	cudaMemcpy(h_para->h_c, h_para->d_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
}

void testUseStructToStoreHostDeviveParams()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 11, 22, 33, 44, 55 };
	int c[arraySize] = { 0 };

	//ProcPara* para = new ProcPara; // 结构体指针实体在cpu mem中，计算结果是错误的
	ProcPara *h_para;
	ProcPara *d_para;

	initProcPara(&h_para, &d_para, arraySize);

	memcpy(h_para->h_a, a, arraySize * sizeof(int));
	memcpy(h_para->h_b, b, arraySize * sizeof(int));

	addWithCuda(h_para, d_para, arraySize);

	memcpy(c, h_para->h_c, arraySize * sizeof(int));

	printf("{ 1, 2, 3, 4, 5 } + { 11, 22, 33, 44, 55 } = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// free
	deInitProcPara(h_para, d_para);

}


