#include "cudaUnifiedMem.h"

__global__ void addKernel_unimem(ProcPara_unimem * para)
{
	int i = threadIdx.x;

	para->c_data[i] = para->a_data[i] + para->b_data[i];
}

void initProcPara_unimem(ProcPara_unimem * para, int arraySize)
{
	//在main中已经 分配了 struct的统一内存，此处分配struct中参数的内存 

	// allocate unified mem
	cudaMallocManaged((void**)&para->a_data, arraySize * sizeof(int));
	cudaMallocManaged((void**)&para->b_data, arraySize * sizeof(int));
	cudaMallocManaged((void**)&para->c_data, arraySize * sizeof(int));

	
}

void deInitProcPara_unimem(ProcPara_unimem * para)
{
	// free memory
	cudaFreeHost(para->a_data);
	cudaFreeHost(para->b_data);
	cudaFreeHost(para->c_data);
}

void addWithCuda_unimem(ProcPara_unimem * para, unsigned int arraySize)
{
	cudaSetDevice(0);

	// Launch a kernel on the GPU with one thread for each element.
	addKernel_unimem << <1, arraySize >> > (para);

	cudaDeviceSynchronize();

}

void uniMemMain()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// 注意：使用结构体存储若干参数时，此结构体需要先 alloc mem
	ProcPara_unimem* para;
	cudaMallocManaged((void**)&para, sizeof(ProcPara_unimem)); // 分配结构体大小的 统一内存

	initProcPara_unimem(para, arraySize); // alloc mem

	memcpy((para)->a_data, a, arraySize * sizeof(int));
	memcpy((para)->b_data, b, arraySize * sizeof(int));

	addWithCuda_unimem(para, arraySize);

	memcpy(c, (para)->c_data, arraySize * sizeof(int));

	printf("{ 1, 2, 3, 4, 5 } + { 10, 20, 30, 40, 50 } = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	deInitProcPara_unimem(para);
	cudaFree(para);
}


