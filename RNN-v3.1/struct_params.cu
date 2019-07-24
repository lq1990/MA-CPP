#include "struct_params.h"

__global__ void add_kernel(Para* para, int size)
{
	int tid = threadIdx.x;

	if (tid < size)
	{
		para->d_c[tid] = para->d_a[tid] + para->d_b[tid];
	}
}

void struct_para_main()
{
	const int size = 10;
	const int bytes = size * sizeof(float);

	// alloc struct on the host
	Para* h_para; // struct 只需要在host端定义即可
	cudaMallocHost((void**)&h_para, sizeof(Para));

	// alloc params in struct on the host/dev
	cudaMallocHost((void**)&h_para->h_a, bytes);
	cudaMallocHost((void**)&h_para->h_b, bytes);
	cudaMallocHost((void**)&h_para->h_c, bytes);
	cudaMalloc((void**)&h_para->d_a, bytes);
	cudaMalloc((void**)&h_para->d_b, bytes);
	cudaMalloc((void**)&h_para->d_c, bytes);

	// init values of h_a, h_b
	for (int i = 0; i < size; i++)
	{
		h_para->h_a[i] = 1.f;
		h_para->h_b[i] = 2.f;
	}

	// mem cpy host 2 dev
	cudaMemcpy(h_para->d_a, h_para->h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(h_para->d_b, h_para->h_b, bytes, cudaMemcpyHostToDevice);

	add_kernel << <1, size >> > (h_para, size);

	// mem cpy dev 2 host
	cudaMemcpy(h_para->h_c, h_para->d_c, bytes, cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; i++)
	{
		cout << h_para->h_c[i] << "  ";
	}
	cout << endl;




	// free
	cudaFree(h_para->d_a);
	cudaFree(h_para->d_b);
	cudaFree(h_para->d_c);
	cudaFreeHost(h_para->h_a);
	cudaFreeHost(h_para->h_b);
	cudaFreeHost(h_para->h_c);
	cudaFreeHost(h_para);
}
