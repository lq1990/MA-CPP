#include "stream_demo.h"

__global__ void MyKernel(float* d_out, float* d_in, int size)
{
	int tid = get_tid();

	if (tid < size)
	{
		d_out[tid] = 2.f * d_in[tid];
	}
}

const int M = 500;
const int N = 500;

void stream01()
{
	cudaStream_t stream[2];
	for (int i = 0; i < 2; i++)
	{
		cudaStreamCreate(&stream[i]);
	}
	float* hostPtr;
	const int size = 10;
	const int bytes = 2*size*sizeof(float);// 2倍大小，给2个stream用
	cudaMallocHost(&hostPtr, bytes); // page-locked mem
	for (int i = 0; i < 2*size; i++)
	{
		hostPtr[i] = (float)i; // init values of host
		// 前一半的数据由stream0做，后一半的数据由stream1做。
	}
	print(hostPtr, 1, 2*size, "init host");

	float* inputDevPtr;
	float* outputDevPtr;
	cudaMalloc((void**)&inputDevPtr, bytes);
	cudaMalloc((void**)&outputDevPtr, bytes);
	for (int i = 0; i < 2; i++)
	{
		// 可对指针进行offset
		cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
			bytes, cudaMemcpyHostToDevice, 
			stream[i]);

		// set stream for kernel
		MyKernel << <1, 512, 0, stream[i] >> > (outputDevPtr + i*size,
			inputDevPtr + i*size, size);

		cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
			bytes, cudaMemcpyDeviceToHost,
			stream[i]);
	}
	//cudaDeviceSynchronize(); // 使用device同步是 heavyweight的，一般不建议使用。可由 stream或event同步代替。
	cudaStreamSynchronize(stream[0]);
	cudaStreamSynchronize(stream[1]);
	print(hostPtr, 1, 2*size, "after computing");


	for (int i = 0; i < 2; i++)
	{
		cudaStreamDestroy(stream[i]);
	}
}



void stream_cublas01()
{
	MyClock mc = MyClock("use stream");

	cublasHandle_t handle; // one handle for one device
	cublasCreate(&handle);

	// declare stream
	const int n_streams = 10;
	cudaStream_t stream[n_streams];

	// create stream
	for (int i = 0; i < n_streams; i++)
	{
		cudaStreamCreate(&stream[i]);
	}

	// var on the host
	float* h_A;
	float* h_x;
	float* h_dest0;
	float* h_dest1;
	float* h_dest2;
	float* h_dest3;
	float* h_dest4;
	cudaMallocHost((void**)&h_A, M*N * sizeof(float)); // use page-locked mem
	cudaMallocHost((void**)&h_x, N * sizeof(float));
	cudaMallocHost((void**)&h_dest0, M * sizeof(float));
	cudaMallocHost((void**)&h_dest1, M * sizeof(float));
	cudaMallocHost((void**)&h_dest2, M * sizeof(float));
	cudaMallocHost((void**)&h_dest3, M * sizeof(float));
	cudaMallocHost((void**)&h_dest4, M * sizeof(float));
	for (int i = 0; i < M*N; i++)
	{
		h_A[i] = float(i);
	}
	for (int i = 0; i < N; i++)
	{
		h_x[i] = 1.f;
	}

	// var on the device
	float* d_A;
	float* d_x;
	float* d_dest0;
	float* d_dest1;
	float* d_dest2;
	float* d_dest3;
	float* d_dest4;
	cudaMalloc((void**)&d_A, M*N * sizeof(float));
	cudaMalloc((void**)&d_x, N * sizeof(float));
	cudaMalloc((void**)&d_dest0, M * sizeof(float));
	cudaMalloc((void**)&d_dest1, M * sizeof(float));
	cudaMalloc((void**)&d_dest2, M * sizeof(float));
	cudaMalloc((void**)&d_dest3, M * sizeof(float));
	cudaMalloc((void**)&d_dest4, M * sizeof(float));

	// memcpy H2D
	cudaMemcpy(d_A, h_A, M*N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

	// cuBLAS and stream
	cublasSetStream(handle, stream[0]);
	gpu_mv(handle, d_A, d_x, M, N, d_dest0, false, false);

	cublasSetStream(handle, stream[1]);
	gpu_mv(handle, d_A, d_x, M, N, d_dest1, false, false);

	cublasSetStream(handle, stream[2]);
	gpu_mv(handle, d_A, d_x, M, N, d_dest2, false, false);

	cublasSetStream(handle, stream[3]);
	gpu_mv(handle, d_A, d_x, M, N, d_dest3, false, false);

	cublasSetStream(handle, stream[4]);
	gpu_mv(handle, d_A, d_x, M, N, d_dest4, false, false);

	// stream sync
	for (int i = 0; i < 5; i++)
	{
		cudaStreamSynchronize(stream[i]);
	}

	// memcpy D2H
	cudaMemcpy(h_dest0, d_dest0, M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dest1, d_dest1, M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dest2, d_dest2, M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dest3, d_dest3, M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dest4, d_dest4, M * sizeof(float), cudaMemcpyDeviceToHost);


	// print res ........


	// destroy stream and free
	for (int i = 0; i < n_streams; i++)
	{
		cuStreamDestroy_v2(stream[i]);
	}
	cudaFree(d_A);
	cudaFree(d_x);
	cudaFree(d_dest0);
	cudaFree(d_dest1);
	cudaFree(d_dest2);
	cudaFree(d_dest3);
	cudaFree(d_dest4);
	cudaFree(h_A);
	cudaFree(h_x);
	cudaFree(h_dest0);
	cudaFree(h_dest1);
	cudaFree(h_dest2);
	cudaFree(h_dest3);
	cudaFree(h_dest4);
	cublasDestroy(handle);

	// ===============================
	mc.stopAndShow();
}

void stream_cublas02()
{
	MyClock mc = MyClock("dont use stream");

	cublasHandle_t handle; // one handle for one device
	cublasCreate(&handle);

	// var on the host
	float* h_A;
	float* h_x;
	float* h_dest0;
	float* h_dest1;
	float* h_dest2;
	float* h_dest3;
	float* h_dest4;
	cudaMallocHost((void**)&h_A, M*N * sizeof(float)); // use page-locked mem
	cudaMallocHost((void**)&h_x, N * sizeof(float));
	cudaMallocHost((void**)&h_dest0, M * sizeof(float));
	cudaMallocHost((void**)&h_dest1, M * sizeof(float));
	cudaMallocHost((void**)&h_dest2, M * sizeof(float));
	cudaMallocHost((void**)&h_dest3, M * sizeof(float));
	cudaMallocHost((void**)&h_dest4, M * sizeof(float));
	for (int i = 0; i < M*N; i++)
	{
		h_A[i] = float(i);
	}
	for (int i = 0; i < N; i++)
	{
		h_x[i] = 1.f;
	}

	// var on the device
	float* d_A;
	float* d_x;
	float* d_dest0;
	float* d_dest1;
	float* d_dest2;
	float* d_dest3;
	float* d_dest4;
	cudaMalloc((void**)&d_A, M*N * sizeof(float));
	cudaMalloc((void**)&d_x, N * sizeof(float));
	cudaMalloc((void**)&d_dest0, M * sizeof(float));
	cudaMalloc((void**)&d_dest1, M * sizeof(float));
	cudaMalloc((void**)&d_dest2, M * sizeof(float));
	cudaMalloc((void**)&d_dest3, M * sizeof(float));
	cudaMalloc((void**)&d_dest4, M * sizeof(float));

	// memcpy H2D
	cudaMemcpy(d_A, h_A, M*N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

	// cuBLAS and stream
	gpu_mv(handle, d_A, d_x, M, N, d_dest0);

	gpu_mv(handle, d_A, d_x, M, N, d_dest1);

	gpu_mv(handle, d_A, d_x, M, N, d_dest2);

	gpu_mv(handle, d_A, d_x, M, N, d_dest3);

	gpu_mv(handle, d_A, d_x, M, N, d_dest4);

	// memcpy D2H
	cudaMemcpy(h_dest0, d_dest0, M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dest1, d_dest1, M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dest2, d_dest2, M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dest3, d_dest3, M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dest4, d_dest4, M * sizeof(float), cudaMemcpyDeviceToHost);


	// print res ........

	// free
	cudaFree(d_A);
	cudaFree(d_x);
	cudaFree(d_dest0);
	cudaFree(d_dest1);
	cudaFree(d_dest2);
	cudaFree(d_dest3);
	cudaFree(d_dest4);
	cudaFree(h_A);
	cudaFree(h_x);
	cudaFree(h_dest0);
	cudaFree(h_dest1);
	cudaFree(h_dest2);
	cudaFree(h_dest3);
	cudaFree(h_dest4);
	cublasDestroy(handle);

	// ===============================
	mc.stopAndShow();
}


/**
	log of result:
						not   /   use 5 streams for mv / M   N limit
	cudaMallocManaged: 1.634		1.251				/ 500 500
	cudaMalloc(Host):	0.207		0.004				/ no limit
*/


void print(float* h, int n_rows, int n_cols, string title)
{
	cout << "---------- " << title << " --------------\n";

	for (int i = 0; i < n_rows; i++)
	{
		for (int j = 0; j < n_cols; j++)
		{
			cout << h[j*n_rows + i] << "  ";
		}
		cout << endl;
	}

}
