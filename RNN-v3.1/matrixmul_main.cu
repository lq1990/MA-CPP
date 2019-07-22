#include "matrixmul_main.h"


void mmul_main()
{
	// perform matrix mul C = A*B
	// dim(N,N)
	int N = 10;
	int SIZE = N * N;

	// alloc mem on the host
	vector<float> h_A(SIZE);
	vector<float> h_B(SIZE);
	vector<float> h_C(SIZE);

	// init matrix
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			h_A[i*N + j] = sin(i);
			h_B[i*N + j] = cos(j);
		}
	}

	// alloc mem on the device
	// 自定义的 dev_array 功能和 thrust::device_vector<> 类似
	cuda_class_dev_array<float> d_A(SIZE);
	cuda_class_dev_array<float> d_B(SIZE);
	cuda_class_dev_array<float> d_C(SIZE);

	d_A.set(&h_A[0], SIZE); // cudaMemcpyHostToDevice
	d_B.set(&h_B[0], SIZE);

	// 从kernel.h中拿到fn
	matrixMul(d_A.getData(), d_B.getData(), d_C.getData(), N);
	cudaDeviceSynchronize(); // wait for device to finish

	d_C.get(&h_C[0], SIZE);
	cudaDeviceSynchronize();

	float *cpu_C;
	cpu_C = new float[SIZE];

	// now do the matrix mul on the GPU
	float sum;
	for (int row = 0; row < N; row++)
	{
		for (int col = 0; col < N; col++)
		{
			sum = 0.f;
			for (int n = 0; n < N; n++)
			{
				sum += h_A[row*N + n] * h_B[n*N + col];
			}
			cpu_C[row*N + col] = sum;
		}
	}

	float err = 0;
	// check the result and make sure it is correct
	for (int ROW = 0; ROW < N; ROW++)
	{
		for (int COL = 0; COL < N; COL++)
		{
			err += cpu_C[ROW*N + COL] - h_C[ROW*N + COL];
		}
	}
	cout << "Error: " << err << endl;

}

