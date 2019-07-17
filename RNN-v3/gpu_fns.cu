#include "gpu_fns.h"


device_vector<float> gpu_mmul(cublasHandle_t handle,
	device_vector<float> d_A, 
	device_vector<float> d_B, 
	int A_M, int A_N, int B_N)
{
	device_vector<float> d_C(A_M * B_N);

	float alpha = 1.0f;
	float beta = 0.0f;
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		A_M, B_N, A_N,
		&alpha,
		raw_pointer_cast(&d_A[0]), A_M,
		raw_pointer_cast(&d_B[0]), A_N,
		&beta,
		raw_pointer_cast(&d_C[0]), A_M);

	cudaDeviceSynchronize(); // wait for gpu to finish
	return d_C;
}

device_vector<float> gpu_mv(cublasHandle_t handle, 
	device_vector<float> d_A,
	device_vector<float> d_x,
	int am, int an, bool isAT)
{
	float alpha = 1.0f;
	float beta = 0.0f;
	if (isAT)
	{
		// y = A' * x
		 device_vector<float> d_y(an * 1);

		cublasSgemv(handle, // y = alpha * A*x + beta*y
			CUBLAS_OP_T,
			am, an,
			&alpha,
			raw_pointer_cast(&d_A[0]), am,
			raw_pointer_cast(&d_x[0]), 1,
			&beta,
			raw_pointer_cast(&d_y[0]), 1);

		cudaDeviceSynchronize(); // wait for gpu to finish
		return d_y;
	}
	else
	{
		// y = A * x
		device_vector<float> d_y(am * 1);
		cublasSgemv(handle, // y = alpha * A*x + beta*y
			CUBLAS_OP_N,
			am, an,
			&alpha,
			raw_pointer_cast(&d_A[0]), am,
			raw_pointer_cast(&d_x[0]), 1,
			&beta,
			raw_pointer_cast(&d_y[0]), 1);

		cudaDeviceSynchronize(); // wait for gpu to finish
		return d_y;
	}
	
}

device_vector<float> gpu_add(cublasHandle_t handle, 
	device_vector<float> d_a,
	device_vector<float> d_b,
	int size)
{
	// c = a + b, use thrust or cuBLAS
	device_vector<float> d_c(size);

	thrust::transform(d_a.begin(), d_a.end(), d_b.begin(),
		d_c.begin(), thrust::plus<float>());

	cudaDeviceSynchronize();
	return d_c;
}

void gpu_mul_elementwise(const float * d_a, const float * d_b, float * d_c, int size)
{
}

device_vector<float> gpu_tanh(device_vector<float> d_x, int size)
{
	// y = tanh(x)
	thrust::device_vector<float> d_y(size);
	thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), thrust::);

}

device_vector<float> gpu_fill_rand(device_vector<float> d_A, int n_rows, int n_cols,
	float low, float high)
{
	thrust::counting_iterator<int> index_seq_begin(0);

	thrust::transform(index_seq_begin,
		index_seq_begin + n_rows * n_cols,
		d_A.begin(),
		prg(low, high));

	cudaDeviceSynchronize();
	return d_A;
}

void printToHost(device_vector<float> d_A, int n_rows, int n_cols, string title)
{
	cout << title << endl;
	
	host_vector<float> h_A = d_A;
	for (int i = 0; i < n_rows; i++)
	{
		for (int j = 0; j < n_cols; j++)
		{
			cout << h_A[j*n_rows + i] << "\t";
		}
		cout << endl;
	}

	cout << endl;
}

void printToHost(float * d, int n_rows, int n_cols, string title)
{
	// 先用 device_vector将 raw_pointer 转换过来
	device_vector<float> d_vec(d, d + n_rows*n_cols); 
	
	printToHost(d_vec, n_rows, n_cols, title);
}
