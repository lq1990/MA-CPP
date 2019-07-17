#include "cublas_thrust.h"







void gpu_fill_rand0(float * A, int n_rows, int n_cols)
{
	// create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

	// fill
	curandGenerateUniform(prng, A, n_rows*n_cols); // 0~1

}

void gpu_blas_mmul(cublasHandle_t handle, const float * A, const float * B, float * C, const int m, const int k, const int n)
{
	int lda = m, ldb = k, ldc = m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		m, n, k,
		alpha,
		A, lda,
		B, ldb,
		beta,
		C, ldc);
}

void print_matrix(const float * A, int n_rows, int n_cols)
{
	for (int i = 0; i < n_rows; i++)
	{
		for (int j = 0; j < n_cols; j++)
		{
			cout << A[j*n_rows + i] << " ";
		}
		cout << endl;
	}
}



void cublas_thrust_main1()
{
	cout << "cublas_thrust main: " << endl;
	cublasHandle_t handle;
	cublasCreate(&handle);
	// ================ 普通版，不用thrust ========================
	// alloc 3 arrays on CPU
	int n_rows_A, n_cols_A,
		n_rows_B, n_cols_B,
		n_rows_C, n_cols_C;

	n_rows_A = n_cols_A = n_rows_B = n_cols_B = n_rows_C = n_cols_C = 3000;

	float *h_A = (float*)malloc(n_rows_A * n_cols_A * sizeof(float));
	float *h_B = (float*)malloc(n_rows_B * n_cols_B * sizeof(float));
	float *h_C = (float*)malloc(n_rows_C * n_cols_C * sizeof(float));

	// alloc 3 arrays on GPU
	float *d_A, *d_B, *d_C;
	cudaMalloc((void**)&d_A, n_rows_A*n_cols_A * sizeof(float));
	cudaMalloc((void**)&d_B, n_rows_B*n_cols_B * sizeof(float));
	cudaMalloc((void**)&d_C, n_rows_C*n_cols_C * sizeof(float));

	// fill
	gpu_fill_rand0(d_A, n_rows_A, n_cols_A);
	gpu_fill_rand0(d_B, n_rows_B, n_cols_B);

	// mul
	gpu_blas_mmul(handle, d_A, d_B, d_C, n_rows_A, n_cols_A, n_cols_B);

	// copy back to host
	cudaMemcpy(h_C, d_C, n_rows_C*n_cols_C * sizeof(float), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < 10; i++)
	{
		cout << h_C[i] << endl;
	}

	/*print_matrix(h_C, n_rows_C, n_cols_C);*/
	


	// free
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);




	// ====================== 使用thrust ===========================







	cublasDestroy(handle);
}

void cublas_thrust_main2()
{
	cout << "cublas_thrust main: " << endl;
	cublasHandle_t handle;
	cublasCreate(&handle);
	// =============== 用thrust ========================
	// alloc 3 arrays on CPU
	int n_rows_A, n_cols_A,
		n_rows_B, n_cols_B,
		n_rows_C, n_cols_C;

	n_rows_A = n_cols_A = n_rows_B = n_cols_B = n_rows_C = n_cols_C = 3;

	//float *h_A = (float*)malloc(n_rows_A * n_cols_A * sizeof(float));
	//float *h_B = (float*)malloc(n_rows_B * n_cols_B * sizeof(float));
	//float *h_C = (float*)malloc(n_rows_C * n_cols_C * sizeof(float));

	//// alloc 3 arrays on GPU
	//float *d_A, *d_B, *d_C;
	//cudaMalloc((void**)&d_A, n_rows_A*n_cols_A * sizeof(float));
	//cudaMalloc((void**)&d_B, n_rows_B*n_cols_B * sizeof(float));
	//cudaMalloc((void**)&d_C, n_rows_C*n_cols_C * sizeof(float));

	thrust::device_vector<float> d_A(n_rows_A*n_cols_A),
		d_B(n_rows_B*n_cols_B),
		d_C(n_rows_C*n_cols_C);


	// fill
	/*gpu_fill_rand(d_A, n_rows_A, n_cols_A);
	gpu_fill_rand(d_B, n_rows_B, n_cols_B);*/
	gpu_fill_rand0(thrust::raw_pointer_cast(&d_A[0]), n_rows_A, n_cols_A);
	gpu_fill_rand0(thrust::raw_pointer_cast(&d_B[0]), n_rows_B, n_cols_B);


	// mul
	//gpu_blas_mmul(handle, d_A, d_B, d_C, n_rows_A, n_cols_A, n_cols_B);
	gpu_blas_mmul(handle, thrust::raw_pointer_cast(&d_A[0]),
		thrust::raw_pointer_cast(&d_B[0]),
		thrust::raw_pointer_cast(&d_C[0]),
		n_rows_A, n_cols_A, n_cols_B);

	// copy back to host
	//cudaMemcpy(h_C, d_C, n_rows_C*n_cols_C * sizeof(float), cudaMemcpyDeviceToHost);

	//print_matrix(d_C, n_rows_C, n_cols_C);

	// 能否直接print device_vector ???
	//print_matrix(thrust::raw_pointer_cast(&d_C[0]), n_rows_C, n_cols_C);
	thrust::host_vector<float> h_C = d_C;
	print_matrix(thrust::raw_pointer_cast(&h_C[0]), n_rows_C, n_cols_C);

	// free
	/*cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);*/
	// device_vector 会自动删除，当fn return时

	cublasDestroy(handle);
}
