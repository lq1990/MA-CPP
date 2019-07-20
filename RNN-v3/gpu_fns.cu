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

	return d_C;
}

device_vector<float> gpu_mv(cublasHandle_t handle, 
	device_vector<float> d_A,
	device_vector<float> d_x,
	int am, int an, bool isMT)
{
	float alpha = 1.0f;
	float beta = 0.0f;
	if (isMT)
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

device_vector<float> gpu_add(device_vector<float> d_x,
	device_vector<float> d_y,
	int size)
{
	device_vector<float> d_z(size);

	thrust::transform(d_x.begin(), d_x.end(), d_y.begin(),
		d_z.begin(), thrust::plus<float>());

	return d_z;
}

device_vector<float> gpu_mul_elemwise(device_vector<float> d_x,
	device_vector<float> d_y,
	int size)
{
	// z = x .* y
	device_vector<float> d_z(size);

	thrust::transform(d_x.begin(), d_x.end(), d_y.begin(),
		d_z.begin(), thrust::multiplies<float>());

	cudaDeviceSynchronize();
	return d_z;
}

device_vector<float> gpu_tanh(device_vector<float> d_x, int size)
{
	// y = tanh(x)
	thrust::device_vector<float> d_y(size);

	thrust::transform(d_x.begin(), d_x.end(), 
		d_y.begin(),
		tanh_functor());

	return d_y;
}

device_vector<float> gpu_scal(device_vector<float> d_x, int size, float alpha)
{
	device_vector<float> d_y(size);
	device_vector<float> d_alpha(size, alpha);

	thrust::transform(d_x.begin(), d_x.end(), 
		d_alpha.begin(), 
		d_y.begin(),
		thrust::multiplies<float>());

	return d_y;
}

device_vector<float> gpu_get_col(device_vector<float> d_A, int M, int N, int col)
{

	device_vector<float> res_vec(M);
	int begin = col * M; // [begin, end)
	int end = (col+1) * M;
	thrust::copy(d_A.begin() + begin, d_A.begin() + end, res_vec.begin());

	return res_vec;
}

void gpu_set_col(device_vector<float>& d_A, int M, int N, 
	int col, device_vector<float> values)
{

	int setBeginIdx = col * M;
	thrust::copy(values.begin(), values.end(), d_A.begin() + setBeginIdx);

}


device_vector<float> gpu_get_row(device_vector<float> d_A, 
	thrust::device_vector<int> rowLoc, 
	int an)
{
	// 每隔一个间断取值，涉及到 strided_range, 此处先用普通方法
	/*thrust::device_vector<int> rowLoc(2);
	rowLoc[0] = 0;
	rowLoc[1] = 3;*/

	thrust::device_vector<float> targets(an);

	thrust::transform(thrust::make_permutation_iterator(d_A.begin(),
		rowLoc.begin()),
		thrust::make_permutation_iterator(d_A.begin(),
			rowLoc.end()),
		targets.begin(), thrust::identity<float>());

	return targets;
}

void gpu_set_row(device_vector<float>& d_A, int M, int N, 
	int row,
	device_vector<float> values, 
	bool isColMajor)
{
	if (isColMajor)
	{
		// column-major

	}
	else
	{
		// 对于 lossFun中的 xs, hs, ys, ps都是 row-major

		int begin = N * row;
		thrust::copy(values.begin(), values.end(), d_A.begin() + begin);
		// 有问题, copy是按照列优先做的，与自定义的行优先A 冲突

		// =>
		// create rowLoc;
		device_vector<int> rowLoc(N); // matrix第r行元素的index
		for (int i = 0; i < N; i++)
		{
			rowLoc[i] = i * M + row;
		}


	}
}

void gpu_fill_rand(device_vector<float>& d_A, 
	int n_rows, int n_cols,
	float low, float high, int seed)
{
	thrust::counting_iterator<int> index_seq_begin(0);

	thrust::transform(index_seq_begin,
		index_seq_begin + n_rows * n_cols,
		d_A.begin(),
		prg(low, high, seed));

	return;
}

device_vector<float> gpu_generate_rand(int n_rows, int n_cols,
	float low, float high, int seed)
{
	device_vector<float> d_A(n_rows * n_cols);

	thrust::counting_iterator<int> index_seq_begin(0);

	thrust::transform(index_seq_begin,
		index_seq_begin + n_rows * n_cols,
		d_A.begin(),
		prg(low, high, seed));

	return d_A;
}

device_vector<float> gpu_softmax(device_vector<float> d_x, int size)
{
	// declare
	device_vector<float> exp_x(size); // saves exp(x)
	float sum_exp_x; // sum of exp(x)
	device_vector<float> exp_div_sum(size);

	// compute
	/*thrust::transform(d_x.begin(), d_x.end(), exp_x.begin(), exp_functor());
	sum_exp_x = thrust::reduce(exp_x.begin(), exp_x.end());
	device_vector<float> sum_vec(size, sum_exp_x);

	thrust::transform(exp_x.begin(), exp_x.end(), 
		sum_vec.begin(),
		exp_div_sum.begin(), 
		thrust::divides<float>());*/

	// transform_reduce 优化，可减少使用一个kernel
	sum_exp_x = thrust::transform_reduce(d_x.begin(), d_x.end(), exp_functor(), 
		0.000f, 
		thrust::plus<float>());

	thrust::transform(d_x.begin(), d_x.end(), 
		exp_div_sum.begin(), 
		exp_div_sum_functor(sum_exp_x));

	return exp_div_sum;
}

int gpu_max_index(device_vector<float> d_x)
{
	device_vector<float>::iterator iter =
		thrust::max_element(d_x.begin(), d_x.end());

	int pos = iter - d_x.begin();
	return pos;
}

float gpu_max_value(device_vector<float> d_x)
{
	device_vector<float>::iterator iter =
		thrust::max_element(d_x.begin(), d_x.end());

	float max_val = *iter;
	return max_val;
}

void gpu_clip(device_vector<float>& d_x, float lowMargin, float highMargin)
{
	thrust::transform(d_x.begin(), d_x.end(), d_x.begin(), 
		clip_functor(lowMargin, highMargin));
}

void printToHost(device_vector<float> d_A, int n_rows, int n_cols, string title)
{
	cout << title << endl;
	
	host_vector<float> h_A = d_A;
	for (int i = 0; i < n_rows; i++)
	{
		for (int j = 0; j < n_cols; j++)
		{
			cout << h_A[j*n_rows + i] << "  ";
		}
		cout << endl;
	}

	cout << endl;
}

void printToHost(device_vector<int> d_A, int n_rows, int n_cols, string title)
{
	cout << title << endl;

	host_vector<int> h_A = d_A;
	for (int i = 0; i < n_rows; i++)
	{
		for (int j = 0; j < n_cols; j++)
		{
			cout << h_A[j*n_rows + i] << "  ";
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

void gpu_warmup()
{
	float* h_a;
	h_a = (float*)malloc(10000*sizeof(float));
	h_a[0] = 1.0f;

	float* d_a;
	cudaMalloc((void**)&d_a, 10000*sizeof(float));
	cudaMemcpy(d_a, h_a, 10000*sizeof(float), cudaMemcpyHostToDevice);

	cudaFree(d_a);
	free(h_a);

	thrust::device_vector<float> vec(10);
	for (int i = 0; i < 10; i++)
	{
		vec[i] = 1.0f;
	}
}