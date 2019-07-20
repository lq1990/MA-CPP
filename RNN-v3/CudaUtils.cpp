#include "CudaUtils.h"



CudaUtils::CudaUtils()
{
	// if not use CudaUtils(cublasHandle_t handle)
}

CudaUtils::CudaUtils(cublasHandle_t handle, 
	device_vector<float> d_vec, int M, int N)
{
	this->handle = handle;
	this->d_vec = d_vec;
	this->M = M;
	this->N = N;
	this->size = M * N;
}


CudaUtils::~CudaUtils()
{
}




CudaUtils * CudaUtils::mmul(device_vector<float> d_o, int om, int on)
{
	int M = this->M;
	int N = this->N;

	device_vector<float> res_vec;
	res_vec =
		gpu_mmul(this->handle, 
			this->d_vec, d_o, 
			M, N, on);

	// box the res_vec, and return the boxed res
	/*CudaUtils* cu_res = new CudaUtils(this->handle);
	cu_res->box(this->res_vec, M, on);*/

	// update 将指针游标移动到计算结果上，这样才能实现链式向后计算新的 
	this->d_vec = res_vec;
	this->M = M;
	this->N = on;
	this->size = M * on;

	// this 指针难道没更新？
	return this; // 返回一个全新的 CudaUtils，和this无关了

	// 解决方法：只用类似游标，随着链式计算往后，游标或指针不断往后
}

CudaUtils * CudaUtils::mv(device_vector<float> d_v, bool isMT)
{
	int M = this->M;
	int N = this->N;

	device_vector<float> res_vec;

	res_vec = gpu_mv(this->handle,
		this->d_vec, d_v,
		M, N, isMT);

	// update 将指针游标移动到计算结果上，这样才能实现链式向后计算新的 
	this->d_vec = res_vec;
	if (isMT)
	{
		this->M = N;
		this->N = 1;
		this->size = N * 1;
	}
	else
	{
		this->M = M;
		this->N = 1;
		this->size = M * 1;
	}

	return this;
}

CudaUtils * CudaUtils::add(device_vector<float> d_o)
{
	int M = this->M;
	int N = this->N;

	device_vector<float> res_vec;

	res_vec = gpu_add(this->d_vec, d_o, M*N);

	// update 将指针游标移动到计算结果上，这样才能实现链式向后计算新的 
	this->d_vec = res_vec;

	return this;
}

CudaUtils * CudaUtils::add(CudaUtils * cu_d_o)
{
	device_vector<float> tmp = cu_d_o->getResDevVec();
	return this->add(tmp);
}

CudaUtils * CudaUtils::mul_elemwise(device_vector<float> d_o)
{
	int M = this->M;
	int N = this->N;

	device_vector<float> res_vec;
	res_vec = gpu_mul_elemwise(this->d_vec, d_o, M*N);

	// update this
	this->d_vec = res_vec;

	return this;
}

CudaUtils * CudaUtils::tanh()
{
	int M = this->M;
	int N = this->N;

	thrust::device_vector<float> d_y(M*N);

	thrust::transform(this->d_vec.begin(), this->d_vec.end(),
		d_y.begin(),
		tanh_functor());

	// update
	this->d_vec = d_y;
	return this;
}

CudaUtils * CudaUtils::scal(float alpha)
{
	int M = this->M;
	int N = this->N;

	device_vector<float> d_y(M*N);
	device_vector<float> d_alpha(M*N, alpha);

	thrust::transform(this->d_vec.begin(), this->d_vec.end(),
		d_alpha.begin(),
		d_y.begin(),
		thrust::multiplies<float>());

	// update 
	this->d_vec = d_y;
	return this;
}

CudaUtils * CudaUtils::getRow(int r)
{
	int M = this->M;
	int N = this->N;

	// rowLoc
	device_vector<int> rowLoc(N); // matrix第r行元素的index
	for (int i = 0; i < N; i++)
	{
		rowLoc[i] = i * M + r; 
		// 此处使用了loop，但时间复杂度是常数级别，因为场景矩阵的列数N是常数。
		// 因此运算效率不低。
	}

	device_vector<float> res =
		gpu_get_row(this->d_vec, rowLoc, N);

	// update this
	this->d_vec = res;
	this->M = N;
	this->N = 1;
	this->size = N * 1;

	return this;
}




device_vector<float> CudaUtils::getResDevVec()
{
	return this->d_vec;
}

int CudaUtils::getResM()
{
	return this->M;
}

int CudaUtils::getResN()
{
	return this->N;
}



