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

	res_vec = gpu_add(this->handle,
		this->d_vec, d_o, M*N);

	// update 将指针游标移动到计算结果上，这样才能实现链式向后计算新的 
	this->d_vec = res_vec;

	return this;
}

CudaUtils * CudaUtils::add(CudaUtils * cu_d_o)
{
	device_vector<float> tmp = cu_d_o->getResDevVec();
	return this->add(tmp);
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

//device_vector<float> CudaUtils::mmul(
//	device_vector<float> d_A, 
//	device_vector<float> d_B, 
//	int A_M, int A_N, int B_N)
//{
//
//	return gpu_mmul(this->handle, d_A, d_B,
//		A_M, A_N, B_N);
//}


void CudaUtils::warmup()
{
	float* h_a;
	h_a = (float*)malloc(sizeof(float));
	h_a[0] = 1.0f;

	float* d_a;
	cudaMalloc((void**)&d_a, sizeof(float));
	cudaMemcpy(d_a, h_a, sizeof(float), cudaMemcpyHostToDevice);

	cudaFree(d_a);
	free(h_a);
}

