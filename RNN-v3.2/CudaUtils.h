#pragma once

/*
	注意：fns的核心实现在 gpu_fns.h .cu中
	即使不用此类，也可以去用 gpu_fns。

	此类特点：
	把此类做成一个包装类，实现功能如下：
	A.mmul(B).mmul(C) 链式效果
	=> A*B*C

	使用须知：
	CudaUtils* cuA = new CudaUtils(handle, A, am, an); // 实例了一个包装对象 cuA，
	cuA.mmul(B, bm, bn).add(v); // 随着链条的往后，cuA已经被update了。

	// 若想重头计算另外一组计算的话，需要重新new一个实例。不能在cuA的基础上计算，因为cuA还保留着上个链条的结果。
	// 为避免误用，简单起见，使用匿名实例，如下
	thrust::device_vector<float> y = new CudaUtils(handle, M, 3, 2)->mv(v)->getResDevVec();

*/

#include <iostream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "curand.h"
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "gpu_fns.h"

using namespace std;
using namespace thrust;

class CudaUtils
{
public:
	CudaUtils();
	CudaUtils(cublasHandle_t handle, 
		device_vector<float> d_vec, int M, int N);
	~CudaUtils();

	/*
		this->d_vec mul other(om, on)
	*/
	CudaUtils* mmul(device_vector<float> d_o, int om, int on);

	/*
		matrix-vector multiplication
	*/
	CudaUtils* mv(device_vector<float> d_v, bool isMT=false);

	CudaUtils* add(device_vector<float> d_o);

	CudaUtils* add(CudaUtils* cu_d_o);

	CudaUtils* mul_elemwise(device_vector<float> d_o);

	CudaUtils* tanh();

	CudaUtils* scal(float alpha);

	/*
		if d_vec stores data of a matrix,
		getRow(r) returns the rth row,
		return format is also the wrapped vector => CudaUtils*

		r: 0-based indexing
	*/
	CudaUtils* getRow(int r);

	device_vector<float> getResDevVec();
	int getResM();
	int getResN();

private:
	cublasHandle_t handle;

	/*
		一对一关系
		每个vector都有自己的 包装 CudaUtils类.

		在计算前后，vec是不一样的。
		计算后，vec就会更新为计算的结果
	*/
	device_vector<float> d_vec;
	int M; // n_rows of d_vec
	int N; // n_cols of d_vec
	int size; // num of elems of d_vec = M*N

};

