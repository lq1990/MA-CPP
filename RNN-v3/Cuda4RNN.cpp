#include "Cuda4RNN.h"

/*
	注意：cuda编译的文件中，不能有Armadillo
*/


Cuda4RNN::Cuda4RNN()
{
}


Cuda4RNN::~Cuda4RNN()
{
}

CudaMap Cuda4RNN::lossFun(MyArray inputs, 
	float score, 
	MyArray hprev, 
	device_vector<float>& true_false, 
	device_vector<float>& log_target, 
	device_vector<float>& log_prediction)
{

	CudaMap marr;
	return marr;
}

