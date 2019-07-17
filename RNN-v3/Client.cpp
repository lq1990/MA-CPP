/*
	RNN version 3.

	The cores of RNN-v3 are CUDA and LSTM.

	author: LQ
*/

#include <iostream>
#include <Windows.h>

#include "IOMatlab.h"
#include "Cuda4RNN.h"
#include "CudaMap.h"
#include "MyClock.h"

// for Test
#include "CudaScript.h"
#include "gpu_fns.h"
#include "CudaUtils.h"



using namespace std;

void test()
{
	// -------------- gpu_fns --------------------------------
	Cuda4RNN::test_gpu_fns_CudaUtils();

	

	// --------- SceStruct of Scenarios from IOMatlab --------------
	/*
	thrust::host_vector<SceStruct> vec =
		IOMatlab::read("listStructTrain"); // read返回device，但被host强转
	cout << "iomatlab size: " << vec.size() << endl;

	// first scenario {id, score, matDataZScore}
	SceStruct first = vec[0];
	cout << "id: " << first.id
		<< ", score: " << first.score
		<< endl;

	MyArray *matData = first.matDataZScore;
	cout << "matData size: " << matData->size << endl;

	MyArray *row0 = matData->getRowToHost(0);

	// row0 of matData
	row0->printToHost(row0->arr, row0->size, "row0", false);

	matData->printMatToHost(matData->arr,
		matData->n_rows_origin,
		matData->size / matData->n_rows_origin,
		"fist matDataZScore");
	*/

	// -------------- test Cuda4RNN --------------------
	/*Cuda4RNN cuRNN = Cuda4RNN();

	MyArray* inputs = MyArray::randn(200, 17);
	float score = 7.0f;
	MyArray* hprev = MyArray::randn(50, 1);
	device_vector<float> true_false;
	device_vector<float> log_target;
	device_vector<float> log_prediction;

	CudaMap map = cuRNN.lossFun(inputs, score, hprev,
		true_false, log_target, log_prediction);
	map.printToHost("result of lossFun");*/

	// ---------- test lossFun --------------
	/*MyArray* inputs = new MyArray();
	float score = 0;
	MyArray* hprev = new MyArray();
	device_vector<float> true_false;
	device_vector<float> log_target;
	device_vector<float> log_prediction;

	CudaMap mp = cuRNN.lossFun(inputs, score, hprev,
		true_false, log_target, log_prediction);
	mp.printToHost("res of lossFun");*/

	// show Wxh
	/*MyArray* Wxh = MyParams::Wxh;
	Wxh->printMatToHost(Wxh->arr,
		Wxh->n_rows_origin,
		Wxh->size / Wxh->n_rows_origin,
		"Wxh");*/

	// test CudaMap
	/*CudaMap map = CudaMap();
	map.put(-1, MyArray::randn(3, 2));
	map.put(2, MyArray::randn(3, 4));

	map.printToHost("test map -1");*/

	//// --------- test score2onehot ----------
	//MyArray* marr = cuRNN.score2onehot(7.0);
	//marr->printToHost(marr->arr, marr->size, "score2onehot(7.0)");

}


int main()
{
	CudaUtils().warmup();
	MyClock mclock = MyClock("main");
	// ===============================================
	

	test();
	

	// ============================================
	mclock.stopAndShow();
	std::system("pause");
	return 0;
}
