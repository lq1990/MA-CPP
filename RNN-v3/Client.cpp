#include <iostream>
#include <armadillo>
#include <Windows.h>

#include "IOMatlab.h"
#include "Cuda4RNN.h"

using namespace std;
using namespace arma;

void test()
{
	cout << "hello, C++" << endl;
	// ---------------- matlab -----------------
	std::vector<SceStruct> lstrain =
		IOMatlab::read("listStructTrain");

	cout << lstrain.size() << endl;
	/*SceStruct first = lstrain[0];
	cout << "id: " << first.id
		<< ", score: " << first.score << endl
		<< "dim: " << first.matDataZScore.n_rows
		<< " " << first.matDataZScore.n_cols << endl;*/
	//<< "matDataZScore: \n" << first.matDataZScore << endl;

	// ============ arma ======================
	


	// ========= cuda ==================
	cout << endl;
	Cuda4RNN::cublasDemo();
}

int main()
{
	clock_t t_begin, t_end;
	t_begin = clock();
	// ===============================================

	test();



	
	// ============================================
	t_end = clock();
	cout << "=======================\ntotal time: "
		<< (double)(t_end - t_begin) / CLOCKS_PER_SEC
		<< " s" << endl;
	
	system("pause");
	return 0;
}
