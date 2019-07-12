/*
	RNN version 3.

	The cores of RNN-v3 are CUDA and LSTM.

	author: LQ
*/

#include <iostream>
#include <armadillo>
#include <Windows.h>

#include "IOMatlab.h"
#include "Cuda4RNN.h"
#include "CudaMap.h"
#include "Script.h"
#include "MyClock.h"

using namespace std;
using namespace arma;

int main()
{
	MyClock mclock = MyClock("main");
	// ===============================================
	
	

	
	// ============================================
	mclock.showDuration();
	std::system("pause");
	return 0;
}
