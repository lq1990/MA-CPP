#include "SGD.h"



SGD::SGD()
{
}


SGD::~SGD()
{
}


void SGD::optimize(mat & P, double alpha, mat dP, mat & mdP, int i)
{
	// P = P - alpha * dP
	P -= alpha * dP;
}
