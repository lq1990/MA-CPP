#include "SGD.h"



SGD::SGD()
{
}


SGD::~SGD()
{
}


void SGD::optimize(Params * P, double alpha, DParams* dP, MDParams* mdP, int i)
{
	// P = P - alpha * dP
	//P -= alpha * dP;

	P->Wf -= alpha * dP->dWfSum;
	P->Wi -= alpha * dP->dWiSum;
	P->Wc -= alpha * dP->dWcSum;
	P->Wo -= alpha * dP->dWoSum;
	P->Whh -= alpha * dP->dWhhSum;

	P->bf -= alpha * dP->dbfSum;
	P->bi -= alpha * dP->dbiSum;
	P->bc -= alpha * dP->dbcSum;
	P->bo -= alpha * dP->dboSum;
	P->bhh -= alpha * dP->dbhhSum;

}
