#include "Adagrad.h"



Adagrad::Adagrad()
{
}


Adagrad::~Adagrad()
{
}

/*
	P -= alpha * dPt /norm2(dP0, dP1, dP2,..., dPt)
*/
void Adagrad::optimize(Params* P, double alpha, DParams* dP, MDParams* mdP, int i)
{
	/*
	mdP += dP % dP; // 平方，求和

		// update Param
	P -= alpha * dP / arma::sqrt(mdP + pow(10, -8));
	*/

	mdP->mdWfSum += dP->dWfSum % dP->dWfSum;
	mdP->mdWiSum += dP->dWiSum % dP->dWiSum;
	mdP->mdWcSum += dP->dWcSum % dP->dWcSum;
	mdP->mdWoSum += dP->dWoSum % dP->dWoSum;
	mdP->mdWhhSum += dP->dWhhSum % dP->dWhhSum;

	mdP->mdbfSum += dP->dbfSum % dP->dbfSum;
	mdP->mdbiSum += dP->dbiSum % dP->dbiSum;
	mdP->mdbcSum += dP->dbcSum % dP->dbcSum;
	mdP->mdboSum += dP->dboSum % dP->dboSum;
	mdP->mdbhhSum += dP->dbhhSum % dP->dbhhSum;

	// update params
	P->Wf -= alpha * dP->dWfSum / arma::sqrt(mdP->mdWfSum + pow(10, -8));
	P->Wi -= alpha * dP->dWiSum / arma::sqrt(mdP->mdWiSum + pow(10, -8));
	P->Wc -= alpha * dP->dWcSum / arma::sqrt(mdP->mdWcSum + pow(10, -8));
	P->Wo -= alpha * dP->dWoSum / arma::sqrt(mdP->mdWoSum + pow(10, -8));
	P->Whh -= alpha * dP->dWhhSum / arma::sqrt(mdP->mdWhhSum + pow(10, -8));

	P->bf -= alpha * dP->dbfSum / arma::sqrt(mdP->mdbfSum + pow(10, -8));
	P->bi -= alpha * dP->dbiSum / arma::sqrt(mdP->mdbiSum + pow(10, -8));
	P->bc -= alpha * dP->dbcSum / arma::sqrt(mdP->mdbcSum + pow(10, -8));
	P->bo -= alpha * dP->dboSum / arma::sqrt(mdP->mdboSum + pow(10, -8));
	P->bhh -= alpha * dP->dbhhSum / arma::sqrt(mdP->mdbhhSum + pow(10, -8));

}
