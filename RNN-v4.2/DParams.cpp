#include "DParams.h"



DParams::DParams()
{
}


DParams::~DParams()
{
}

void DParams::setZeros(Params * p)
{
	this->dWfSum  = arma::zeros<mat>(p->Wf.n_rows, p->Wf.n_cols);
	this->dWiSum  = arma::zeros<mat>(p->Wi.n_rows, p->Wi.n_cols);
	this->dWcSum  = arma::zeros<mat>(p->Wc.n_rows, p->Wc.n_cols);
	this->dWoSum  = arma::zeros<mat>(p->Wo.n_rows, p->Wo.n_cols);
	this->dWhhSum = arma::zeros<mat>(p->Whh.n_rows, p->Whh.n_cols);

	this->dbfSum  = arma::zeros<mat>(p->bf.n_rows, p->bf.n_cols);
	this->dbiSum  = arma::zeros<mat>(p->bi.n_rows, p->bi.n_cols);
	this->dbcSum  = arma::zeros<mat>(p->bc.n_rows, p->bc.n_cols);
	this->dboSum  = arma::zeros<mat>(p->bo.n_rows, p->bo.n_cols);
	this->dbhhSum = arma::zeros<mat>(p->bhh.n_rows, p->bhh.n_cols);


}
