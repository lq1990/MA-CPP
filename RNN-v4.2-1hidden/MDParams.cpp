#include "MDParams.h"



MDParams::MDParams()
{
}


MDParams::~MDParams()
{
}

void MDParams::setZeros(Params * p)
{
	this->mdWfSum = arma::zeros<mat>(p->Wf.n_rows, p->Wf.n_cols);
	this->mdWiSum = arma::zeros<mat>(p->Wi.n_rows, p->Wi.n_cols);
	this->mdWcSum = arma::zeros<mat>(p->Wc.n_rows, p->Wc.n_cols);
	this->mdWoSum = arma::zeros<mat>(p->Wo.n_rows, p->Wo.n_cols);
	this->mdWhhSum = arma::zeros<mat>(p->Whh.n_rows, p->Whh.n_cols);

	this->mdbfSum = arma::zeros<mat>(p->bf.n_rows, p->bf.n_cols);
	this->mdbiSum = arma::zeros<mat>(p->bi.n_rows, p->bi.n_cols);
	this->mdbcSum = arma::zeros<mat>(p->bc.n_rows, p->bc.n_cols);
	this->mdboSum = arma::zeros<mat>(p->bo.n_rows, p->bo.n_cols);
	this->mdbhhSum = arma::zeros<mat>(p->bhh.n_rows, p->bhh.n_cols);


}
