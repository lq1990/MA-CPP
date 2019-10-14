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
void Adagrad::optimize(arma::mat & P, double alpha, arma::mat dP, mat& mdP, int i)
{
	mdP += dP % dP; // 平方，求和

	// update Param
	P -= alpha * dP / arma::sqrt(mdP + pow(10, -8));


	// show learning_rate
	/*if (i % 249 == 0)
	{
		mat m_mean = arma::mean(arma::sqrt(mdP + pow(10, -8)));
		cout << "---- learning_rate roughly: " << alpha / m_mean(0, 0) << endl;
	}*/
}
