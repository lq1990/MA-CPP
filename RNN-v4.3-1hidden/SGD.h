#pragma once
#include "AOptimizer.h"


class SGD :
	public AOptimizer
{
public:
	SGD();
	~SGD();



	// 通过 AOptimizer 继承
	virtual void optimize(Params * P, double alpha, DParams* dP, MDParams* mdP, int i) override;

};

