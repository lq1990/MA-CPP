#pragma once
#include "AOptimizer.h"
class Adagrad :
	public AOptimizer
{
public:
	Adagrad();
	~Adagrad();

	// 通过 AOptimizer 继承
	virtual void optimize() override;
};

