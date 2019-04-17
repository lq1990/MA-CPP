#pragma once
#include "AOptimizer.h"
class Rmsprop :
	public AOptimizer
{
public:
	Rmsprop();
	~Rmsprop();

	// 通过 AOptimizer 继承
	virtual void optimize() override;
};

