#pragma once
#include "AOptimizer.h"
class Adadelta :
	public AOptimizer
{
public:
	Adadelta();
	~Adadelta();

	// 通过 AOptimizer 继承
	virtual void optimize() override;
};

