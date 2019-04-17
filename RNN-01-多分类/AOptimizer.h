#pragma once
class AOptimizer
{
public:
	AOptimizer() {};
	virtual ~AOptimizer() {};
	virtual void optimize() = 0;
};

