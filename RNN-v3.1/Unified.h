#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
	cudaMallocManaged
	alloc mem on CPU/GPU for classes
*/
class Unified
{
public:
	Unified();
	~Unified();

	/**
		Allocate instances in CPU/GPU unified memory
	*/
	void* operator new(size_t len);
	void operator delete(void* ptr);

	/**
		Allocate all arrays in CPU/GPU unified memory
	*/
	void* operator new[](std::size_t size);
	void operator delete[](void* ptr);

};

