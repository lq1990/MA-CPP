#include "Unified.h"



Unified::Unified()
{
}


Unified::~Unified()
{
}

void * Unified::operator new(size_t len)
{
	void* ptr;
	cudaMallocManaged(&ptr, len);

	return ptr;
}

void Unified::operator delete(void * ptr)
{
	cudaFree(ptr);
}

void * Unified::operator new[](std::size_t size)
{
	void* ptr;
	cudaMallocManaged(&ptr, size);

	return ptr;
}

void Unified::operator delete[](void * ptr)
{
	cudaFree(ptr);
}
