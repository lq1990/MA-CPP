#pragma once

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

template <class T>

/*
	使用CUDA和Class，
	尝试实现OOP
*/
class cuda_class_dev_array
{
public:
	explicit cuda_class_dev_array() : start_(0), end_(0) {}

	explicit cuda_class_dev_array(size_t size)
	{
		allocate(size);
	}

	~cuda_class_dev_array()
	{
		free();
	}

	void resize(size_t size)
	{
		free();
		allocate(size);
	}

	size_t getSize() const
	{
		return end_ - start_;
	}

	const T* getData() const
	{
		return start_;
	}

	T* getData()
	{
		return start_;
	}

	void set(const T* src, size_t size)
	{
		size_t min = std::min(size, getSize());
		cudaError_t result = cudaMemcpy(start_, src, min * sizeof(T), 
			cudaMemcpyHostToDevice);
		if (result != cudaSuccess)
		{
			throw std::runtime_error("failed to copy to device mem");
		}
	}

	void get(T* dest, size_t size)
	{
		size_t min = std::min(size, getSize());
		cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), 
			cudaMemcpyDeviceToHost);
		if (result != cudaSuccess)
		{
			throw std::runtime_error("failed to copy back to host");
		}
	}


private:

	void allocate(size_t size)
	{
		cudaError_t result = cudaMalloc((void**)&start_, size * sizeof(T));
		if (result != cudaSuccess)
		{
			start_ = end_ = 0;
			throw std::runtime_error("failed to allocate dev mem");
		}
		end_ = start_ + size;
	}

	void free()
	{
		if (start_ != 0)
		{
			cudaFree(start_);
			start_ = end_ = 0;
		}
	}

	T* start_;
	T* end_;

};