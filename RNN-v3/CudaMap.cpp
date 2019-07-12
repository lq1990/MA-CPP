#include "CudaMap.h"

CudaMap::CudaMap()
{
	this->capacity = 256;
	thrust::device_vector<MyArray> vec(capacity);
	this->d_map = vec;
}

CudaMap::~CudaMap()
{
}

void CudaMap::put(int indexAsKey, MyArray marr)
{
	while (indexAsKey >= this->capacity)
	{
		this->capacity <<= 1;
		thrust::device_vector<MyArray> tmp = this->d_map;
		thrust::device_vector<MyArray> newMap(this->capacity);
		thrust::copy(tmp.begin(), tmp.end(), newMap.begin());
		this->d_map = newMap;
	}

	this->d_map[indexAsKey] = marr;
	
	// 判断 key是否已经存在于map中
	int flag = 0;
	for (int i = 0; i < this->keyList.size(); i++)
	{
		int val = this->keyList[i];
		if (val == indexAsKey)
		{
			flag = 1;
			break;
		}
	}
	if (flag == 0)
	{
		this->keyList.push_back(indexAsKey);
	}
}

MyArray CudaMap::get(int indexAsKey)
{
	MyArray marr = this->d_map[indexAsKey];
	return marr;
}

int CudaMap::getSize()
{
	return this->keyList.size();
}

thrust::device_vector<int> CudaMap::getKeyListToDevice()
{
	return this->keyList;
}

thrust::host_vector<int> CudaMap::getKeyListToHost()
{
	thrust::host_vector<int> h_vec = this->keyList;
	return h_vec;
}
