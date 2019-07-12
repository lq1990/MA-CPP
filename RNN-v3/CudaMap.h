﻿#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>


#include "MyStruct.h"


using namespace std;



/*
	key of map: index of vector 
	val of map: array

*/
class CudaMap
{
public:

	CudaMap();

	~CudaMap();

	void put(int indexAsKey, MyArray marr);

	MyArray get(int indexAsKey);

	int getSize();

	thrust::device_vector<int> getKeyListToDevice();

	thrust::host_vector<int> getKeyListToHost();


private:
	int capacity;

	thrust::device_vector<int> keyList; // keySet

	thrust::device_vector<MyArray> d_map; // use vector to implement map

};
