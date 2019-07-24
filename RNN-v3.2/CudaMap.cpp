#include "CudaMap.h"

CudaMap::CudaMap()
{
	this->capacity = 256;
	thrust::device_vector<MyArray*> vec(capacity);
	this->d_map = vec;
}

CudaMap::~CudaMap()
{
}

void CudaMap::put(int indexAsKey, MyArray* marr)
{
	int key = indexAsKey + 1; // key有可能是 -1，避免-1，所以key都+1
	while (key >= this->capacity)
	{
		this->capacity <<= 1;
		thrust::device_vector<MyArray*> tmp = this->d_map;
		thrust::device_vector<MyArray*> newMap(this->capacity);
		thrust::copy(tmp.begin(), tmp.end(), newMap.begin());
		this->d_map = newMap;
	}

	this->d_map[key] = marr;
	
	// 判断 key是否已经存在于map中
	int flag = 0;
	for (int i = 0; i < this->keyList.size(); i++)
	{
		int k = this->keyList[i];
		if (k == indexAsKey)
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

MyArray* CudaMap::get(int indexAsKey)
{
	int key = indexAsKey + 1;
	MyArray* marr = this->d_map[key];
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

void CudaMap::printToHost(string title)
{
	cout << "---------- " << title << " ----------------\n";
	cout << "size: " << this->getSize() << endl;
	thrust::host_vector<int> keyList = this->getKeyListToHost();
	cout << "keyList:\t";
	for (int i = 0; i < keyList.size(); i++)
	{
		cout << keyList[i] << "\t";
	}
	cout << endl;

	for (int i = 0; i < this->getSize(); i++)
	{
		int key = this->getKeyListToHost()[i];
		cout << "key: " << key << endl;
		MyArray* marr = this->get(key);
		if (marr == nullptr)
		{
			return;
		}
		float* arr = marr->arr;
		cout << "values: " << "\n";
		for (int j = 0; j < marr->size; j++)
		{
			cout << arr[j] << "\n";
		}
		cout << "n_elems: " << marr->size
			<< ", n_rows: " << marr->n_rows_origin << "\n\n";
	}
	cout << "--------------------------------------------\n";
}
