#include "MyArray.h"


MyArray::MyArray()
{
}

MyArray::MyArray(int size, int n_rows, float *arr)
{
	this->size = size;
	this->n_rows_origin = n_rows;
	this->arr = arr;
}


int MyArray::createRows()
{
	if (this->size <= 0)
	{
		cout << "can not create rows from arr, because no elems in arr" << endl;
		return -1;
	}

	int M = this->n_rows_origin;
	int N = this->size / M;
	thrust::device_vector<MyArray*> rows_vec;

	/*
		此处可以改进，只需遍历一遍 arr。而不需要外层的遍历。
	*/
	for (int r = 0; r < M; r++)
	{
		MyArray *marr = new MyArray();
		float *arr;
		arr = (float*)malloc(N * sizeof(float));

		for (int i = 0, count = 0; i < this->size; i++)
		{
			if (i % M == r)
			{
				arr[count++] = this->arr[i];
			}
		}

		marr->arr = arr;
		marr->size = N;
		marr->n_rows_origin = N;

		rows_vec.push_back(marr);
	}

	this->rows = rows_vec;

	return 0;
}


//MyArray * MyArray::getRow(int row)
//{
//	int M = this->n_rows_origin;
//	int N = this->size / M;
//
//	float *arr;
//	arr = (float*)malloc(N * sizeof(float));
//	int count = 0;
//	for (int i = 0; i < this->size; i++)
//	{
//		if (i % M == row)
//		{
//			//float val = this.
//			arr[count++] = i;
//		}
//	}
//
//	MyArray *marrRow = new MyArray();
//	marrRow->n_rows_origin = N;
//	marrRow->size = N;
//	marrRow->arr = arr;
//
//	return marrRow;
//}



MyArray * MyArray::getRowToDevice(int index)
{
	if (this->size <= 0) // size of arr
	{
		cout << "no elems to get" << endl;
		return nullptr;
	}

	// 当用到 getRow 时，才createRow。因为不是所有的MyArray都用到getRow
	if (this->rows.size() <= 0) // only when rows is not created yet
	{
		this->createRows();
	}

	// 当rows已经创建了，直接从 rows中取。类似缓存的效果。
	return this->rows[index];
}

MyArray * MyArray::getRowToHost(int index)
{
	if (this->size <= 0) // size of arr
	{
		cout << "no elems to get" << endl;
		return nullptr;
	}

	if (this->rows.size() <= 0) // only when rows is not created yet
	{
		this->createRows();
	}

	thrust::host_vector<MyArray*> h_vec = this->rows;
	return h_vec[index];
}


void MyArray::printToHost(float *arr, int size, string title, bool isCol)
{
	cout << "------------ " << title << " --------------\n";

	if (isCol)
	{
		for (int i = 0; i < size; i++)
		{
			printf("%.3f\n", arr[i]);
		}
	}
	else
	{
		for (int i = 0; i < size; i++)
		{
			printf("%.3f  ", arr[i]);
		}
		cout << endl;
	}
	cout << "size: " << size << endl;

	cout << "---------------------------------------------\n";
}

void MyArray::printMatToHost(float *arr, int M, int N, string title)
{
	cout << "------------ " << title << " --------------\n";

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			//cout << arr[j*M + i] << "  ";
			printf("%.3f\t", arr[j*M + i]);
		}
		cout << endl;
	}
	cout << "n_rows: " << M << ", n_cols: " << N << endl;
	cout << "---------------------------------------------\n";
}

MyArray * MyArray::randn(int M, int N, float factor)
{
	float* arr;
	arr = (float*)malloc(M*N * sizeof(float));
	for (int i = 0; i < M*N; i++)
	{
		arr[i] = (rand() % 2000 - 1000.0f) / 1000.0f * factor;
	}

	MyArray* marr = new MyArray();
	marr->arr = arr;
	marr->n_rows_origin = M;
	marr->size = M * N;

	return marr;
}
