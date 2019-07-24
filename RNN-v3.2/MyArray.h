#pragma once
#include <string>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace std;

/*
	<why>
	MyArray replaces arma::mat.
	Since CUDA Files does not support Armadillo, 
	we use MyArray to store array.
	</why>

	<items>
	int size;			// num of all elems in array.
	int n_rows_origin;	// n_rows of orignal array
	float *arr;			// saves only the values of array in column-major and 1-dim format
	thrust::device_vector<MyArray*> rows; // elem in vec: one row
	</items>

	<tip>
	Considering the original array can be 2-dim,
	With "n_rows_origin" and "arr", the original array can be reconstructed.
	</tip>
*/
class MyArray
{
public:
	int size; // num of all elems, size = M * N = n_rows * n_cols
	int n_rows_origin;   // n_rows of original mat
	float *arr; // pointer to all elems
	thrust::device_vector<MyArray*> rows; // prepared here in order to getRow() faster

	MyArray();
	MyArray(int size, int n_rows, float *arr);
	

	/*
		similar to arma::mat.row(i)
		For matrix.
		getRow from thrust::device_vector<MyArray*> rows

		index: 0-based indexing

		但是每次get都要重新遍历，效率低。
		因此，先将每行提前准备保存好
	*/
	MyArray* getRowToDevice(int index);

	MyArray* getRowToHost(int index);


	/*
		print info of MyArray to the Host.
		print out "arr" column-wise as default.
	*/
	void printToHost(float *arr, int size, string title, bool isCol=true);

	/*
		for 2-dim matrix.
	*/
	void printMatToHost(float *arr, int M, int N, string title);

	/*
		similar to arma::randn(int M, int N);
		create a matrix(M,N) with random values.

		alpha:: scale factor
	*/
	static MyArray* randn(int M, int N, float factor = 0.1);

private:
	/*
		use "arr" to create "rows"
	*/
	int createRows();
};

