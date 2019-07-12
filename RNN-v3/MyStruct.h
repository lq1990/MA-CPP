#pragma once

/*
	<why>
	Since CUDA Files does not support Armadillo, 
	we use MyArray to store array.
	</why>

	<items>
	int size;			// num of all elems in array.
	int n_rows_origin;	// n_rows of orignal array
	float *arr;			// saves only the values of array in column-major and 1-dim format
	</items>

	<tip>
	Considering the original array can be 2-dim,
	With "n_rows_origin" and "arr", the original array can be reconstructed.
	</tip>
*/
typedef struct MyArray
{
	int size; // num of all elems, size = M * N = n_rows * n_cols
	int n_rows_origin;   // n_rows of original mat
	float *arr;
};
