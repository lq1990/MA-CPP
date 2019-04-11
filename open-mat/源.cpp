#include <iostream>
#include <mat.h>
#include <string>

using namespace std;

double mat_read(string path, string name, mxArray *&Array1, mxArray *&Array2)
{
	MATFile* pmatFile = matOpen((path + "\\" + name + ".mat").c_str(), "r");
	// r 只读， u 更新模式 可读可写， w 只写入 若文件不存在则新建

	if (pmatFile == NULL)
	{
		cout << "MatOpen error" << endl;
	}

	Array1 = matGetVariable(pmatFile, "length");
	Array2 = matGetVariable(pmatFile, "tracts");
	// 读取mat文件中的变量，返回一个数据阵列指针

	matClose(pmatFile);

	return 0;
}

int main()
{
	mxArray *Array1 = NULL; // length
	mxArray *Array2 = NULL; // tracts
	string filepath
		= "C:\\Program Files\\MATLAB\\MATLAB Production Server\\R2015a\\MA_Matlab";
	string filename = "InputTest";
	mat_read(filepath, filename, Array1, Array2);

	//int64_T *dim = (int64_T *)(mxGetDimensions(Array1));
	int* dim = (int*)(mxGetDimensions(Array1));

	int ss = mxGetNumberOfDimensions(Array1);
	int pp = mxGetM(Array1);
	cout << "length 数据有 " << ss << "维度" << endl;
	for (int i = 0; i < ss; i++)
	{
		cout << "第" << (i + 1) << "维度的大小是" << dim[i] << endl;
	}

	double *Data = (double*)mxGetData(Array1);

	cout << "Data[0]: " << Data[0] << endl;

	double *out = new double[6];
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			out[i * 3 + j] = 5 * j + i;
		}
	}

	MATFile* OutFile = NULL;
	mxArray* pMxArray = NULL;

	OutFile
		= matOpen("outTest.mat", "w");
	pMxArray = mxCreateDoubleMatrix(2,3,mxREAL);
	memcpy((void*)(mxGetPr(pMxArray)), (void*)out, sizeof(double) * 6);

	Data = (double*)mxGetData(pMxArray);
	matPutVariable(OutFile, "test", pMxArray);


	system("pause");
	return 0;
}