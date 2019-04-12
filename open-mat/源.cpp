#include <iostream>
#include <mat.h>
#include <string>
#include <armadillo>

using namespace arma;
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

void read_write01()
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
	int rows = mxGetM(Array1); // mxGetM 获取 rows数量
	int cols = mxGetN(Array1); // mxGetN 获取 cols数量
	cout << "length 数据有 " << ss << "维度" << endl;
	cout << "rows of length: " << rows << endl;
	cout << "cols of length: " << cols << endl;

	for (int i = 0; i < ss; i++)
	{
		cout << "第" << (i + 1) << "维度的大小是" << dim[i] << endl;
	}

	double *Data = (double*)mxGetData(Array1);
	// mxGetData 获取数据，需要类型强制转换，转换的类型由mat中变量存储的类型确定

	cout << "Data[0]: " << Data[0] << endl;

	// ---------------------Array2: tracts-----------------------------------
	cout << "------------------\n";
	int rows_tracts = mxGetM(Array2);
	int cols_tracts = mxGetN(Array2);
	cout << "rows_tracts: " << rows_tracts << endl;
	cout << "cols_tracts: " << cols_tracts << endl;
	double* data_tracts = (double*)mxGetData(Array2);

	// 用 arma 的mat格式数据，将matlab中数据复原成多维
	arma::mat mat_tracts(rows_tracts*cols_tracts, 1);

	for (int i = 0; i < rows_tracts*cols_tracts; i++)
	{
		// mat数据中 100行 n列，在c++中会转成一维。
		// mat数据转方式：一列一列的
		mat_tracts(i, 0) = data_tracts[i];
	}

	mat_tracts = reshape(mat_tracts, rows_tracts, cols_tracts);
	mat_tracts.print("mat_tracts");


	// ---------------------------------------------------------
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
		= matOpen("C:\\Program Files\\MATLAB\\MATLAB Production Server\\R2015a\\MA_Matlab\\outTest.mat", "w");
	// 在往 C盘的这个路径下写文件时，必须先设置对这个目录的权限为可写

	pMxArray = mxCreateDoubleMatrix(2, 3, mxREAL);
	memcpy((void*)(mxGetPr(pMxArray)), (void*)out, sizeof(double) * 6);

	Data = (double*)mxGetData(pMxArray);
	matPutVariable(OutFile, "test", pMxArray);
}

void read_write02()
{
	// 读取 dataSScaling.mat
	const char* path = "C:\\Program Files\\MATLAB\\MATLAB Production Server\\R2015a\\MA_Matlab\\Arteon\\start_loadSync\\DataFinalSave\\dataSScaling.mat";
	MATFile* file 
		= matOpen(path, "r");
	if (file == NULL)
	{
		cout << "matOpen error" << endl;
	}

	mxArray* dataSScaling = matGetVariable(file, "dataSScaling");
	int numF = mxGetNumberOfFields(dataSScaling); // 有 20 个 Fields
	cout << "#fields: " << numF << endl;

	
	mxArray* id16_start = mxGetField(dataSScaling, 0, "id16_start"); // matlab:: dataSSCaling(1).id16_start
	int numF16 = mxGetNumberOfFields(id16_start);
	cout << "#fields of id16: " << numF16 << endl;
	mxArray* id16_start_score = mxGetField(id16_start, 0, "score"); // matlab:: id16_start(1).score
	double* id16_start_score_data = (double*)mxGetData(id16_start_score); // 拿到数据指针
	cout << "id16_start_score_data: " << id16_start_score_data[0] << endl; // 取到数据


	mxArray* id17_start = mxGetField(dataSScaling, 0, "id17_start");
	int numF17 = mxGetNumberOfFields(id17_start);
	cout << "#fields of id17: " << numF17 << endl;
	const char* signal = "LongAcce"; 
	mxArray* id17_start_signal = mxGetField(id17_start, 0, signal);
	int rows_id17_start_signal = mxGetM(id17_start_signal);
	int cols_id17_start_signal = mxGetN(id17_start_signal);
	cout << "rows_id17_start_signal: " << rows_id17_start_signal << endl;
	cout << "cols_id17_start_signal: " << cols_id17_start_signal << endl;

	double* id17_start_signal_data = (double*)mxGetData(id17_start_signal);
	for (int i = 0; i < rows_id17_start_signal; i++)
	{
		cout << signal << " data: " << id17_start_signal_data[i] << endl;
	}



	matClose(file);
	return;
}


int main()
{
	read_write02();


	system("pause");
	return 0;
}