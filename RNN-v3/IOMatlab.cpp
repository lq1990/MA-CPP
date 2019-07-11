#include "IOMatlab.h"



IOMatlab::IOMatlab()
{
}


IOMatlab::~IOMatlab()
{
}

/*
	fileName:: listStructTrain, listStructCV
*/
vector<SceStruct> IOMatlab::read(const char * fileName)
{
	vector<SceStruct> vec;

	const char* dir = "C:/Program Files/MATLAB/MATLAB Production Server/R2015a/MA_Matlab/Arteon/start_loadSync/DataFinalSave/list_data/";
	const char* tail = ".mat";
	char path[1000];
	strcpy_s(path, dir);
	strcat_s(path, fileName);
	strcat_s(path, tail);

	MATFile* file = matOpen(path, "r");
	mxArray* listStructTrain = matGetVariable(file, fileName);

	int numElems = mxGetNumberOfElements(listStructTrain); // 27 scenarios
	int numFields = mxGetNumberOfFields(listStructTrain); // 6 fields: {id, score, details, matData, matDataPcAll, matDataZScore}
	//cout << numElems << ", " << numFields << endl;

	// traverse rows, i.e. scenarios
	for (int i = 0; i < numElems; i++)
	{
		SceStruct ms;

		mxArray* item_id = mxGetField(listStructTrain, i, "id");
		mxArray* item_score = mxGetField(listStructTrain, i, "score");
		mxArray* item_matData = mxGetField(listStructTrain, i, "matData");
		mxArray* item_matDataZScore = mxGetField(listStructTrain, i, "matDataZScore");

		double* id = (double*)mxGetData(item_id);
		double* score = (double*)mxGetData(item_score);
		double* matData = (double*)mxGetData(item_matData);
		int rows_matData = (int)mxGetM(item_matData); // n_rows of matData 
		int cols_matData = (int)mxGetN(item_matData); // n_cols of matData
		double* matDataZScore = (double*)mxGetData(item_matDataZScore);
		int rows_matDataZScore = (int)mxGetM(item_matDataZScore);
		int cols_matDataZScore = (int)mxGetN(item_matDataZScore);

		mat matTmp = mat(rows_matDataZScore*cols_matDataZScore, 1);
		for (int i = 0; i < rows_matDataZScore*cols_matDataZScore; i++) // 把matData中数据一列列读取
		{
			matTmp(i, 0) = matDataZScore[i];
		}
		matTmp.reshape(rows_matDataZScore, cols_matDataZScore);

		ms.id = id[0];
		ms.score = score[0];
		ms.matDataZScore = matTmp;

		// save ms in vector
		vec.push_back(ms);
	}

	return vec;
}

