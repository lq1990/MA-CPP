#include "IOMatlab.h"



IOMatlab::IOMatlab()
{
}


IOMatlab::~IOMatlab()
{
}


void IOMatlab::read(const char* fileName,
	device_vector<float>& sces_id_score, // [id0,score0, id1,score1, id2,score2, ...]
	device_vector<float>& sces_data,     // all data
	device_vector<int>& sces_data_mn,  // [m0,n0, m1,n1, m2,n2, ...]
	device_vector<int>& sces_data_idx_begin) // [idx0, idx1, idx2, ...]
{
	//thrust::device_vector<float> vec;

	const char* dir = "C:/Program Files/MATLAB/MATLAB Production Server/R2015a/MA_Matlab/Arteon/start_loadSync/DataFinalSave/list_data/";
	const char* tail = ".mat";
	char path[1000];
	strcpy_s(path, dir);
	strcat_s(path, fileName);
	strcat_s(path, tail);

	MATFile* file = matOpen(path, "r");
	mxArray* listStructTrain = matGetVariable(file, fileName);

	int numSces = mxGetNumberOfElements(listStructTrain); // 22/7/7 scenarios
	int numFields = mxGetNumberOfFields(listStructTrain); // 7 fields: {id, score, details, matData, matDataPcAll, matDataZScore, matDataZScore_t}
	//cout << numElems << ", " << numFields << endl;

	// traverse rows, i.e. scenarios
	for (int i = 0; i < numSces; i++)
	{
		// 第 i 个 sce

		//SceStruct ms;
		mxArray* item_id = mxGetField(listStructTrain, i, "id");
		mxArray* item_score = mxGetField(listStructTrain, i, "score");
		mxArray* item_matDataZScore = mxGetField(listStructTrain, i, "matDataZScore");
		mxArray* item_matDataZScore_t = mxGetField(listStructTrain, i, "matDataZScore_t");

		double* id = (double*)mxGetData(item_id);
		double* score = (double*)mxGetData(item_score);

	  /*double* matDataZScore = (double*)mxGetData(item_matDataZScore);
		int rows_matDataZScore = (int)mxGetM(item_matDataZScore);
		int cols_matDataZScore = (int)mxGetN(item_matDataZScore);*/

		double* matDataZScore_t = (double*)mxGetData(item_matDataZScore_t); // trans of matDataZScore
		int rows_matDataZScore_t = (int)mxGetM(item_matDataZScore_t);
		int cols_matDataZScore_t = (int)mxGetN(item_matDataZScore_t);

		//mat matTmp = mat(rows_matDataZScore*cols_matDataZScore, 1);
		//float *arr; // stores all the elems
		//arr = (float*)malloc(rows_matDataZScore * cols_matDataZScore * sizeof(float));
		
		for (int i = 0; i < rows_matDataZScore_t*cols_matDataZScore_t; i++) 
			// 把matData中数据一列列读取
		{
			//matTmp(i, 0) = matDataZScore[i];
			//arr[i] = matDataZScore[i];
			sces_data.push_back(matDataZScore_t[i]);
		}

		//matTmp.reshape(rows_matDataZScore, cols_matDataZScore);

		/*ms.id = id[0];
		ms.score = score[0];
		ms.matDataZScore = marr;*/
		//ms.matDataZScore = matTmp;

		sces_id_score.push_back(id[0]);
		sces_id_score.push_back(score[0]);
		sces_data_mn.push_back(rows_matDataZScore_t);
		sces_data_mn.push_back(cols_matDataZScore_t);

		// save ms in vector
		//vec.push_back(ms);
	}

	// use sces_data_mn to create sces_data_idx_begin
	sces_data_idx_begin.push_back(0);
	int idx_cumsum = 0;
	for (int i = 0; i < numSces-1; i++)
	{
		idx_cumsum += sces_data_mn[i * 2] * sces_data_mn[i * 2 + 1];
		sces_data_idx_begin.push_back(idx_cumsum);
	}

	return;
}

