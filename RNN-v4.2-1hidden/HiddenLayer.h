#pragma once
#include <iostream>
#include <ctime>
#include <cstdlib>
#include "MyLib.h"
#include <armadillo>
#include <string>
#include <map>
#include <set>
#include <vector>
#include "Params.h"

using namespace std;
using namespace arma;
/*
	��LSTM����ķ�װ��
	Ŀ�ģ���model��Ҫ�������ʱ���ɼ򵥵�new����

	������Ĳ�����ϵ����ֻ�������


	ע�⣺�������ʱ������H1 H2, 
		ǰ��ʱ X���뵽H1�� H1�Ľ��hs���뵽H2.			X   --> H1 --> H2
		����ʱ loss���ݵ�H2 ����� dxnexts, �ٴ���H1.    loss --> H2 --> H1 
*/
class HiddenLayer
{
public:
	/*
		n_input_dim: ���ڵ�һ��hidden���ԣ�n_input_dim=n_features
		n_hidden_cur: ��ǰhidden��neurons��Ŀ
		n_hidden_next: ��ǰhidden����һ��hidden��neurons��Ŀ
			�������������һ��������ԣ�n_hidden_next=n_output_classes

		HiddenLayerʵ����ʱ����Ҫ��Params�в������г�ʼ���Լ��Ĳ���
	*/
	HiddenLayer(int n_input_dim, int n_hidden_cur, int n_hidden_next, Params* ps); // ����ʵ��ͬʱ ��ʼ��ʵ�������Լ�����
	~HiddenLayer();

	/*
		����ǰ������� hs��
			������ ��ͳ��ǰ���Ǽ����predֵ��loss.

		Xs: ��һ�����������ʱXs, ����������������ǰһ������� Hs
	*/
	map<int, mat> hiddenForward(mat inputs, mat hprev, mat cprev, double dropout_prob);


	/*
		���㷴������dh��� deltaParams ��������Щֵ dP.

		// softmax loss gradient
			dy = ps[t];
			dy[idx1] -= 1; ���򴫹�����loss

			���һ��hidden��d_outputs_in[tend]=dy

		return keys of map: dWf, dWi, dWc, dWo, dWhh, dbf, dbi, dbc, dbo, dbhh
	*/
	map<string, mat> hiddenBackward(mat inputs, map<int, mat> d_outputs_in, map<int, mat>& d_outputs_out, double lambda0);



	/*
		�õ�RNN�в�������HiddenLayer��������set
	*/
	void setParams(Params* ps); 

	/*get W b of this layer*/
	map<string, mat> getParams(); 

	/*
		��map��key    Ϊmat����index
		key��Ӧ��valueΪmat�����е�����

		key: [startIdx, endIdx]
	*/
	static mat map2mat(map<int, mat> mp, int startIdx, int endIdx);

	/*
		title: 1,2
	*/
	void saveParams(string title);

	void loadParams(string title);

	/*
		����prob���� dropout vector ֻ�� 0/1. ����Ϊlen
		ÿ��Ԫ��prob�ĸ�������0������Ϊ1.

		prob: ��ȥ����Ԫ��ռ��
	*/
	static mat generateDropoutVector(int len, double prob);

private:
	mat sigmoid(arma::mat mx);


public: 
	/*
	*/
	// ����Ϊʵ�����ԣ�������ĳ������ר�У���Щ����Ҳ�Ǵ�ѵ��
	mat Wf, Wi, Wc, Wo; // weight of forget gate, input gate, candidate cell, output gate
	mat Whh; // weight between hidden-hidden, �������һ�������Whh=Why���������õ�Whh
	mat bf, bi, bc, bo, bhh; // bias
	
	map<int, mat> cs, hs;


private:
	map<int, mat> xs, X,
		h_fs, h_is, h_os, h_cs; // hidden gate state

private:
	int n_input_dim;
	int n_hidden_cur;
	int n_hidden_next;

};

