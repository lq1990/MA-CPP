#pragma once
#include <iostream>
#include "MyLib.h"
#include <armadillo>
#include <string>
#include <map>
#include <vector>

using namespace std;
using namespace arma;
/*
	��LSTM����ķ�װ��
	Ŀ�ģ���model��Ҫ�������ʱ���ɼ򵥵�new����

	������Ĳ�����ϵ����ֻ�������
*/
class HiddenLayer
{
public:
	/*
		n_features:
		n_hidden_cur: ��ǰhidden��neurons��Ŀ
		n_hidden_next: ����ڵ�ǰhidden����һ��hidden��neurons��Ŀ
			�������������һ��������ԣ�n_hidden_next=n_output_classes
	*/
	HiddenLayer(int n_features, int n_hidden_cur, int n_hidden_next); // ����ʵ��ͬʱ ��ʼ��ʵ�������Լ�����
	~HiddenLayer();

	/*
		����ǰ������� hs��
			������ ��ͳ��ǰ���Ǽ����predֵ��loss.

		Xs: ��һ�����������ʱXs, ����������������ǰһ������� Hs
	*/
	map<int, mat> hiddenForward(mat inputs);


	/*
		���㷴������dh��� deltaParams ��������Щֵ dP.

		// softmax loss gradient
			dy = ps[t];
			dy[idx1] -= 1; ���򴫹�����loss

			���һ��hidden��d_output=dy
	*/
	map<string, mat> hiddenBackward(mat inputs, mat d_output, double lambda0); 

	/*
		������������и��£�ʹ�� deltaParams
	*/
	void updateParams(map<string, mat> deltaParams); 

	/*get W b of this layer*/
	map<string, mat> getParams(); 


private:
	mat sigmoid(arma::mat mx);

private: 
	// ����Ϊʵ�����ԣ�������ĳ������ר�У���Щ����Ҳ�Ǵ�ѵ���ģ�Whh������֮���W���������һ���������Whh����Wy
	mat Wf, Wi, Wc, Wo, Whh; // weight of forget gate, input gate, candidate cell, output gate, output
	mat bf, bi, bc, bo, bhh; // bias
	

	map<int, mat> xs, X,
		h_fs, h_is, h_os, h_cs, // hidden gate state
		cs, hs;


	int n_features;
	int n_hidden_cur;
	int n_hidden_next;

};

