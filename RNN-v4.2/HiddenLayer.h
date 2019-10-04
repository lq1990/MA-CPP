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
	对LSTM隐层的封装。
	目的：当model需要多个隐层时，可简单的new出来

	抽出来的参数和系数：只针对隐层
*/
class HiddenLayer
{
public:
	/*
		n_features:
		n_hidden_cur: 当前hidden的neurons数目
		n_hidden_next: 相对于当前hidden的下一个hidden的neurons数目
			特例：对于最后一个隐层而言，n_hidden_next=n_output_classes
	*/
	HiddenLayer(int n_features, int n_hidden_cur, int n_hidden_next); // 构造实例同时 初始化实例的属性即参数
	~HiddenLayer();

	/*
		隐层前传：算出 hs，
			区别于 正统的前传是计算出pred值和loss.

		Xs: 第一个隐层的输入时Xs, 其它隐层输入是它前一个隐层的 Hs
	*/
	map<int, mat> hiddenForward(mat inputs);


	/*
		隐层反传：由dh算出 deltaParams 并返回这些值 dP.

		// softmax loss gradient
			dy = ps[t];
			dy[idx1] -= 1; 反向传过来的loss

			最后一个hidden的d_output=dy
	*/
	map<string, mat> hiddenBackward(mat inputs, mat d_output, double lambda0); 

	/*
		对隐层参数进行更新，使用 deltaParams
	*/
	void updateParams(map<string, mat> deltaParams); 

	/*get W b of this layer*/
	map<string, mat> getParams(); 


private:
	mat sigmoid(arma::mat mx);

private: 
	// 以下为实例属性，属性是某个隐层专有，这些参数也是待训练的，Whh：隐层之间的W，对于最后一个隐层而言Whh就是Wy
	mat Wf, Wi, Wc, Wo, Whh; // weight of forget gate, input gate, candidate cell, output gate, output
	mat bf, bi, bc, bo, bhh; // bias
	

	map<int, mat> xs, X,
		h_fs, h_is, h_os, h_cs, // hidden gate state
		cs, hs;


	int n_features;
	int n_hidden_cur;
	int n_hidden_next;

};

