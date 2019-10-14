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
	对LSTM隐层的封装。
	目的：当model需要多个隐层时，可简单的new出来

	抽出来的参数和系数：只针对隐层


	注意：多个隐层时，比如H1 H2, 
		前传时 X输入到H1， H1的结果hs输入到H2.			X   --> H1 --> H2
		反传时 loss传递到H2 计算出 dxnexts, 再传给H1.    loss --> H2 --> H1 
*/
class HiddenLayer
{
public:
	/*
		n_input_dim: 对于第一个hidden而言，n_input_dim=n_features
		n_hidden_cur: 当前hidden的neurons数目
		n_hidden_next: 当前hidden的下一个hidden的neurons数目
			特例：对于最后一个隐层而言，n_hidden_next=n_output_classes

		HiddenLayer实例化时：需要用Params中参数进行初始化自己的参数
	*/
	HiddenLayer(int n_input_dim, int n_hidden_cur, int n_hidden_next, Params* ps); // 构造实例同时 初始化实例的属性即参数
	~HiddenLayer();

	/*
		隐层前传：算出 hs，
			区别于 正统的前传是计算出pred值和loss.

		Xs: 第一个隐层的输入时Xs, 其它隐层输入是它前一个隐层的 Hs
	*/
	map<int, mat> hiddenForward(mat inputs, mat hprev, mat cprev, double dropout_prob);


	/*
		隐层反传：由dh算出 deltaParams 并返回这些值 dP.

		// softmax loss gradient
			dy = ps[t];
			dy[idx1] -= 1; 反向传过来的loss

			最后一个hidden的d_outputs_in[tend]=dy

		return keys of map: dWf, dWi, dWc, dWo, dWhh, dbf, dbi, dbc, dbo, dbhh
	*/
	map<string, mat> hiddenBackward(mat inputs, map<int, mat> d_outputs_in, map<int, mat>& d_outputs_out, double lambda0);



	/*
		拿到RNN中参数，对HiddenLayer参数进行set
	*/
	void setParams(Params* ps); 

	/*get W b of this layer*/
	map<string, mat> getParams(); 

	/*
		以map的key    为mat的行index
		key对应的value为mat的行中的数据

		key: [startIdx, endIdx]
	*/
	static mat map2mat(map<int, mat> mp, int startIdx, int endIdx);

	/*
		title: 1,2
	*/
	void saveParams(string title);

	void loadParams(string title);

	/*
		根据prob生成 dropout vector 只含 0/1. 长度为len
		每个元素prob的概率生成0，其它为1.

		prob: 是去除神经元的占比
	*/
	static mat generateDropoutVector(int len, double prob);

private:
	mat sigmoid(arma::mat mx);


public: 
	/*
	*/
	// 以下为实例属性，属性是某个隐层专有，这些参数也是待训练
	mat Wf, Wi, Wc, Wo; // weight of forget gate, input gate, candidate cell, output gate
	mat Whh; // weight between hidden-hidden, 特例最后一个隐层的Whh=Why，反传中用到Whh
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

