#pragma once
#ifndef   MY_H_MYLIB       //如果没有定义这个宏  
#define   MY_H_MYLIB       //定义这个宏  

#include <armadillo>
#include <vector>

using namespace std;
using namespace arma;

template<typename T>

/*
	自定义我的库
	实现一些common的功能
*/
class MyLib
{
public:
	MyLib() {};
	~MyLib() {};

	/*
		sigma'(x) = sigma(x) % (1 - sigma(x) )
	*/
	static mat dsigmoid(arma::mat mx);

	static mat dtanh(arma::mat mx);

	static mat softmax(arma::mat mx)
	{
		// softmax = exp(x) / sum(exp(x))
		mat sum_exp = arma::sum(arma::exp(mx), 0); // sum(,0) 列方向sum
		return arma::exp(mx) / (sum_exp(0, 0) + pow(10, -8) );
	}

	static mat sigmoid(arma::mat mx)
	{
		// sigmoid = 1 / (1+exp(-x))
		return 1 / (1 + arma::exp(-mx));
		// arma 
		/*
			+ - % / elemwise
			* mul
		*/
	}

	static void printVector(vector<T> vec)
	{
		for (int i = 0; i < vec.size(); i++)
		{
			cout << vec[i] << endl;
		}
	}

	/*
		向量转成 多行一列的arma::mat格式
	*/
	static mat vector2mat(vector<T> vec)
	{
		mat m1(vec.size(), 1);
		for (int i = 0; i < vec.size(); i++)
		{
			m1(i, 0) = vec[i];
		}
		return m1;
	}

	/*
		get mean of vector
	*/
	static double mean_vector(vector<T> vec)
	{
		double sum = 0;
		for (int i = 0; i < vec.size(); i++)
		{
			sum += vec[i];
		}
		double mean_vec = sum / vec.size();
		return mean_vec;
	}


	/*
		对dPVec中存储的所有mat，进行elementwise 的平方、求和、sqrt。
		L2范式。
	*/
	static mat norm_elementwise(vector<arma::mat> dPVec)
	{
		mat mat2_sum = 0;
		for (int i = 0; i < dPVec.size(); i++)
		{
			mat amat = dPVec[i];
			mat amat2 = amat % amat; // square

			mat2_sum += amat2; // sum
		}

		mat mat2_sum_sqrt = arma::sqrt(mat2_sum);

		return mat2_sum_sqrt;
	}
};



#endif 

template<typename T>
inline mat MyLib<T>::dsigmoid(arma::mat mx)
{
	return MyLib<mat>::sigmoid(mx) % (1 - MyLib<mat>::sigmoid(mx));
}

template<typename T>
inline mat MyLib<T>::dtanh(arma::mat mx)
{
	return 1 - arma::tanh(mx) % arma::tanh(mx);
}


