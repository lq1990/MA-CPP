#pragma once
#ifndef   MY_H_MYLIB       //如果没有定义这个宏  
#define   MY_H_MYLIB       //定义这个宏  


#include <armadillo>
#include <vector>

using namespace std;
using namespace arma;


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
		向量转成 多行一列的arma::mat格式
	*/
	static mat vector2mat(vector<int> vec)
	{
		mat m1(vec.size(), 1);
		for (int i = 0; i < vec.size(); i++)
		{
			m1(i, 0) = vec[i];
		}
		return m1;
	}

	static mat vector2mat(vector<double> vec)
	{
		mat m1(vec.size(), 1);
		for (int i = 0; i < vec.size(); i++)
		{
			m1(i, 0) = vec[i];
		}
		return m1;
	}

	static double mean_vector(vector<int> vec)
	{
		double sum = 0;
		for (int i = 0; i < vec.size(); i++)
		{
			sum += vec[i];
		}
		double mean_vec = sum / vec.size();
		return mean_vec;
	}

	static double mean_vector(vector<double> vec)
	{
		double sum = 0;
		for (int i = 0; i < vec.size(); i++)
		{
			sum += vec[i];
		}
		double mean_vec = sum / vec.size();
		return mean_vec;
	}
};



#endif 