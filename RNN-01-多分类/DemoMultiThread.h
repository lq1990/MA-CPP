#pragma once

#include <iostream>
#include <mutex>
#include <thread>
#include <future>
#include <armadillo>

using namespace std;
using namespace arma;

class DemoMultiThread
{
public:
	DemoMultiThread();
	~DemoMultiThread();
	static void getSum(double left, double right, double & res);
	static double getSum2(double left, double right);
	static void demo_multi_thread0(int max_n_threads);
	static void mySum(int & n);
	static void demo_multi_thread_mutex();
	static void sumFn(int left, int right, double & res);

	static double sumFn1(int left, int right);

	static void demo_multi_thread();
private:
	static std::mutex mtx;
};

