#include "DemoMultiThread.h"

std::mutex DemoMultiThread::mtx;

DemoMultiThread::DemoMultiThread()
{
}


DemoMultiThread::~DemoMultiThread()
{
}


void DemoMultiThread::getSum(double left, double right, double& res)
{
	// 使用临时变量，而非用直接res+=i。因为 res为引用，引用再取值的速度，不如设定局部变量栈操作。
	double tmp_sum = 0;

	for (double i = left; i < right; i++)
	{
		tmp_sum += i;
	}
	res = tmp_sum;
}

double DemoMultiThread::getSum2(double left, double right)
{
	double tmp = 0;
	for (double i = left; i < right; i++)
	{
		tmp += i;
	}

	return tmp;
}

void DemoMultiThread::demo_multi_thread0(int max_n_threads)
{
	double max_num = pow(10, 9);
	clock_t t_begin, t_end;

	// 普通一个线程
	/*t_begin = clock();
	double sum_one_thread = 0;
	getSum(0, max_num, std::ref(sum_one_thread));
	cout << "sum_one_thread: " << sum_one_thread << endl;
	t_end = clock();
	cout << "time needed: " << (double)(t_end - t_begin) / CLOCKS_PER_SEC << endl;*/

	// 多线程
	mat dura_log_multi_thread = mat(max_n_threads, 3); // arma::mat 存储 n_threads 和 时间

	for (int t = 1; t <= max_n_threads; t++)
	{
		int n_threads = t;
		t_begin = clock();
		vector<future<double>> vec_future;
		vec_future.reserve(n_threads); // vector预留空间
		for (int i = 0; i < n_threads; i++)
		{
			vec_future.push_back(async(getSum2, i == 0 ? 0 :
				max_num / n_threads * i
				, max_num / n_threads * (i + 1)));
		}

		double sum_multi_thread = 0;
		for (int i = 0; i < n_threads; i++)
		{
			sum_multi_thread += vec_future[i].get(); // get 有阻塞
		}
		//cout << "sum_multi_thread: " << sum_multi_thread << endl;
		t_end = clock();
		double dura = (double)(t_end - t_begin) / CLOCKS_PER_SEC;
		//cout << "time needed: " << dura << endl;

		dura_log_multi_thread(t - 1, 0) = t;
		dura_log_multi_thread(t - 1, 1) = dura;
		dura_log_multi_thread(t - 1, 2) = sum_multi_thread;

	}



	dura_log_multi_thread.print("n_thread, dura");
	dura_log_multi_thread.save("dura_log.txt", raw_ascii);


}

void DemoMultiThread::mySum(int& n)
{
	mtx.lock(); // lock放到 for外面，会快很多。比lock放到for内部快。
	for (int i = 0; i < 100000; i++)
	{
		n++;
	}
	mtx.unlock();
}

void DemoMultiThread::demo_multi_thread_mutex()
{
	vector<thread> vec;
	int s = 0;
	for (int i = 0; i < 4; i++)
	{
		//vec.push_back(thread(mySum, std::ref(s))); 
		vec.emplace_back(mySum, std::ref(s));
		// emplace_back 和push_back 相比较，好处：省去主动调用thread()
	}
	for (int i = 0; i < 4; i++)
	{
		vec[i].join();
	}
	cout << s << endl;
}

/*
	练习多线程的 并行计算
*/
void DemoMultiThread::demo_multi_thread()
{
	clock_t t_begin, t_end;
	int max_num = pow(10, 9);
	double sum = 0;

	/*
	t_begin = clock();
	for (int i = 0; i < max_num; i++)
	{
		sum += i;
	}
	t_end = clock();
	cout << "1 thread, sum: " << sum <<
		", time needed: " << (double)(t_end-t_begin)/CLOCKS_PER_SEC << endl;
*/

	// multi thread
	int total_n_threads = 10;
	for (int n_threads = 1; n_threads <= total_n_threads; n_threads++)
	{
		t_begin = clock();
		// 使用vector存储 future
		vector<future<double>> vec_future;
		vec_future.reserve(n_threads); // 预备空间
		sum = 0;
		for (int i = 0; i < n_threads; i++)
		{
			vec_future.push_back(std::async(sumFn1, i==0 ? 0 : max_num/n_threads * i, max_num /n_threads *(i+1)));
		}

		for (int i = 0; i < n_threads; i++)
		{
			sum += vec_future[i].get();
		}

		/*
		double res1;
		double res2;
		std::thread myt1(sumFn, 0, max_num / 2, std::ref(res1));
		std::thread myt2(sumFn, max_num / 2,max_num, std::ref(res2));
		myt1.join();
		myt2.join();
		sum = res1 + res2;
		*/
		t_end = clock();
		double dura = (double)(t_end - t_begin) / CLOCKS_PER_SEC;
		cout << n_threads <<" thread, sum: " << sum <<
			", time needed: " << dura << endl;
	}




}

void DemoMultiThread::sumFn(int left, int right, double& res) 
{
	double tmp = 0;
	for (int i = left; i < right; i++)
	{
		tmp += i;
	}
	res = tmp;
}

double DemoMultiThread::sumFn1(int left, int right)
{
	double tmp = 0;
	for (int i = left; i < right; i++)
	{
		tmp += i;
	}

	return tmp;
}

