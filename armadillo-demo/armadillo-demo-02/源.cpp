#include <armadillo>

// 实验结果：armadillo 内部确实用 多线程并行计算加速了
using namespace std;
using namespace arma;

const int num = 100000;

// 使用 Armadillo，对比计算是否会加速

void calc_normal()
{
	double temp = 0;
	for (double i = 0; i < num; i++)
	{
		temp += i * i;
	}
	cout << "calc_normal, res: " << temp << endl;
}

void calc_arma(mat A)
{
	mat res = A * A.t();
	cout << "calc_arma, res: " << res(0,0) << endl;
}

int main()
{
	int nrows = 8; // 1: 0.0026, 2: 0.002867, 4: 0.0071, 8: 0.017
	mat A = Mat<double>(nrows, num);
	// 初始化A
	for (double i = 0; i < num; i++)
	{
		for (int j = 0; j < nrows; j++)
		{
			A(j, i) = i;
		}
	}
	// arma 计算
	auto start = std::chrono::system_clock::now();
	calc_arma(A);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	cout << "time needed: " << diff.count() << endl;

	cout << "----------------\n";
	// 普通计算
	start = std::chrono::system_clock::now();
	calc_normal();
	end = std::chrono::system_clock::now();
	diff = end - start;
	cout << "time needed: (*nrows*nrows) " << diff.count() *nrows*nrows << endl;
	

	system("pause");
	return 0;
}