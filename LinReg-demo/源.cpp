#include <armadillo>
#include <string>
#include <time.h>

using namespace std;
using namespace arma;

void myplot(string filename)
{
	const char* gnuplotPath = "gnuplot.exe";
	FILE* gp = _popen(gnuplotPath, "w");
	if (gp == NULL)
	{
		cout << "can't open gnuplot.exe" << endl;
		return;
	}

	fprintf(gp, "set term wxt enhanced\n");
	fprintf(gp, "set grid\n");
	fprintf(gp, "set yrange [0:]\n");
	string str = "plot '" + filename + "' lw 3\n";
	fprintf(gp, str.data());

	fprintf(gp, "pause mouse\n");
	_pclose(gp);
	return;
}

double printLoss(mat X, mat W, mat Y)
{
	mat tmp = X * W - Y;
	mat tmp2 = tmp % tmp * 0.5;
	mat m = mean(tmp2, 0);
	//m.print("loss");
	return m(0, 0);
}

int main()
{
	clock_t t_begin, t_end;
	t_begin = clock();
	// ====================================================
	// 线性回归
	int nx_extend = 3; // 对X扩展到 n 阶
	double alpha = 0.01;
	int totalSteps = 1000;

	// 1. 生成所用数据，并写入到txt 供给可视化用
	int nrows = 100;
	mat m1(nrows, 2);
	//m1.print("m1");

	for (int i = 0; i < nrows; ++i)
	{
		m1(i, 0) = i;

		double randomVal = rand() % 100 / 50.0;
		//cout << "randomVal: " << randomVal << endl;
		m1(i, 1) = randomVal + 2 + i;
	}

	m1.print("original m1");
	m1.save("m1.txt", file_type::raw_ascii); 

	// 2. 对原始数据预处理，扩展X，提取Y
	mat X(nrows, nx_extend);
	mat Y = m1.col(1);
	for (int i = 0; i < nrows; ++i)
	{
		for (int j = 0; j < nx_extend; ++j)
		{
			X(i, j) = pow(m1(i, 0), j);
		}
	}


	// 3. X归一化
	mat X_max = max(X, 0);
	mat X_max_new = repelem(X_max, nrows, 1);
	mat X_min = min(X, 0);
	mat X_min_new = repelem(X_min, nrows, 1);

	mat X_scaling = (X - X_min_new) / (X_max_new - X_min_new + pow(10, -8));
	// 第一列赋值为1
	X_scaling.col(0) = ones(nrows, 1);
	X_scaling.print("X_scaling");

	// 4. 输入到模型，训练
	// 初始化 W
	mat W(nx_extend, 1, fill::randn);
	W.print("init W");


	mat lossMat(totalSteps, 1);
	for (int i = 0; i < totalSteps; i++)
	{
		double loss = printLoss(X_scaling, W, Y);
		lossMat(i, 0) = loss;
		if (i % 10 == 0) 
		{
			cout << "loss: " << loss << endl;
		}
		mat dJdW = X_scaling.t() * (X_scaling*W - Y);
		W -= alpha * dJdW;
	}

	W.print("final W");
	mat H = X_scaling * W;
	Y.print("Y");
	H.print("from model");

	H.save("H.txt", raw_ascii);
	Y.save("Y.txt", raw_ascii);
	lossMat.save("lossMat.txt", file_type::raw_ascii);

	// 5. 泛化
	mat XPred(101, 1);
	for (int i = 0; i < 101; i++)
	{
		XPred(i, 0) = i / 100.0;
	}
	// 扩展，归一，再输入模型泛化
	// 此处用 类来封装 =======================================


	// =======================================================
	t_end = clock();
	cout << "time needed: [s] " << (double)(t_end - t_begin) / CLOCKS_PER_SEC << endl;
	// gnuplot
	//myplot("m1.txt");

	system("pause");
	return 0;
}