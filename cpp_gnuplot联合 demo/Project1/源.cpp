#include <armadillo>
#include <iostream>
#include <string>

using namespace arma;
using namespace std;

void myplot(string filename)
{
	const char* gnuplotPath = "gnuplot.exe";
	FILE* gp = _popen(gnuplotPath, "w");
	if (gp == NULL)
	{
		cout << "can not open gnuplot\n";
		return;
	}
	
	fprintf(gp, "set title 'a figure'\n");
	fprintf(gp, "set term wxt enhanced\n");
	fprintf(gp, "set grid\n");

	/*string s1 = "plot '";
	string s2 = "' lw 5 lc rgb 'red'\n";
	string s_all = s1 + filename+s2;*/
	string s_all = "plot '" + filename + "' lw 5 lc rgb 'red'\n";
	fprintf(gp, s_all.data()); // .data 从string转到char

	fprintf(gp, "pause mouse\n");
	_pclose(gp);
	return;
}

int main()
{
	// 写入一个 txt文件，写入数据，再用gnuplot可视
	mat m1(3, 2, fill::randu);
	//m1.print("m1");
	
	m1.save("m11.txt", file_type::raw_ascii); // 此种方法生成的txt，可被 gnuplot
	// m1.save("m12.txt", file_type::csv_ascii); // 此种方法生成的，plot有问题

	myplot("m11.txt");



	system("pause");
	return 0;
}