#include <iostream>
#include <string>
#include <sstream>
#include <Windows.h>

using namespace std;

void myplot()
{
	const char* gnuplotPath = "gnuplot.exe";
	FILE* gp = _popen(gnuplotPath, "w");
	if (gp == NULL)
	{
		cout << "can not open gnuplot\n";
		return;
	}

	fprintf(gp, "set title 'mytitle'\n");
	fprintf(gp, "set xlabel 'x'\n");
	fprintf(gp, "plot x**2\n");
	fprintf(gp, "plot [x=0:2] [0:20] exp(x**2)\n");
	fprintf(gp, "pause mouse\n"); // 用户点击后退出
	_pclose(gp);
	return;
}

void myplot2()
{
	const char* gppath = "gnuplot.exe";
	FILE* gp = _popen(gppath, "w");
	if (gp == NULL)
	{
		cout << "not found\n";
		return;
	}

	fprintf(gp, "plot 'data.txt'\n");
	fprintf(gp, "pause mouse\n");
	_pclose(gp);
	return;
}

void mystring()
{
	string s1 = "abc";
	string s2 = "cde";
	string s3 = s1 + s2; // 用 + 既可以实现字符串拼接
	cout << "s1: " << s1 << endl;
	cout << "s2: " << s2 << endl;
	cout << "s3: " << s3 << endl;

	// 字符串和数字 拼接
	int i1 = 11;
	stringstream si1;
	si1 << s1 << i1 << s2;
	cout << "s1 i1 s2: " << si1.str() << endl;
}

// anim
void myplot3()
{
	const char* gppath = "gnuplot.exe";
	FILE* gp = _popen(gppath, "w");
	if (gp == NULL)
	{
		cout << "not found\n";
		return;
	}

	fprintf(gp, "set term wxt enhanced\n");

	for (int i = 0; i < 10; i++)
	{
		string temp_s = "";
		stringstream ss;
		
		//"plot i*sin(x)\n";
		string s1 = "plot ";
		string s2 = "*sin(x)\n";
		ss << s1 << i << s2;
		temp_s = ss.str();
		cout << temp_s << endl;
		fprintf(gp, temp_s.data());
		fprintf(gp, "set yrange [-11:11]\n");
	}
	fprintf(gp, "pause mouse\n");

	_pclose(gp);
	return;
}

int main()
{
	myplot3();

	
	

	//system("pause");
	return 0;
}