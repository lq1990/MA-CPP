#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main()
{
	mat A(2, 3); // 没有初始值

	cout << "A.nrows: " << A.n_rows << endl;
	cout << "A.ncols: " << A.n_cols << endl;

	A(1, 2) = 456.0; // indexing 从 0 开始
	A.print("A: ");

	A = 5.0; // scalar 被当做 1x1 维度的 矩阵
	A.print("A: ");

	A.set_size(4, 5); // 设置维度
	A.fill(5.0); // 填充一个具体的值
	A.print("A: ");

	system("pause");
	return 0;
}