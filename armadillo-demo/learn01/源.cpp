#include <armadillo>

using namespace std;
using namespace arma;

// indexing
void learn01()
{
	mat A(5, 3, fill::randn);
	A.print("A: ");
	cout << A.n_cols << endl; 
	cout << A.n_rows << endl;
	cout << A.n_elem << endl;
	cout << A.col(0) << endl;
	cout << A.row(1) << endl;
	cout << A.cols(0, 2) << endl;
	cout << A.rows(1, 3) << endl;
	cout << A(span(0, 1), span(0, 2)) << endl;
}

// zeros, ones
void learn02()
{
	mat A(4, 2);
	A.zeros();
	A.print("A:");

	A.ones();
	A.print("A:");

	A = zeros<mat>(6, 6);
	A.print();

	A = ones<mat>(3, 4);
	A.print();

	cx_mat C = cx_mat(A, A);
	C.print("C: ");
}

// 元素乘，元素除以，除以
void learn03()
{
	mat A(5, 3, fill::ones);
	mat B(5, 3, fill::randn);
	A.print("A");
	B.print("B");

	mat C = A % B; // .*
	C.print("C");

	mat D = A / B; // ./
	D.print("D");

	mat E(5, 1, fill::randn);
	mat res = solve(A, E); // A\E
	res.print("res");
}

void learn04_vectorise_join()
{
	mat A;
	A << 1 << 2 << endr
		<< 3 << 4 << endr;
	A.print("A");

	mat X = vectorise(A);
	X.print("X");

	mat B(2, 4, fill::ones);
	X = join_horiz(A, B); // 横向，join
	X.print();

	mat C(5, 2, fill::eye);
	X = join_vert(A, C);
	X.print();

}

void learn05_save_load()
{
	mat A(4, 2, fill::randn);
	A.print("A");
	A.save("A.dat", raw_ascii);

	mat B;
	B.load("A.dat", raw_ascii);
	B.print("B");

}

void learn06_field()
{
	mat A = randn(2, 3);
	A.print("A");
	mat B = randn(4, 5);
	B.print("B");

	field<mat> F(2, 1);
	F(0, 0) = A;
	F(1, 0) = B;
	F.print("F");
}

void learn07_diag_fill_replace()
{
	mat A(5, 6, fill::randu);
	A.print("A");
	A.diag().print("A.diag()");
	A.diag().fill(datum::nan); // A主对角性元素 被填充为nan
	A.print("A fill datum::nan");

	A.replace(datum::nan, 0); // replace each nan with 0
	A.print("A");
}

// transform each elem using a fn
void learn08()
{
	mat A = ones<mat>(4, 5);
	A.print();
	// transform each element using a fn or lambda fn
	A.transform([](double val) {
		return (val + 123.0);
	});
	A.print();
}

// for each elem, passing its reference to a fn
void learn09()
{
	mat A = ones<mat>(4, 5);
	A.for_each([](mat::elem_type& val) {
		val += 100.0;
	});
	A.print();

	
}

// submat, find
void learn10()
{
	mat A = randu<mat>(5, 10);
	A.print();
	mat B =A.submat(0, 1, 2, 3);
	B.print();

	vec q = A.elem(find(A > 0.5));
	q.print("q");

	A.elem(find(A > 0.5)) += 1.0;
	A.print("A");

	uvec indices;
	indices << 2 << 3 << 6 << 8;
	A.elem(indices) = ones<vec>(4);
	A.print("A");

}

// index_max
void learn11()
{
	mat A(3, 4, fill::randn);
	A.print("A");
	auto m0 = index_max(A, 0); // 返回每一列的max index
	m0.print("index_max0");
	auto val0 = max(A, 0);
	val0.print("maxA0");

	auto m1 = index_max(A, 1); // 每一行的max	index
	m1.print("index_max1");

	vec v = randu<vec>(10);
	v.print("v");
	auto i = index_max(v);
	cout << "i " << i << endl;
}

// cumsum(A, dim), accu(A)
void learn12()
{
	mat A(4, 5, fill::randu);
	A.print("A");
	cout << A.is_sorted() << endl;
	mat B = cumsum(A, 0);
	B.print("B");

	double s = accu(A); // 所有元素的和
	cout << s << endl;
}

// norm
void learn13()
{
	vec q = randu<vec>(5);
	q.print("q");

	vec q2 = q % q;
	q2.print("q2");

	double sum_q2 = accu(q2);
	cout<< "sqrt( sum_q2 ): "  << sqrt( sum_q2 )<< endl;

	vec q2sqrt = sqrt(q2); // elem. wise
	q2sqrt.print("sqrt(q2)");

	double x = norm(q, 2);
	cout << "norm2: " << x << endl;

	mat A(4, 3, fill::randu);
	cout << "norm(A,2)" << norm(A, 2) << endl;
	cout << sqrt(A) << endl;
}

// sort, sort_index
void learn14()
{
	mat A(5, 4, fill::randu);
	A.print("A");
	mat B = sort(A, "ascend", 1);
	B.print("B");
	
	auto C = sort_index(A);
	cout << C << endl;
}



int main()
{
	learn15();


	system("pause");
	return 0;
}