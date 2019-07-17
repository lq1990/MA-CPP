#include "Cuda4RNN.h"

int Cuda4RNN::n_features = MyParams::n_features; // num of columns os matData
int Cuda4RNN::n_hidden = MyParams::n_hidden;	// num of hidden neurons
int Cuda4RNN::n_output_classes = MyParams::n_output_classes;	// num of predicted classes of Scenarios
float Cuda4RNN::score_min = MyParams::score_min;
float Cuda4RNN::score_max = MyParams::score_max;

MyArray* Cuda4RNN::Wxh = MyParams::Wxh;
MyArray* Cuda4RNN::Whh = MyParams::Whh;
MyArray* Cuda4RNN::Why = MyParams::Why;
MyArray* Cuda4RNN::bh = MyParams::bh;
MyArray* Cuda4RNN::by = MyParams::by;



Cuda4RNN::Cuda4RNN()
{

}


Cuda4RNN::~Cuda4RNN()
{
}


/*
	arma::mat in RNN-v2 => MyArray*
*/
CudaMap Cuda4RNN::lossFun(MyArray* inputs, 
	float score,
	MyArray* hprev, 
	device_vector<float>& true_false, 
	device_vector<float>& log_target, 
	device_vector<float>& log_prediction)
{
	cublasStatus_t stat;
	cublasHandle_t handle;
	cublasCreate(&handle);

	MyArray* targets = score2onehot(score);

	// map stores values in FP, the values are used in BPTT
	CudaMap xs, hs, ys, ps;

	hs.put(-1, hprev); // set hprev
	float loss = 0.0f;

	// -------------- Forward Pass ------------------------
	for (int t = 0; t < inputs->n_rows_origin; t++)
	{
		xs.put(t, inputs->getRowToDevice(t));
		/*
		// -------- hs[t] = arma::tanh(Wxh * xs[t] + Whh*hs[t-1] + bh) ------------
		// 根据cuBLAS的要求，是否需要在GPU分配内存给变量 ?????
		// d_hst_raw01 = Wxh * xs[t]
		MyArray* d_hst_raw01 = new MyArray(hprev->size, hprev->n_rows_origin, hprev->arr); // temp var
		
		float alpha = 1.0f, beta = 0.0f;
		cublasSgemv(handle, // y = 1 * A * x + 0 * y
			CUBLAS_OP_N,
			Wxh->n_rows_origin, Wxh->size / Wxh->n_rows_origin,
			&alpha,
			Wxh->arr, Wxh->n_rows_origin,
			xs.get(t)->arr, 1,
			&beta,
			d_hst_raw01->arr, 1);

		// d_hst_raw02 = Whh * hs[t-1]
		MyArray* d_hst_raw02 = new MyArray(hprev->size, hprev->n_rows_origin, hprev->arr); // temp var

		alpha = 1.0f, beta = 0.0f;
		cublasSgemv(handle, // y = 1 * A * x + 0 * y
			CUBLAS_OP_N,
			Whh->n_rows_origin, Whh->size / Whh->n_rows_origin,
			&alpha,
			Whh->arr, Whh->n_rows_origin,
			hs.get(t-1)->arr, 1,
			&beta,
			d_hst_raw02->arr, 1);

		// d_hst_raw02 = d_hst_raw01 + d_hst_raw02
		alpha = 1.0f;
		cublasSaxpy(handle, // y = 1 * x + y
			hprev->size,
			&alpha,
			d_hst_raw01->arr, 1,
			d_hst_raw02->arr, 1);

		// d_hst_raw02 = bh + d_hst_raw02
		alpha = 1.0f;
		cublasSaxpy(handle, // y = 1 * x + y
			hprev->size,
			&alpha,
			bh->arr, 1,
			d_hst_raw02->arr, 1); // hst_raw saves in d_hst_raw02

		// hst = tanh(d_hst_raw02) 自己写kernel实现，或者用thrust，封装cublas类似于arma
		MyArray* hst = new MyArray(hprev->size, hprev->n_rows_origin, hprev->arr);
		//g_tanh << <1, d_hst_raw02->size >> > (hst->arr, d_hst_raw02->arr, hst->size);
		hs.put(t, hst);

		*/
	}
	

	/*float *arr1, *arr2;
	arr1 = (float*)malloc(6 * sizeof(float));
	for (int i = 0; i < 6; i++)
	{
		arr1[i] = (float)i;
	}
	arr2 = (float*)malloc(4* sizeof(float));
	for (int i = 0; i < 4; i++)
	{
		arr2[i] = (float)(rand() % 10);
	}


	MyArray* marr1 = new MyArray(6, 3, arr1);
	MyArray* marr2 = new MyArray(4, 2, arr2);

	CudaMap map;
	map.put(1, marr1);
	map.put(3, marr2);*/

	cublasDestroy(handle);
	CudaMap map;
	return xs;
}

MyArray * Cuda4RNN::score2onehot(float score)
{
	float part = 1.0f / this->n_output_classes;

	float pos = (score - score_min) /
		(score_max - score_min + powf(10, -8));

	int pos_idx = floorf(pos / part);

	// create array with 0 or 1
	float* arr;
	arr = (float*)malloc(this->n_output_classes * sizeof(float));
	for (int i = 0; i < this->n_output_classes; i++)
	{
		arr[i] = 0.0f;
	}
	arr[pos_idx] = 1.0f;

	// stores info of arr into marr
	MyArray* marr = new MyArray();
	marr->arr = arr;
	marr->n_rows_origin = this->n_output_classes;
	marr->size = this->n_output_classes;

	//free(arr); // 能否free。不能free，否则 marr->arr 的数据也被free了。
	return marr;
}

void Cuda4RNN::test_gpu_fns_CudaUtils()
{
	// ------------------------------------
	cublasHandle_t handle;
	cublasCreate(&handle);

	/*
	cout << "test gpu_fns" << endl;
	const int A_M = 4;
	const int A_N = 2;
	const int B_M = 2;
	const int B_N = 3;

	device_vector<float> d_A(A_M * A_N); // device_vector会自动释放资源
	device_vector<float> d_B(B_M * B_N);
	device_vector<float> d_C; // d_C 是动态可变的

	cout << "A.size: " << d_A.size()
		<< "\tB.size: " << d_B.size()
		<< "\tC.size: " << d_C.size() << endl;

	// init values
	for (int i = 0; i < A_M*A_N; i++)
	{
		d_A[i] = float(i);
	}
	for (int i = 0; i < B_M*B_N; i++)
	{
		d_B[i] = i/2.0f;
	}

	d_C = gpu_mmul(handle, d_A, d_B, A_M, A_N, B_N);
	cout << "after mmul, C.size: " << d_C.size() << endl;
	printToHost(d_A, A_M, A_N, "A");
	printToHost(d_B, B_M, B_N, "B");
	printToHost(d_C, A_M, B_N, "C = A*B");

	// G = D*E*F ，将gpu_fns串起来使用
	device_vector<float> D(4 * 2);
	device_vector<float> E(2 * 3);
	device_vector<float> F(3 * 5);
	device_vector<float> G;
	for (int i = 0; i < 8; i++)
	{
		D[i] = float(i);
	}
	for (int i = 0; i < 6; i++)
	{
		E[i] = float(i/2);
	}
	for (int i = 0; i < 15; i++)
	{
		F[i] = float(1);
	}

	G = gpu_mmul(handle,
			gpu_mmul(handle, D, E, 4, 2, 3), 
			F, 4, 3, 5);

	printToHost(D, 4, 2, "D");
	printToHost(E, 2, 3, "E");
	printToHost(F, 3, 5, "F");
	printToHost(G, 4, 5, "G = D*E*F");
	*/
	
	// -------------- 使用 封装的类 CudaUtils --------------
	// G = D*E*F ，将gpu_fns串起来使用
	/*int dm = 1, dn = 2;
	int em = 2, en = 3;
	int fm = 3, fn = 1;

	device_vector<float> D(dm * dn);
	device_vector<float> E(em * en);
	device_vector<float> F(fm * fn);
	device_vector<float> G;
	for (int i = 0; i < 2; i++)
	{
		D[i] = float(i);
	}
	for (int i = 0; i < 6; i++)
	{
		E[i] = float(1);
	}
	for (int i = 0; i < 3; i++)
	{
		F[i] = float(1);
	}
	printToHost(D, dm, dn, "D");
	printToHost(E, em, en, "E");
	printToHost(F, fm, fn, "F");*/

	// ----------- mul ----------------
	/*
	cout << "by gpu_fns: " << endl;
	G = gpu_mmul(handle, 
			gpu_mmul(handle, 
				D, E, 1, 2, 3), 
			F, 1, 3, 1); // G = D*E*F
	//G = gpu_mmul(handle, D, E, 4, 2, 3);
	printToHost(D, 1, 2, "D");
	printToHost(E, 2, 3, "E");
	printToHost(F, 3, 1, "F");
	printToHost(G, 1, 1, "G = D*E");

	cout << "by CudaUtils \n";
	CudaUtils* cu = new CudaUtils(handle); 
	// 思考下，有问题，不能只在cpu上分配内存，否则结果是错的 
	// 因为此类不会真正在gpu上执行，仅仅在cpu上调用gpu fn，所以没问题

	// 注意链式计算时，并不会区分 +*优先级，只会顺序往后
	cu->box(D, 1, 2)->mmul(E, 2, 3)->mmul(F, 3, 1);
	auto res = cu->getResDevVec();
	int resM = cu->getResM(); cout << "resM: " << resM << endl;
	int resN = cu->getResN(); cout << "resN: " << resN << endl;
	printToHost(res, resM, resN, "G, by CudaUtils");
	*/
	// ----------- add ------------------
	/*
	cout << "add by gpu_add" << endl;
	G = gpu_add(handle, E, E, em*en);
	printToHost(G, em, en, "E+E");

	cout << "add by CudaUtils\n";
	CudaUtils* cuE = new CudaUtils(handle, E, em, en); // 思考下handle的作用
	// 注意此处在cpu上new内存，因为传入gpu_fns的参数都是gpu上分配的，所以没问题
	auto res = cuE->add(E, em, en)->getResDevVec();
	//auto res = cu->getResDevVec();
	printToHost(res, em, en, "add by CudaUtils");
	*/
	// ------------ z = (A*x + B*y) 混合 ------------
	int am = 3, an = 2;
	int bm = 3, bn = 2;
	device_vector<float> A(am*an);
	device_vector<float> x(an);
	device_vector<float> B(bm*bn);
	device_vector<float> y(bn);
	device_vector<float> z;
	//gpu_fill_rand(raw_pointer_cast(&A[0]), am, an);
	A = gpu_fill_rand(A, am, an, 1.f, 3.f); // 必须return A，否则不改变
	thrust::fill(x.begin(), x.end(), 1.0f);
	B = gpu_fill_rand(B, bm, bn);
	thrust::fill(y.begin(), y.end(), 1.0f);

	printToHost(A, am, an, "A");
	printToHost(x, an, 1, "x");
	printToHost(B, bm, bn, "B");
	printToHost(y, bn, 1, "y");

	// use gpu_fns
	z = gpu_add(handle, gpu_mv(handle, A, x, am, an),
		gpu_mv(handle, B, y, bm, bn), am);
	printToHost(z, am, 1, "gpu_fns, A*x + B*y");

	// use CudaUtils
	CudaUtils* cuA = new CudaUtils(handle, A, am, an);
	CudaUtils* cuB = new CudaUtils(handle, B, bm, bn);
	
	cuA->mv(x) // A*x  => (am, 1)
		->add( // +
	cuB->mv(y)); // B*y => (bm, 1)

	z = cuA->getResDevVec();
	printToHost(z, am, 1, "CudaUtils, A*x + B*y");


	// -------- 尝试用 类 封装gpu_fns，有了类后可以做, e.g. gpu.mmul(A, B).add(gpu.mmul(C, D)); => A*B + C*D---------
	/*
	// G = D*E*F ，将gpu_fns串起来使用

	cout << "test CudaUtils" << endl;
	CudaUtils cu = CudaUtils(handle);

	device_vector<float> D(4 * 2);
	device_vector<float> E(2 * 3);
	device_vector<float> F(3 * 5);
	device_vector<float> G;

	for (int i = 0; i < 8; i++)
	{
		D[i] = float(i);
	}
	for (int i = 0; i < 6; i++)
	{
		E[i] = float(i / 2.0f);
	}
	for (int i = 0; i < 15; i++)
	{
		F[i] = float(1);
	}
	printToHost(D, 4, 2, "D pre");

	cu.mmul(D, E, 4, 2, 3);
	*/
	

	// ------------- device_vector<T> 测试泛型 ---------------
	/*
	cout << "test vec<int> 保存vec的地址" << endl; 
	device_vector<float*> d_pool;
	cout << "d_pool.size(): " << d_pool.size() << endl;

	d_pool.push_back(raw_pointer_cast(&D[0]));
	cout << "d_pool.size(): " << d_pool.size() << endl;

	float* d = d_pool[0];
	//device_vector<float> d_vec(d, d + 8); // 必须用device_vector还原
	printToHost(d, 4, 2, "d_vec");
	*/

	cublasDestroy(handle);
}

//__device__ void Cuda4RNN::g_tanh(float * d_out, float * d_in, int size)
//{
//	// kernel<<<1, bs>>>
//	int tid = threadIdx.x;
//
//	if (tid < size)
//	{
//		d_out[tid] = tanhf(d_in[tid]);
//	}
//}

