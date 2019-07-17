//#include "Script.h"
//
//
//
//Script::Script()
//{
//}
//
//
//Script::~Script()
//{
//}
//
//void Script::test()
//{
//	cout << "hello, C++" << endl;
//	// ---------------- matlab -----------------
//	std::vector<SceStruct> lstrain =
//		IOMatlab::read("listStructTrain");
//
//	cout << lstrain.size() << endl;
//	/*SceStruct first = lstrain[0];
//	cout << "id: " << first.id
//		<< ", score: " << first.score << endl
//		<< "dim: " << first.matDataZScore.n_rows
//		<< " " << first.matDataZScore.n_cols << endl;*/
//		//<< "matDataZScore: \n" << first.matDataZScore << endl;
//
//		// ============ arma ======================
//	arma::mat m(3, 2, fill::randn);
//	m.print("m");
//	//m.reshape(6, 1); // col-major
//	//m.print("m reshape");
//
//
//	// ========= cuda ==================
//	cout << endl;
//	arma::mat m1(3, 2, fill::randn);
//	arma::mat m2(3, 1, fill::randn);
//	arma::mat m3(2, 2, fill::randn);
//
//	MyArray *marr1 = IOMatlab::mat2arr(m1);
//	MyArray *marr2 = IOMatlab::mat2arr(m2);
//	MyArray *marr3 = IOMatlab::mat2arr(m3);
//
//	cout << "sizeof arr1: " << sizeof(marr1) << endl;
//
//	CudaMap map = CudaMap();
//	map.put(0, marr1);
//	map.put(1, marr2);
//	map.put(3, marr3);
//	map.put(4, marr3);
//	map.put(4, marr1);
//	map.put(4, marr2);
//
//	cout << "map size: " << map.getSize() << endl;
//	auto keyList = map.getKeyListToHost(); // 此处必须是 toHost
//
//	cout << "keyList: " << endl;
//	for (int i = 0; i < keyList.size(); i++)
//	{
//		cout << keyList[i] << "\t";
//	}
//	cout << endl;
//
//	cout << "values in map:" << endl;
//	for (int i = 0; i < map.getSize(); i++)
//	{
//		int key = keyList[i];
//		cout << "---------\nkey is " << key << endl;
//
//		cout << "values: \n";
//		MyArray* marr = map.get(key);
//		cout << "n_rows_origin: " << marr->n_rows_origin << endl;
//		for (int j = 0; j < marr->size; j++) // 疑问：为什么 sizeof不能得到准确的arr大小, 因为arr内存大小是动态分配的, sizeof的大小将是一个指针变量的长度
//		{
//			cout << marr->arr[j] << endl;
//		}
//
//	}
//
//}
