//
//
//__global__ void add_kernel(Para* d_para)
//{
//	int i = threadIdx.x;
//
//	d_para->d_c[i] = d_para->d_a[i] + d_para->d_b[i];
//}
//
//
//void initPara(Para** ha_para, Para** da_para, int size)
//{
//	// allocate struct mem
//	cudaMallocHost((void**)&ha_para, sizeof(Para));
//	cudaMalloc((void**)&da_para, sizeof(Para));
//
//	Para* h_para = *ha_para;
//
//	cudaMallocHost((void**)&h_para->h_a, size * sizeof(float));
//	cudaMallocHost((void**)&h_para->h_b, size * sizeof(float));
//	cudaMallocHost((void**)&h_para->h_c, size * sizeof(float));
//
//	cudaMalloc((void**)&h_para->d_a, size * sizeof(float));
//	cudaMalloc((void**)&h_para->d_b, size * sizeof(float));
//	cudaMalloc((void**)&h_para->d_c, size * sizeof(float));
//
//	// ha => da
//	cudaMemcpy(*da_para, *ha_para, sizeof(Para), cudaMemcpyHostToDevice);
//
//}
//
//void deInitPara(Para * h_para, Para* d_para)
//{
//	// free host mem
//	cudaFreeHost(h_para->h_a);
//	cudaFreeHost(h_para->h_b);
//	cudaFreeHost(h_para->h_c);
//
//	// free dev mem
//	cudaFree(h_para->d_a);
//	cudaFree(h_para->d_b);
//	cudaFree(h_para->d_c);
//
//	// release struct mem
//	cudaFreeHost(h_para);
//	cudaFree(d_para);
//}
//
//void addWithCuda(Para * h_para, Para* d_para, int size)
//{
//	cudaSetDevice(0);
//
//	cudaMemcpy(h_para->d_a, h_para->h_a, size * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(h_para->d_b, h_para->h_b, size * sizeof(float), cudaMemcpyHostToDevice);
//	cudaDeviceSynchronize();
//
//	cudaMemcpy(&d_para, &h_para, sizeof(Para), cudaMemcpyHostToDevice);
//	cudaDeviceSynchronize();
//
//	// launch kernel
//	add_kernel << <1, 5 >> > (d_para);
//
//	cudaMemcpy(&h_para, &d_para, sizeof(Para), cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_para->h_c, h_para->d_c, size * sizeof(float), cudaMemcpyDeviceToHost);
//	cudaDeviceSynchronize();
//
//}
//
//void struct_para_main()
//{
//	const int size = 5;
//	const float a[size] = { 1,2,3,4,5 };
//	const float b[size] = { 10,20,30,40,50 };
//	float c[size] = { 0 };
//	
//	Para *h_para, *d_para;
//	initPara(&h_para, &d_para, size);
//
//	memcpy(h_para->h_a, a, size * sizeof(float));
//	memcpy(h_para->h_b, b, size * sizeof(float));
//
//	addWithCuda(h_para, d_para, size);
//
//	memcpy(c, h_para->h_c, size * sizeof(float));
//
//	for (int i = 0; i < 5; i++)
//	{
//		cout << c[i] << "  ";
//	}
//	cout << endl;
//
//	deInitPara(h_para, d_para);
//	return;
//}
//
//
