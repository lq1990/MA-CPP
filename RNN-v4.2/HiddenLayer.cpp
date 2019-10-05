#include "HiddenLayer.h"



HiddenLayer::HiddenLayer(int n_features, int n_hidden_cur, int n_hidden_next)
{
	this->n_features = n_features;
	this->n_hidden_cur = n_hidden_cur;
	this->n_hidden_next = n_hidden_next;

	int H = n_hidden_cur;
	int Z = n_features + n_hidden_cur;
	int H_next = n_hidden_next;

	this->Wf = arma::randn(Z, H) / sqrt(Z / 2.); // W 包含h x, hidden forget
	this->Wi = arma::randn(Z, H) / sqrt(Z / 2.); // W 包含h x, hidden input
	this->Wc = arma::randn(Z, H) / sqrt(Z / 2.); // W 包含h x, hidden candidate cell
	this->Wo = arma::randn(Z, H) / sqrt(Z / 2.); // W 包含h x, hidden output
	this->Whh = arma::randn(H, H_next) / sqrt(H_next / 2.); //  W of hidden-hidden

	this->bf = arma::zeros(1, H); // 可看出来，默认 行向量
	this->bi = arma::zeros(1, H);
	this->bc = arma::zeros(1, H);
	this->bo = arma::zeros(1, H);
	this->bhh = arma::zeros(1, H_next);
}


HiddenLayer::~HiddenLayer()
{
}

map<int, mat> HiddenLayer::hiddenForward(mat inputs)
{
	// inputs.n_rows 是 samples的数量，inputs.n_cols 是 features数量
	for (int t = 0; t < inputs.n_rows; t++)
	{
		xs[t] = inputs.row(t); // (1,d)
		X[t] = arma::join_horiz(hs[t - 1], xs[t]); // X: concat [h_old, curx] (1,h+d)

		//hs[t] = arma::tanh(Wxh * xs[t] + Whh * hs[t - 1] + bh);

		h_fs[t] = this->sigmoid(X[t] * Wf + bf); // (1,h+d)(h+d,h) = (1,h)
		h_is[t] = this->sigmoid(X[t] * Wi + bi);
		h_os[t] = this->sigmoid(X[t] * Wo + bo);
		h_cs[t] = arma::tanh(X[t] * Wc + bc);

		cs[t] = h_fs[t] % cs[t - 1] + h_is[t] % h_cs[t]; // (1,h)
		hs[t] = h_os[t] % arma::tanh(cs[t]); // (1,h)

		/*
		
		if (t == inputs.n_rows - 1)
		{
			ys[t] = hs[t] * Wy + by; // (1,h)(h,y) = (1,y)
			ps[t] = RNN::softmax(ys[t]);

			loss += -log(ps[t](idx1));

			//cout << "loss: " << loss << endl;

			// accuracy
			uvec idx_max_ps = arma::index_max(ps[t], 1); // index_prediction

			mtx.lock();
			log_target.push_back(idx1);
			log_prediction.push_back(idx_max_ps(0));
			if (idx1 == idx_max_ps(0))
			{
				true_false.push_back(1);
			}
			else
			{
				true_false.push_back(0);
			}
			mtx.unlock();
		}

		*/

	}


	return hs;
}

map<string, mat> HiddenLayer::hiddenBackward(mat inputs, map<int, mat> d_outputs_in, map<int, mat>& d_outputs_out, double lambda0)
{
	mat dWf, dbf;
	mat dWi, dbi;
	mat dWc, dbc;
	mat dWo, dbo;
	mat dWhh, dbhh; // dWhh对于Whh的优化量

	dWf = arma::zeros<mat>(Wf.n_rows, Wf.n_cols);
	dWi = arma::zeros<mat>(Wi.n_rows, Wi.n_cols);
	dWc = arma::zeros<mat>(Wc.n_rows, Wc.n_cols);
	dWo = arma::zeros<mat>(Wo.n_rows, Wo.n_cols);
	dWhh = arma::zeros<mat>(Whh.n_rows, Whh.n_cols);
	dbf = arma::zeros<mat>(bf.n_rows, bf.n_cols);
	dbi = arma::zeros<mat>(bi.n_rows, bi.n_cols);
	dbc = arma::zeros<mat>(bc.n_rows, bc.n_cols);
	dbo = arma::zeros<mat>(bo.n_rows, bo.n_cols);
	dbhh = arma::zeros<mat>(bhh.n_rows, bhh.n_cols);

	// dnext init
	mat dhnext = arma::zeros<mat>(hs[0].n_rows, hs[0].n_cols);
	mat dcnext = arma::zeros<mat>(cs[0].n_rows, cs[0].n_cols);
	mat dxnext;

	mat dh, dho, dc, dhf, dhi, dhc;
	mat dXf, dXi, dXo, dXc, dX;

	// 反向遍历每个time step
	for (int t = (int)inputs.n_rows - 1; t >= 0; t--)
	{
		double lambda = (t == inputs.n_rows - 1) ? lambda0 : 0.; // 一次反传中只需要一次 惩罚项相加，无论time steps有多少


		// hidden to output (hidden-hidden) gradient
		dWhh += hs[t].t() * d_outputs_in[t] + lambda * Whh;
		dbhh += d_outputs_in[t];

		dh = d_outputs_in[t] * Whh.t() + dhnext; // adding dhnext

		// gradient for ho in h = ho * tanh(c)
		dho = arma::tanh(cs[t]) % dh;
		dho = MyLib<mat>::dsigmoid(h_os[t]) % dho;

		// gradient for c in h = ho * tanh(c), adding dcnext
		dc = h_os[t] % dh % MyLib<mat>::dtanh(cs[t]);
		dc = dc + dcnext;

		// gradient for hf in c = hf * c_old + hi * hc
		dhf = cs[t - 1] % dc; // c_old => cs[t-1]
		dhf = MyLib<mat>::dsigmoid(h_fs[t]) % dhf;

		// gradient for hi in c = hf * c_old + hi * hc
		dhi = h_cs[t] % dc;
		dhi = MyLib<mat>::dsigmoid(h_is[t]) % dhi;

		// gradient for hc in c = hf * c_old + hi * hc
		dhc = h_is[t] % dc;
		dhc = MyLib<mat>::dtanh(h_cs[t]) % dhc;

		// gate gradinets
		dWf += X[t].t() * dhf + lambda * Wf;
		dbf += dhf;
		dXf = dhf * Wf.t();

		dWi += X[t].t() * dhi + lambda * Wi;
		dbi += dhi;
		dXi = dhi * Wi.t();

		dWo += X[t].t() * dho + lambda * Wo;
		dbo += dho;
		dXo = dho * Wo.t();

		dWc += X[t].t() * dhc + lambda * Wc;
		dbc += dhc;
		dXc = dhc * Wc.t();

		// as X was used in multiple gates
		dX = dXo + dXc + dXi + dXf;

		// update dhnext dcnext
		dhnext = dX.cols(0, this->n_hidden_cur - 1); // 左闭右闭[]
		dxnext = dX.cols(this->n_hidden_cur, dX.n_cols-1);
		dcnext = h_fs[t] % dc;

		// 本hidden layer的dxnext会传给上一个hidden layer，要传递的值用 d_output_out存储
		d_outputs_out[t] = dxnext;
	}


	map<string, mat> mymap;
	mymap["dWf"] = dWf;
	mymap["dWi"] = dWi;
	mymap["dWc"] = dWc;
	mymap["dWo"] = dWo;
	mymap["dWhh"] = dWhh;
	mymap["dbf"] = dbf;
	mymap["dbi"] = dbi;
	mymap["dbc"] = dbc;
	mymap["dbo"] = dbo;
	mymap["dbhh"] = dbhh;

	return mymap;
}

mat HiddenLayer::map2mat(map<int, mat> mp)
{
	// mp的size()确定mat的行数；mp中任一个value（是一个行mat）的长度确定mat的列数
	mat mpmat = mat(mp.size(), mp[0].n_cols);

	mat tmp;
	for (int i = 0; i < mp.size(); i++)
	{
		// 取得mp[i]的value
		tmp = mp[i]; // mp的每个value都是一个 行mat

		// 赋给mpmat
		for (int j=0; j<tmp.n_cols; j++) 
		{
			mpmat[i, j]= tmp[j];
		}
	}

	return mpmat;
}

void HiddenLayer::saveParams(string title)
{
	this->Wf.save("Wf"+title+".txt", file_type::raw_ascii);
	this->Wi.save("Wi"+title+".txt", file_type::raw_ascii);
	this->Wc.save("Wc"+title+".txt", file_type::raw_ascii);
	this->Wo.save("Wo"+title+".txt", file_type::raw_ascii);
	this->Whh.save("Whh"+title+".txt", file_type::raw_ascii);

	this->bf.save("bf" + title + ".txt", file_type::raw_ascii);
	this->bi.save("bi" + title + ".txt", file_type::raw_ascii);
	this->bc.save("bc" + title + ".txt", file_type::raw_ascii);
	this->bo.save("bo" + title + ".txt", file_type::raw_ascii);
	this->bhh.save("bhh" + title + ".txt", file_type::raw_ascii);

}

void HiddenLayer::loadParams(string title)
{
	this->Wf.load("Wf" + title + ".txt", file_type::raw_ascii);
	this->Wi.load("Wi" + title + ".txt", file_type::raw_ascii);
	this->Wc.load("Wc" + title + ".txt", file_type::raw_ascii);
	this->Wo.load("Wo" + title + ".txt", file_type::raw_ascii);
	this->Whh.load("Whh" + title + ".txt", file_type::raw_ascii);

	this->bf.load("bf" + title + ".txt", file_type::raw_ascii);
	this->bi.load("bi" + title + ".txt", file_type::raw_ascii);
	this->bc.load("bc" + title + ".txt", file_type::raw_ascii);
	this->bo.load("bo" + title + ".txt", file_type::raw_ascii);
	this->bhh.load("bhh" + title + ".txt", file_type::raw_ascii);
}

mat HiddenLayer::sigmoid(arma::mat mx)
{
	// sigmoid = 1 / (1+exp(-x))
	return 1. / (1 + arma::exp(-mx));
}

