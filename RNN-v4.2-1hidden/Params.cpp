#include "Params.h"



Params::Params(int n_input_dim, int n_hidden_cur, int n_hidden_next)
{
	int H = n_hidden_cur;
	int Z = n_input_dim + n_hidden_cur;
	int H_next = n_hidden_next;

	// ��ʼ���������Ҳ����load���ز������г�ʼ��
	/*
	*/
	this->Wf = arma::randn(Z, H) / sqrt(Z / 2.); // W ����h x, hidden forget
	this->Wi = arma::randn(Z, H) / sqrt(Z / 2.); // W ����h x, hidden input
	this->Wc = arma::randn(Z, H) / sqrt(Z / 2.); // W ����h x, hidden candidate cell
	this->Wo = arma::randn(Z, H) / sqrt(Z / 2.); // W ����h x, hidden output
	this->Whh = arma::randn(H, H_next) / sqrt(H_next / 2.); //  W of hidden-hidden

	/*
	this->Wf = arma::randn(Z, H) * 0.01; // W ����h x, hidden forget
	this->Wi = arma::randn(Z, H) * 0.01; // W ����h x, hidden input
	this->Wc = arma::randn(Z, H) * 0.01; // W ����h x, hidden candidate cell
	this->Wo = arma::randn(Z, H) * 0.01; // W ����h x, hidden output
	this->Whh = arma::randn(H, H_next) * 0.01; //  W of hidden-hidden
	*/

	this->bf = arma::zeros(1, H); // �ɿ�������Ĭ�� ������
	this->bi = arma::zeros(1, H);
	this->bc = arma::zeros(1, H);
	this->bo = arma::zeros(1, H);
	this->bhh = arma::zeros(1, H_next);
}


Params::~Params()
{
}

void Params::save(string title)
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

void Params::load(string title)
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
