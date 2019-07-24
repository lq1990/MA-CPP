#pragma once

const float total_epoches = 51;
const float n_features = 17;
const float n_hidden = 50;
const float n_output_classes = 10;
const float alpha = 0.1f;
const float score_min = 6.0f;
const float score_max = 8.9f;


typedef struct Para
{
	// --------- RNN params，need cudaMemcpyH2D----------------
	float* h_total_epoches;
	float* h_n_features;
	float* h_n_hidden;
	float* h_n_output_classes;
	float* h_alpha;
	float* h_score_min;
	float* h_score_max;

	float* d_total_epoches;
	float* d_n_features;
	float* d_n_hidden;
	float* d_n_output_classes;
	float* d_alpha;
	float* d_score_min;
	float* d_score_max;

	// ---------- d_params that are used to update W b, 
	// only in GPU needed -------------
	float* d_dWxh;
	float* d_dWhh;
	float* d_dWhy;
	float* d_dbh;
	float* d_dby;
	float* d_dhnext;
	float* d_dy;
	float* d_dh;
	float* d_dhraw;

	// ------- params to be trained, need D2H --------------
	float* h_Wxh;
	float* h_Whh;
	float* h_Why;
	float* h_bh;
	float* h_by;

	float* d_Wxh;
	float* d_Whh;
	float* d_Why;
	float* d_bh;
	float* d_by;

	// ---- scenarios data, need H2D 并且需要在IOMatlab中确定内存 --------------
	float* h_sces_id_score;// [id0,score0, id1,score1, id2,score2, ...]
	float* h_sces_data;// all data
	float* h_sces_data_mn;// [m0,n0, m1,n1, m2,n2, ...]
	float* h_sces_data_idx_begin;// [idx0, idx1, idx2, ...]
	float* h_num_sces; // [0]
	float* h_total_size; // [0]

	float* d_sces_id_score;// [id0,score0, id1,score1, id2,score2, ...]
	float* d_sces_data;// all data
	float* d_sces_data_mn;// [m0,n0, m1,n1, m2,n2, ...]
	float* d_sces_data_idx_begin;// [idx0, idx1, idx2, ...]
	float* d_num_sces; // [0]
	float* d_total_size; // [0]

	// ----- map to saves states, 
	// only in GPU needed ---------
	float* d_xs;
	float* d_hs; // n_cols of hs is 1 more than others
	float* d_ys;
	float* d_ps;
	float* d_Nmax;

	// ------- cache, 
	// only in GPU needed -------------
	float* d_tmp_d_vec;
	float* d_tmp_d_vec2;
	float* d_W_tmp1;
	float* d_W_tmp2;
	float* d_W_tmp3;

};