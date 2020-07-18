# include "decode.h"
# include <mkl.h>
# include <memory>
# include <string.h>
# include <assert.h>
# include <numeric>
# include <algorithm>
# include <iostream>
# include <time.h>
# include <chrono>

NMT::Decoder::Decoder(
	const size_t& head_num,
	const size_t& hidden_num,
	const size_t& layer_num,
	const size_t& vocabe_size,
	const size_t& filter_size,
	vector<vector<float*>>& weight,
	float* weight_embedding,
	float* weight_language,
	float* weight_scale,
	float* weight_bias,
	float* logit_weight)
{
	this->head_num = head_num;
	this->hidden_num = hidden_num;
	this->layer_num = layer_num;
	this->vocabe_size = vocabe_size;
	this->filter_size = filter_size;
	this->weight = weight;
	this->weight_embedding = weight_embedding;
	this->weight_language = weight_language;
	this->weight_scale = weight_scale;
	this->weight_bias = weight_bias;
	this->logit_weight = logit_weight;
}

void NMT::Decoder::SetCache(const float* encode_out, const size_t& batch_size, const size_t& length, vector<vector<float>>& cache_out_k, vector<vector<float>>& cache_out_v)
/* @brief: get k and v for encdec_attention
 * @param:
 *     	encode_out: the result of encoder,the size is [batch_size, length, hidden_num]
 *	batch_size: the input of batch_size
 *	length: the length of a sentence
 *	cache_out_k: output of this function, for encdec_attention, the size is [bathc_size, length, hidden_num] 
 *	cache_out_v: output of this function, for encdec_attention, the size is [bathc_size, length, hidden_num] 
 * @usage: 
 */
{
	for (int i = 0; i < layer_num; i++)
	{
		MKL_INT m = batch_size * length;
		MKL_INT k = hidden_num;
		MKL_INT n = hidden_num;
		float alpha = 1.0;
		float beta = 0.0;
		//set encdec_k of the i layer 
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			m, n, k, alpha,
			encode_out, k,
			weight[i][9], n, beta,
			cache_out_k[i].data(), n);
		//set encdec_v of the i layer 
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			m, n, k, alpha,
			encode_out, k,
			weight[i][10], n, beta,
			cache_out_v[i].data(), n);
	}
	
}

void NMT::Decoder::EmbeddingInit(vector<int>& language_id, vector<float>& embedding_word)
/* @brief: change int to vector(embedding) 
 * @param:
 *     	language_id: input, the id of language 
 *	embedding_word: output
 * @usage: [1],[0] -> [0.1,0.2..., 0.1023], [0.1, 0.2..., 0.1023]
 */
{
	embedding_word.clear();
	int size = language_id.size();
	for(int i=0; i<size; i++)
	{
	  float* begin = weight_language + hidden_num * language_id[i];
	  embedding_word.insert(embedding_word.end(), begin, begin + hidden_num);
	};
}

void NMT::Decoder::EmbeddingLookup(vector<size_t>& input, vector<float>& embedding_word)
/* @brief: change int to vector(embedding) 
 * @param:
 *     	input: the id of word  
 *	embedding_word: output
 * @usage: [211],[985] -> [0.1,0.2..., 0.1023], [0.1, 0.2..., 0.1023]
 */
{
	embedding_word.clear();
	//assert(input <= vocabe_size);
        int size = input.size();
	for(int i=0; i<size; i++)
	{
	  float* begin = weight_embedding + hidden_num * input[i];
	  embedding_word.insert(embedding_word.end(), begin, begin + hidden_num);
	};
	for(auto& info:embedding_word)
	{
	 info *= 32.0;
	};
}

void NMT::Decoder::GetPosWeight(vector<float>& position_weight, size_t decode_length, size_t hidden_num)
/* @brief: get position encode(abandon it now)  
 */
{
	// get position
	int i = 0;
	vector<float> position(decode_length);
	generate(position.begin(), position.end(), [&](){return i++;});

	int num_timescales = hidden_num / 2; 
	float log_timescale_increment = 9.21034 / std::max(int(num_timescales - 1),1);

	i = 0;
	vector<float> inv_timescales(num_timescales);
	generate(inv_timescales.begin(), inv_timescales.end(), [&]() {return i++; });
	for_each(inv_timescales.begin(), inv_timescales.end(), [&](float& x) { x = std::exp(x * (-log_timescale_increment)); });

	vector<float> scaled_time_sin(decode_length * num_timescales, 0.1);
	vector<float> scaled_time_cos(decode_length * num_timescales, 0.1);
	MKL_INT m = decode_length;
	MKL_INT k = 1;
	MKL_INT n = num_timescales;
	float alpha = 1.0;
	float beta = 0.0;

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, n, k, alpha,
		position.data(), k,
		inv_timescales.data(), n, beta,
		scaled_time_sin.data(), n);

	for (int i = 0; i < decode_length * num_timescales; i++)
	{
		scaled_time_cos[i] = std::cos(scaled_time_sin[i]);
		scaled_time_sin[i] = std::sin(scaled_time_sin[i]);
	}

	//concat
	for (int i = 0; i < decode_length; i++)
	{
		position_weight.insert(position_weight.end(), scaled_time_sin.begin() + num_timescales * i, scaled_time_sin.begin() + num_timescales * i + num_timescales);
		position_weight.insert(position_weight.end(), scaled_time_cos.begin() + num_timescales * i, scaled_time_cos.begin() + num_timescales * i + num_timescales);
	}
}

void NMT::Decoder::PositionEncode(const size_t& time, vector<float>& embedding_word, vector<float>& weight_position)
/* @brief:  add position encode(abandon it now)  
 */
{
	vector<float>::iterator begin = weight_position.begin() + hidden_num * time;

	for (int i = 0; i < embedding_word.size(); i++)
	{
		embedding_word[i] += *(begin + i % hidden_num);
	}
}

void NMT::Decoder::LayerPreprocess(vector<float>& layer_input, const size_t& batch_size, const size_t& length, const float* scale, const float* bias)
/* @brief: preprocess
 * @param:
 *	batch_size: the input of batch_size
 *	length: the length of a sentence
 * @usage: 
 *      layer_input:|1,2,.., 1024| ,|2,3..., 1025|.....|n, n+1, ..., n+1023|, n = batch_size * length
 * */
{
        //epsilon
        float epsilon = 1E-6;
        vector<float>::iterator begin = layer_input.begin();

        for(int i=0; i < batch_size * length; i++ )
        {
                //mean
                float sum = accumulate(begin, begin + hidden_num, 0.0);
                float mean = sum / hidden_num;
                //variance
                float accum = 0.0;
                for_each(begin, begin + hidden_num, [&](const float& d) {accum += (d - mean) * (d - mean); });
                float variance = accum / hidden_num;
                //norm
                for_each(begin, begin + hidden_num, [&](float& d) { d = (d - mean) / sqrt(variance + epsilon); });
                //mut and add
                for (int j = 0; j < hidden_num; j++, begin++)
                {
                        *begin = (*begin) * scale[j] + bias[j];
                }
        }

}

void NMT::Decoder::LayerPostprocess(vector<float>& layer_input, const vector<float>& temp)
/* @brief: postprocess(layer_input + temp)
 * @param:
 *	layer_input: input_1
 *	temp: input_2
 * @usage: 
 * */
{
	//Add
	for (int i = 0; i < layer_input.size(); i++)
	{
		layer_input[i] = layer_input[i] + temp[i];
	}
}

void  NMT::Decoder::FeedForward(const vector<float>& input, vector<float>& output, const size_t& batch_size, const size_t& length,  int filter, const float* weight, float* bias, string activation)
/* @brief: Matrix multiplication  
 * @param:
 * 	length: the defalut is 1 because of decode 
 *	filter: the size of output ,[batch_size, length, filter]
 *	activation: relu/NUll/... 
 * @usage: 
 * */
{
	MKL_INT m = batch_size * length;
	MKL_INT k = input.size()/m;
	MKL_INT n = filter;
	float alpha = 1.0;
	float beta = 1.0;
	//vector<float> tem_q(m * n, 0.0);
	for (int i =0; i < batch_size * length; i++)
	{
		memcpy(output.data() + i * n, bias, n * 4);//the byte of float is four times to char
	}
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, n, k, alpha,
		input.data(), k,
		weight, n, beta,
		output.data(), n);
	if (activation == "relu")
	{
	  float min = 0.0;
          int size = m * n;
	  for (int i = 0; i < size; i++)
	  {
	      output[i] = max(min, output[i]);
      	  }
	}
}

void NMT::Decoder::GetPositionX(const float* position_embedding, const size_t max_length, const size_t& length, vector<float>& position_x)
/* @brief: get position and embedding position  
 * @param:
 * 	position_embedding: embedding table 
 *	length: this length represents the length of the current k or v
 *	position_x:output 
 * @usage: 
 * */
{
	int max = 2 * max_length;
	vector<int> mat(length);
	for (int i = 0; i < length; i++ )
	{
		int tmp = i - length + 1  + max_length;
		mat[i] = tmp<0? 0:tmp;
		const float* begin = position_embedding + hidden_num / head_num * mat[i];
		position_x.insert(position_x.end(), begin, begin + hidden_num / head_num);
	}
}

void NMT::Decoder::MulPositionKey(const size_t& batch_size, const size_t& length, float* input, float* position_key, float* out)
/* @brief: q * position_key  
 * @param:
 * 	input: q
 *	length: this length represents the length of the current k
 * */
{

        #define GRP_COUNT 1

	MKL_INT    b_m[GRP_COUNT] = { head_num };
	MKL_INT    b_k[GRP_COUNT] = { hidden_num / head_num };
	MKL_INT    b_n[GRP_COUNT] = { length };

	MKL_INT    lda[GRP_COUNT] = { hidden_num / head_num };
	MKL_INT    ldb[GRP_COUNT] = { hidden_num / head_num };
	MKL_INT    ldc[GRP_COUNT] = { length };


	CBLAS_TRANSPOSE    transA[GRP_COUNT] = { CblasNoTrans };
	CBLAS_TRANSPOSE    transB[GRP_COUNT] = { CblasTrans };
	float    b_alpha[GRP_COUNT] = { 1.0 };
	float    b_beta[GRP_COUNT] = { 0.0 };

	const MKL_INT    size_per_grp[GRP_COUNT] = { batch_size };

	vector<float*> a_array(batch_size);
	vector<float*> b_array(batch_size);
	vector<float*> c_array(batch_size);
	for (int i = 0; i < batch_size; ++i)
	{
		a_array[i] = input + i * hidden_num;
		b_array[i] = position_key;
		c_array[i] = out + i *(length * head_num);
	}
	cblas_sgemm_batch(CblasRowMajor, transA, transB,
		b_m, b_n, b_k, b_alpha,
		const_cast<const float**>(a_array.data()), lda,
		const_cast<const float**>(b_array.data()), ldb, b_beta,
		c_array.data(), ldc,
		GRP_COUNT, size_per_grp);
}

void NMT::Decoder::MulPositionValue(const size_t& batch_size, const size_t& length, float*input, float* position_val, float* out)
/* @brief: softmax * position_key  
 * @param:
 * 	input: softmax
 *	length: this length represents the length of the current v
 * */
{
        #define GRP_COUNT 1

	MKL_INT    b_m[GRP_COUNT] = { 1 };
	MKL_INT    b_k[GRP_COUNT] = { length };
	MKL_INT    b_n[GRP_COUNT] = { hidden_num / head_num };

	MKL_INT    lda[GRP_COUNT] = { length * head_num };
	MKL_INT    ldb[GRP_COUNT] = { hidden_num / head_num};
	MKL_INT    ldc[GRP_COUNT] = { hidden_num / head_num};


	CBLAS_TRANSPOSE    transA[GRP_COUNT] = { CblasNoTrans };
	CBLAS_TRANSPOSE    transB[GRP_COUNT] = { CblasNoTrans };
	float    b_alpha[GRP_COUNT] = { 1.0 };
	float    b_beta[GRP_COUNT] = { 0.0 };

	const MKL_INT    size_per_grp[GRP_COUNT] = { batch_size * head_num };

	vector<float*> a_array(batch_size * head_num);
	vector<float*> b_array(batch_size * head_num);
	vector<float*> c_array(batch_size * head_num);
	for (int i = 0; i < batch_size * head_num; ++i)
	{
		a_array[i] = input + i * length;
		b_array[i] = position_val;
		c_array[i] = out + i * (hidden_num/head_num);
	}
	cblas_sgemm_batch(CblasRowMajor, transA, transB,
		b_m, b_n, b_k, b_alpha,
		const_cast<const float**>(a_array.data()), lda,
		const_cast<const float**>(b_array.data()), ldb, b_beta,
		c_array.data(), ldc,
		GRP_COUNT, size_per_grp);

}

void NMT::Decoder::BuildBias(const size_t& batch_size, const size_t& length,  int* mask, float* bias)
/* @brief: the bias for encdec  
 * @param:
 * 	mask:for example [1,1,1,0,0,   1,1,1,1,0]
 *	length: the length of a sentence
 * */
{
	for (int i = 0; i < batch_size*length; i++)
	{
		bias[i] *= (1-mask[i]);
	}
}

void NMT::Decoder::AddBias(float* input, const float* bias, const size_t& batch_size, const size_t& length)
/* @brief: adc bias  
 */
{
	int one_batch_length = length * head_num;
	for (int i = 0; i < batch_size; i++)
	{
		const float* begin_bias = bias + i * length;
		float* begin_input = input + i * one_batch_length;
		for (int j = 0; j < one_batch_length; j++)
		{
			begin_input[j] += begin_bias[j % length];
		}
	}
}

void NMT::Decoder::SelfAttention(float* input, const size_t& batch_size, const size_t& length, const float* q_weight, const float* k_weight, const float* v_weight, const float* key_weight, const float* value_weight, const float* weight, float* output, vector<float>& k_value, vector<float>& v_value, const float* bias)
/* @brief: self attehtion  
 * @param:
 * 	key_weight: the embedding table for position_key
 *	value_weight: the embedding table for position_val
 *	weight: for last dense
 *	k_value: cache k, need to change every time
 *	v_value: cache_v, need to change every time
 *	bias: slef_bias, but it does't work here
 * */
{
	MKL_INT m = batch_size*length;
	MKL_INT k = hidden_num;
	MKL_INT n = hidden_num;
	float alpha = 1.0;
	float beta = 0.0;

	//compute q,k,v
	vector<float> tem_q(m * n, 0.0);
	vector<float> tem_k(m * n, 0.0);
	vector<float> tem_v(m * n, 0.0);

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			m, n, k, alpha,
			input, k,
			q_weight, n, beta,
			tem_q.data(), n);

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			m, n, k, alpha,
			input, k,
			k_weight, n, beta,
			tem_k.data(), n);

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			m, n, k, alpha,
			input, k,
			v_weight, n, beta,
			tem_v.data(), n);
	//cache
	//i is the time, list may be better
	int t = k_value.size()/(hidden_num*batch_size);
	for (int i=1; i<=batch_size; i++)
	{
		k_value.insert(k_value.begin() + (i*t+i-1)*hidden_num, tem_k.begin()+(i-1)*hidden_num, tem_k.begin()+i*hidden_num);
		v_value.insert(v_value.begin() + (i*t+i-1)*hidden_num, tem_v.begin()+(i-1)*hidden_num, tem_v.begin()+i*hidden_num);
	}
	// compute q/sqrt(64)
	 float q_mul_num = (1.0 / sqrt(hidden_num / head_num)); 
	 for_each(tem_q.begin(), tem_q.end(), [&](float& d) {d *= q_mul_num;});
	 		
	//dot_product_attention
        #define GRP_COUNT 1

        //the result of q*k
        int length_1 = 1;
        int length_2 = k_value.size()/(hidden_num*batch_size);
	vector<float> tem_q_k(head_num * batch_size * length_1 * length_2, 0.0);

	MKL_INT    b_m[GRP_COUNT] = { length_1 };
	MKL_INT    b_k[GRP_COUNT] = { hidden_num / head_num };
	MKL_INT    b_n[GRP_COUNT] = { length_2 };

	MKL_INT    lda[GRP_COUNT] = { hidden_num };
	MKL_INT    ldb[GRP_COUNT] = { hidden_num };
	MKL_INT    ldc[GRP_COUNT] = { head_num * length_2 };


	CBLAS_TRANSPOSE    transA[GRP_COUNT] = { CblasNoTrans };
	CBLAS_TRANSPOSE    transB[GRP_COUNT] = { CblasTrans };
	float    b_alpha[GRP_COUNT] = { 1.0 };
	float    b_beta[GRP_COUNT] = { 0.0 };

	const MKL_INT    size_per_grp[GRP_COUNT] = { head_num*batch_size };

	vector<float*> a_array(head_num * batch_size);
	vector<float*> b_array(head_num * batch_size);
	vector<float*> c_array(head_num * batch_size);
	for (int i = 0; i < batch_size; ++i)
	 {
		for(int j = 0; j < head_num; j++)
		{
			a_array[i*head_num+j] = tem_q.data() + i * length_1 * hidden_num + j * (hidden_num/head_num);
			b_array[i*head_num+j] = k_value.data() + i * length_2 * hidden_num + j * (hidden_num/head_num);
			c_array[i*head_num+j] = tem_q_k.data() + i * length_1 * (head_num*length_2) + j * length_2;
		}
	}

	cblas_sgemm_batch(CblasRowMajor, transA, transB,
		b_m, b_n, b_k, b_alpha,
		const_cast<const float**>(a_array.data()), lda,
		const_cast<const float**>(b_array.data()), ldb, b_beta,
		c_array.data(), ldc,
		GRP_COUNT, size_per_grp);
	//get relative_position_key
	vector<float> position_key;
	vector<float> position_q(length_2 * head_num * batch_size);
	GetPositionX(key_weight, 20, length_2, position_key);
	MulPositionKey(batch_size, length_2, tem_q.data(), position_key.data(), position_q.data());
        // the result of tem_q_k + position_q 
 	for(int i=0; i<tem_q_k.size(); i++)
 	{
 		tem_q_k[i] += position_q[i];
	};
	//add self_attention_bias
	// add 0.0 , don't know why, so do nothing 
	//softmax
	BatchSoftmax(tem_q_k.data(), b_n[0], head_num, batch_size, length);
	//softamx * v

	b_m[0] = length_1;
	b_k[0] = length_2;
	b_n[0] = hidden_num / head_num;

	lda[0] = length_2 * head_num;
	ldb[0] = hidden_num;
	ldc[0] = hidden_num;

	transA[0] = CblasNoTrans;
	transB[0] = CblasNoTrans;
	b_alpha[0] = { 1.0 };
	b_beta[0] = { 0.0 };


	for (int i = 0; i < batch_size; ++i) 
	{
		for(int j = 0; j < head_num; j++)
		{
		a_array[i*head_num+j] = tem_q_k.data() + i * length_1 * (head_num*length_2) + j * length_2;
		b_array[i*head_num+j] = v_value.data() + i * length_2 * hidden_num + j * (hidden_num/head_num);
		c_array[i*head_num+j] = tem_q.data() + i * length_1 * hidden_num + j * (hidden_num/head_num) ;
		}
	}
	cblas_sgemm_batch(CblasRowMajor, transA, transB,
		b_m, b_n, b_k, b_alpha,
		const_cast<const float**>(a_array.data()), lda,
		const_cast<const float**>(b_array.data()), ldb, b_beta,
		c_array.data(), ldc,
		GRP_COUNT, size_per_grp);
	//get position_value
	vector<float> position_value;
	vector<float> position_v (batch_size * hidden_num);
	GetPositionX(value_weight, 20, length_2, position_value);
	MulPositionValue(batch_size, length_2, tem_q_k.data(), position_value.data(), position_v.data());

	for(int i=0; i<tem_q.size(); i++)
	{
		tem_q[i] += position_v[i];
	}
	//last dense
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, n, k, alpha,
		tem_q.data(), k,
		weight, n, beta,
		output, n);
}

void NMT::Decoder::EncdecAttention(float* input, const size_t& batch_size, const size_t& length, const float* q_weight, const float* weight, float* output, vector<float>& k_value, vector<float>& v_value, const float* bias)
/* @brief: encdec attehtion  
 * @param:
 *	weight: for last dense
 *	k_value: cache k
 *	v_value: cache_v
 *	bias: encdec bias 
 * */
{
	MKL_INT m = batch_size*length;
	MKL_INT k = hidden_num;
	MKL_INT n = hidden_num;
	float alpha = 1.0;
	float beta = 0.0;

	vector<float> tem_q(m * n, 0.0);
	//compute q,k,v
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			m, n, k, alpha,
			input, k,
			q_weight, n, beta,
			tem_q.data(), n);

	// compute q/sqrt(64)
	 float q_mul_num = (1.0 / sqrt(hidden_num / head_num)); 
	 for_each(tem_q.begin(), tem_q.end(), [&](float& d) {d *= q_mul_num;});
	//dot_product_attention
        #define GRP_COUNT 1

        // the result of q*k
        int length_1 = 1;
        int length_2 = k_value.size()/(hidden_num*batch_size);
	vector<float> tem_q_k(head_num * batch_size * length_1 * length_2, 0.0);

	MKL_INT    b_m[GRP_COUNT] = { length_1 };
	MKL_INT    b_k[GRP_COUNT] = { hidden_num / head_num };
	MKL_INT    b_n[GRP_COUNT] = { length_2 };

	MKL_INT    lda[GRP_COUNT] = { hidden_num };
	MKL_INT    ldb[GRP_COUNT] = { hidden_num };
	MKL_INT    ldc[GRP_COUNT] = { head_num * length_2 };


	CBLAS_TRANSPOSE    transA[GRP_COUNT] = { CblasNoTrans };
	CBLAS_TRANSPOSE    transB[GRP_COUNT] = { CblasTrans };
	float    b_alpha[GRP_COUNT] = { 1.0 };
	float    b_beta[GRP_COUNT] = { 0.0 };

	const MKL_INT    size_per_grp[GRP_COUNT] = { head_num*batch_size };

	vector<float*> a_array(head_num * batch_size);
	vector<float*> b_array(head_num * batch_size);
	vector<float*> c_array(head_num * batch_size);
	for (int i = 0; i < batch_size; ++i)
	 {
		for(int j = 0; j < head_num; j++)
		{
			a_array[i*head_num+j] = tem_q.data() + i * length_1 * hidden_num + j * (hidden_num/head_num);
			b_array[i*head_num+j] = k_value.data() + i * length_2 * hidden_num + j * (hidden_num/head_num);
			c_array[i*head_num+j] = tem_q_k.data() + i * length_1 * (head_num*length_2) + j * length_2;
		}
	}

	cblas_sgemm_batch(CblasRowMajor, transA, transB,
		b_m, b_n, b_k, b_alpha,
		const_cast<const float**>(a_array.data()), lda,
		const_cast<const float**>(b_array.data()), ldb, b_beta,
		c_array.data(), ldc,
		GRP_COUNT, size_per_grp);
	//add attention_bias
	AddBias(tem_q_k.data(), bias, batch_size, length_2);
	//softmax
	BatchSoftmax(tem_q_k.data(), b_n[0], head_num, batch_size, length);
	//softamx * v
	b_m[0] = length_1;
	b_k[0] = length_2;
	b_n[0] = hidden_num / head_num;

	lda[0] = length_2 * head_num;
	ldb[0] = hidden_num;
	ldc[0] = hidden_num;

	transA[0] = CblasNoTrans;
	transB[0] = CblasNoTrans;
	b_alpha[0] = { 1.0 };
	b_beta[0] = { 0.0 };


	for (int i = 0; i < batch_size; ++i) 
	{
		for(int j = 0; j < head_num; j++)
		{
		a_array[i*head_num+j] = tem_q_k.data() + i * length_1 * (head_num*length_2) + j * length_2;
		b_array[i*head_num+j] = v_value.data() + i * length_2 * hidden_num + j * (hidden_num/head_num);
		c_array[i*head_num+j] = tem_q.data() + i * length_1 * hidden_num + j * (hidden_num/head_num) ;
		}
	}
	cblas_sgemm_batch(CblasRowMajor, transA, transB,
		b_m, b_n, b_k, b_alpha,
		const_cast<const float**>(a_array.data()), lda,
		const_cast<const float**>(b_array.data()), ldb, b_beta,
		c_array.data(), ldc,
		GRP_COUNT, size_per_grp);
	//last dense
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, n, k, alpha,
		tem_q.data(), k,
		weight, n, beta,
		output, n);
}

void NMT::Decoder::GenSoftmax(float* input, int num)
{
	float sum = 0.0;
	for (int j = 0; j < num; j++)
	{
		input[j] = exp(input[j]);
		sum += input[j];
	}
	for (int j = 0; j < num; j++)
	{
		input[j] /= sum;
	}

}

void NMT::Decoder::BatchSoftmax(float* input_qk, int k, int head_num, const size_t& batch_size, const size_t& length)
/*
 * |1,2,3|4,5,6|7,8,9|....|2,3,4| is the result of q*k, head_num is 16
 * k is 3 , the number of word
*/
{
	for (int i = 0; i < head_num * batch_size * length; i++)
	{
		float* data = input_qk + i * k;
		float sum = 0.0;
                # pragma omp simd reduction(+:sum)
		for (int j = 0; j < k; j++)
		{
			data[j] = exp(data[j]);
			sum += data[j];
		}
                # pragma omp simd
		for (int j = 0; j < k; j++)
		{
			data[j] /= sum;
		}
	}
}

void NMT::Decoder::ToLogits(float* input, const size_t& batch_size, float* weight, float* output)
/* @brief: 1024 -> 32768 
 * */
{
	MKL_INT m = batch_size;
	MKL_INT k = hidden_num;
	MKL_INT n = vocabe_size;
	float alpha = 1.0;
	float beta = 0.0;
       
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		m, n, k, alpha,
		input, k,
		weight, k, beta,
		output, n);
}

vector<size_t> NMT::Decoder::GetMax(vector<float>& logit, const size_t& batch_size)
/* @brief: get the index(word id) of max value in each list.
 * */
{
	vector<size_t> result(batch_size, 1);
	for(int i=0; i<batch_size; i++)
	{
	vector<float>::iterator begin = logit.begin() + i * vocabe_size;
	vector<float>::iterator biggest = std::max_element(begin, begin+vocabe_size);
	result[i]=std::distance(begin, biggest);
	}
        return result;
}

vector<size_t> NMT::Decoder::Decode(vector<float>& embedding_word,
                            const size_t& batch_size,
                            const size_t& length,
		            vector<vector<float>>& encode_out_k,
                            vector<vector<float>>& encode_out_v,
                            vector<vector<float>>& cache_k,
                            vector<vector<float>>& cache_v,
			    vector<float>& self_bias,
		            vector<float>& encdec_bias)
{
	for (int i = 0; i < layer_num; i++)
	{
		//self_attention
		//vector<float> attention_out(hidden_num, 0.0);
		vector<float> attention_out = embedding_word;
		LayerPreprocess(attention_out, batch_size, 1, weight[i][0], weight[i][1]);	
		SelfAttention(attention_out.data(),batch_size, 1,  weight[i][2], weight[i][3], weight[i][4], weight[i][18], weight[i][19], weight[i][5], attention_out.data(), cache_k[i], cache_v[i], self_bias.data());
		LayerPostprocess(embedding_word, attention_out);
		//encdec_atttion
		attention_out = embedding_word;
		LayerPreprocess(attention_out, batch_size, 1, weight[i][6], weight[i][7]);
		
		EncdecAttention(attention_out.data(), batch_size, 1, weight[i][8], weight[i][11], attention_out.data(), encode_out_k[i], encode_out_v[i], encdec_bias.data());
		LayerPostprocess(embedding_word, attention_out);
		//fnn
		vector<float> ffn_out_1(filter_size*batch_size, 0.0);
		attention_out = embedding_word;
		//vector<float> ffn_out_2(hidden_num, 0.0);
		LayerPreprocess(attention_out, batch_size, 1, weight[i][12], weight[i][13]);
		FeedForward(attention_out, ffn_out_1,batch_size, 1, filter_size, weight[i][14], weight[i][15], "relu");
		FeedForward(ffn_out_1, attention_out, batch_size, 1, hidden_num, weight[i][16], weight[i][17], "none");
		LayerPostprocess(embedding_word, attention_out);
	}
	//layer_preprocess
	LayerPreprocess(embedding_word, batch_size, 1, weight_scale, weight_bias);
	//to logit
	vector<float> logit(batch_size * vocabe_size, 0.0);
	ToLogits(embedding_word.data(), batch_size, logit_weight, logit.data());
	//get_word
	vector<size_t> word = GetMax(logit, batch_size);
	return word;
}

vector<vector<size_t>> NMT::Decoder::Translate(vector<float>& encode_out, const size_t& batch_size, const size_t& length, vector<int>& language_id, vector<int>& mask, int decode_length)
{
	vector<size_t> word(batch_size, 1);
	vector<vector<size_t>> result(batch_size);
	vector<float> embedding_word;
	vector<float> position_weight;
	vector<float> self_bias;
	vector<float> encdec_bias(batch_size*length, -1e9);
	vector<vector<float>> cache_encode_k(layer_num, vector<float>(encode_out.size(), 0.0));
	vector<vector<float>> cache_encode_v(layer_num, vector<float>(encode_out.size(), 0.0));
	vector<vector<float>> cache_k(layer_num);
	vector<vector<float>> cache_v(layer_num);

	SetCache(encode_out.data(), batch_size, length, cache_encode_k, cache_encode_v);
	//GetPosWeight(position_weight, decode_length+1, hidden_num);
	BuildBias(batch_size, length, mask.data(), encdec_bias.data());

	vector<int> tag(batch_size,0);

	for (int i = 0; i < decode_length; i++)
	{
		//init embedding_word
	        if(i==0) 
		{
		EmbeddingInit(language_id, embedding_word);
		}
		else
		{
		EmbeddingLookup(word, embedding_word);
		} 
	//out<<"embedding_word: "<<embedding_word[0]<<" "<<embedding_word[1023]<<endl;
		//position encode cancel
		//PositionEncode(i, embedding_word, position_weight);
		word = Decode(embedding_word, batch_size, length, cache_encode_k, cache_encode_v, cache_k, cache_v, self_bias, encdec_bias);
		//collect result

		int sum = 0;
	        for(int i=0; i<batch_size; i++)
		{
			if(word[i]==2) tag[i]=1;
			sum += tag[i];
			result[i].push_back(word[i]);
		}
		if(sum == batch_size) break;
        }
	// show result
	/*
	std::cout << "result: "<<endl;
	for (auto x : result)
	{
		cout<<"[";
		for(auto info:x)
		{
		cout<<info<<" ";
		}
		cout<<"]"<<endl;
	}
	*/
	return result;
}
