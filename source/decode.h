
#ifndef _DECODE_
#define _DECODE_

#include <vector>
#include <string>
using namespace std;

namespace NMT
{
	class Decoder
	{
	private:
		size_t head_num;
		size_t hidden_num;
		size_t layer_num;
		size_t vocabe_size;
		size_t filter_size;
		vector<vector<float*>> weight;
		float* weight_embedding;
		float* weight_language;
		float* weight_scale;
		float* weight_bias;
		float* logit_weight;


	public:
		vector<vector<size_t>> Translate(vector<float>& encode_out, const size_t& batch_size, const size_t& length, vector<int>& language_id, vector<int>& mask, int decode_length);
		void GetPosWeight(vector<float>& position_weight, size_t decode_length, size_t hidden_num);
                void BuildBias(const size_t& batch_size, const size_t& length,  int* mask, float* bias);
		void AddBias(float* input, const float* bias, const size_t& batch_size, const size_t& length);
		void EmbeddingLookup(vector<size_t>& input, vector<float>& embedding_word);
	        void EmbeddingInit(vector<int>& language_id, vector<float>& embedding_word);
		void PositionEncode(const size_t& time, vector<float>& embedding_word, vector<float>& weight_position);
		void LayerPostprocess(vector<float>& layer_input, const vector<float>& temp);
		void BatchSoftmax(float* input_qk, int k, int head_num, const size_t& batch_size, const size_t& length);
		void GenSoftmax(float* input, int num);
		void ToLogits(float* input, const size_t& batch_size, float* weight, float* output);
		vector<size_t> GetMax(vector<float>& logit, const size_t& batch_size);
                void GetPositionX(const float* position_embedding, const size_t max_length, const size_t& length, vector<float>& position_x);
                void MulPositionKey(const size_t& batch_size, const size_t& length, float* input, float* position_key, float* out);
                void MulPositionValue(const size_t& batch_size, const size_t& length, float* input, float* position_val, float* out);
		Decoder(const size_t& head_num,
			const size_t& hidden_num,
			const size_t& layer_num,
			const size_t& vocabe_size,
			const size_t& filter_size,
			vector<vector<float*>>& weight,
			float* weight_embedding,
			float* weight_language,
			float* weight_scale,
			float* weight_bias,
			float* logit_weight);
		void SetCache(const float* encode_out, 
                        const size_t& batch_size,
                        const size_t& length,
                        vector<vector<float>>& cache_out_k,
                        vector<vector<float>>& cache_out_v);
		void LayerPreprocess(vector<float>& layer_input,
                        const size_t& batch_size, 
                        const size_t& length,
                        const float* scale,
                        const float* bias);
		void FeedForward(const vector<float>& input,
                        vector<float>& output,
			const size_t& batch_size,
			const size_t& length,
                        int filter,
                        const float* weight,
                        float* bias,
                        string activation);
		vector<size_t> Decode(vector<float>& embedding_word,
                        const size_t& batch_size,
                        const size_t& length, 
                        vector<vector<float>>& encode_out_k,
                        vector<vector<float>>& encode_out_v,
                        vector<vector<float>>& cache_k, 
                        vector<vector<float>>& cache_v,
			vector<float>& self_bias,
			vector<float>& encdec_bias);
		void SelfAttention(float* input, 
			const size_t& batch_size,
			const size_t& length,
			const float* q_weight,
			const float* k_weight,
			const float* v_weight,
			const float* key_weight,
			const float* value_weight,
			const float* weight,
			float* output,
			vector<float>& k_value,
			vector<float>& v_value,
			const float* bias);
		void EncdecAttention(float* input, 
			const size_t& batch_size,
			const size_t& length,
			const float* q_weight,
			const float* weight,
			float* output,
			vector<float>& k_value,
			vector<float>& v_value,
			const float* bias);
	};
}

#endif // !_DECODE


