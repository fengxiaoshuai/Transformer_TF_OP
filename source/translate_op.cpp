#include <iostream>
#include <memory>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "encode.h"
#include "decode.h"
#include "vector"


using namespace tensorflow;
using shape_inference::ShapeHandle;
using namespace std;
using namespace NMT;

const int LAYERS = 6;
const int HEAD = 16;
const int HIDDEN = 1024;
const int VOCAB_NUM = 32768;
const int FILTER = 4096;
const int BATCH = 2;
const int LENGTH = 8;
const int MAXPOS = 20;


REGISTER_OP("Translate")

.Input("input_word: int32")
.Input("target_language: int32")
.Input("mask: int32")
.Input("decode_length: int32")
//bottom
.Input("weight_embedding: float")
.Input("language_embedding: float")
//layer_0
.Input("weight0_0: float")
.Input("weight0_1: float")
.Input("weight0_2: float")
.Input("weight0_3: float")
.Input("weight0_4: float")
.Input("weight0_5: float")
.Input("weight0_6: float")
.Input("weight0_7: float")
.Input("weight0_8: float")
.Input("weight0_9: float")
.Input("weight0_10: float")
.Input("weight0_11: float")
.Input("weight0_12: float")
.Input("weight0_13: float")
//layer_1
.Input("weight1_0: float")
.Input("weight1_1: float")
.Input("weight1_2: float")
.Input("weight1_3: float")
.Input("weight1_4: float")
.Input("weight1_5: float")
.Input("weight1_6: float")
.Input("weight1_7: float")
.Input("weight1_8: float")
.Input("weight1_9: float")
.Input("weight1_10: float")
.Input("weight1_11: float")
.Input("weight1_12: float")
.Input("weight1_13: float")
//layer_2
.Input("weight2_0: float")
.Input("weight2_1: float")
.Input("weight2_2: float")
.Input("weight2_3: float")
.Input("weight2_4: float")
.Input("weight2_5: float")
.Input("weight2_6: float")
.Input("weight2_7: float")
.Input("weight2_8: float")
.Input("weight2_9: float")
.Input("weight2_10: float")
.Input("weight2_11: float")
.Input("weight2_12: float")
.Input("weight2_13: float")
//layer_3
.Input("weight3_0: float")
.Input("weight3_1: float")
.Input("weight3_2: float")
.Input("weight3_3: float")
.Input("weight3_4: float")
.Input("weight3_5: float")
.Input("weight3_6: float")
.Input("weight3_7: float")
.Input("weight3_8: float")
.Input("weight3_9: float")
.Input("weight3_10: float")
.Input("weight3_11: float")
.Input("weight3_12: float")
.Input("weight3_13: float")
//layer_4
.Input("weight4_0: float")
.Input("weight4_1: float")
.Input("weight4_2: float")
.Input("weight4_3: float")
.Input("weight4_4: float")
.Input("weight4_5: float")
.Input("weight4_6: float")
.Input("weight4_7: float")
.Input("weight4_8: float")
.Input("weight4_9: float")
.Input("weight4_10: float")
.Input("weight4_11: float")
.Input("weight4_12: float")
.Input("weight4_13: float")
//layer_5
.Input("weight5_0: float")
.Input("weight5_1: float")
.Input("weight5_2: float")
.Input("weight5_3: float")
.Input("weight5_4: float")
.Input("weight5_5: float")
.Input("weight5_6: float")
.Input("weight5_7: float")
.Input("weight5_8: float")
.Input("weight5_9: float")
.Input("weight5_10: float")
.Input("weight5_11: float")
.Input("weight5_12: float")
.Input("weight5_13: float")
//top
.Input("scale: float")
.Input("bias: float")
/*****************decode*************************/
//bottom
.Input("de_weight_embedding: float")
.Input("de_language_embedding: float")
//layer_de_0
.Input("de_weight0_0: float")
.Input("de_weight0_1: float")
.Input("de_weight0_2: float")
.Input("de_weight0_3: float")
.Input("de_weight0_4: float")
.Input("de_weight0_5: float")
.Input("de_weight0_6: float")
.Input("de_weight0_7: float")
.Input("de_weight0_8: float")
.Input("de_weight0_9: float")
.Input("de_weight0_10: float")
.Input("de_weight0_11: float")
.Input("de_weight0_12: float")
.Input("de_weight0_13: float")
.Input("de_weight0_14: float")
.Input("de_weight0_15: float")
.Input("de_weight0_16: float")
.Input("de_weight0_17: float")
.Input("de_weight0_18: float")
.Input("de_weight0_19: float")
//layer_de_1
.Input("de_weight1_0: float")
.Input("de_weight1_1: float")
.Input("de_weight1_2: float")
.Input("de_weight1_3: float")
.Input("de_weight1_4: float")
.Input("de_weight1_5: float")
.Input("de_weight1_6: float")
.Input("de_weight1_7: float")
.Input("de_weight1_8: float")
.Input("de_weight1_9: float")
.Input("de_weight1_10: float")
.Input("de_weight1_11: float")
.Input("de_weight1_12: float")
.Input("de_weight1_13: float")
.Input("de_weight1_14: float")
.Input("de_weight1_15: float")
.Input("de_weight1_16: float")
.Input("de_weight1_17: float")
.Input("de_weight1_18: float")
.Input("de_weight1_19: float")
//layer_de_2
.Input("de_weight2_0: float")
.Input("de_weight2_1: float")
.Input("de_weight2_2: float")
.Input("de_weight2_3: float")
.Input("de_weight2_4: float")
.Input("de_weight2_5: float")
.Input("de_weight2_6: float")
.Input("de_weight2_7: float")
.Input("de_weight2_8: float")
.Input("de_weight2_9: float")
.Input("de_weight2_10: float")
.Input("de_weight2_11: float")
.Input("de_weight2_12: float")
.Input("de_weight2_13: float")
.Input("de_weight2_14: float")
.Input("de_weight2_15: float")
.Input("de_weight2_16: float")
.Input("de_weight2_17: float")
.Input("de_weight2_18: float")
.Input("de_weight2_19: float")
//layer_de_3
.Input("de_weight3_0: float")
.Input("de_weight3_1: float")
.Input("de_weight3_2: float")
.Input("de_weight3_3: float")
.Input("de_weight3_4: float")
.Input("de_weight3_5: float")
.Input("de_weight3_6: float")
.Input("de_weight3_7: float")
.Input("de_weight3_8: float")
.Input("de_weight3_9: float")
.Input("de_weight3_10: float")
.Input("de_weight3_11: float")
.Input("de_weight3_12: float")
.Input("de_weight3_13: float")
.Input("de_weight3_14: float")
.Input("de_weight3_15: float")
.Input("de_weight3_16: float")
.Input("de_weight3_17: float")
.Input("de_weight3_18: float")
.Input("de_weight3_19: float")
//layer_de_4
.Input("de_weight4_0: float")
.Input("de_weight4_1: float")
.Input("de_weight4_2: float")
.Input("de_weight4_3: float")
.Input("de_weight4_4: float")
.Input("de_weight4_5: float")
.Input("de_weight4_6: float")
.Input("de_weight4_7: float")
.Input("de_weight4_8: float")
.Input("de_weight4_9: float")
.Input("de_weight4_10: float")
.Input("de_weight4_11: float")
.Input("de_weight4_12: float")
.Input("de_weight4_13: float")
.Input("de_weight4_14: float")
.Input("de_weight4_15: float")
.Input("de_weight4_16: float")
.Input("de_weight4_17: float")
.Input("de_weight4_18: float")
.Input("de_weight4_19: float")
//layer_de_5
.Input("de_weight5_0: float")
.Input("de_weight5_1: float")
.Input("de_weight5_2: float")
.Input("de_weight5_3: float")
.Input("de_weight5_4: float")
.Input("de_weight5_5: float")
.Input("de_weight5_6: float")
.Input("de_weight5_7: float")
.Input("de_weight5_8: float")
.Input("de_weight5_9: float")
.Input("de_weight5_10: float")
.Input("de_weight5_11: float")
.Input("de_weight5_12: float")
.Input("de_weight5_13: float")
.Input("de_weight5_14: float")
.Input("de_weight5_15: float")
.Input("de_weight5_16: float")
.Input("de_weight5_17: float")
.Input("de_weight5_18: float")
.Input("de_weight5_19: float")
//top
.Input("de_scale: float")
.Input("de_bias: float")
.Input("de_logit: float")
.Output("output: int32");


class TranslateOp : public OpKernel
{
public:
    explicit TranslateOp(OpKernelConstruction* context) : OpKernel(context),en_weight(LAYERS),de_weight(LAYERS)
    {
      initialized = false;
      /*
      en_weight_embedding.resize(0);//(VOCAB_NUM*HIDDEN);
      en_language_embedding.resize(0);
      en_weight_scale.resize(0);//(HIDDEN*HIDDEN);
      en_weight_bias.resize(0);//(HIDDEN);
      de_weight_embedding.resize(0);
      de_language_embedding.resize(0);
      de_weight_scale.resize(0);
      de_weight_bias.resize(0);
      de_weight_logit.resize(0);
      */
    };
    void Compute(OpKernelContext* context) override
    {
        //get tensor
        auto time1 = chrono::steady_clock::now();
        const Tensor& input_tensor = context->input(0);
        int batch = input_tensor.dim_size(0);
        int length = input_tensor.dim_size(1);

        const Tensor& tensor_language = context->input(1);
	const int* language = tensor_language.flat<int32>().data();
        vector<int> v_language(language, language + batch);

        const Tensor& tensor_mask = context->input(2);
	const int* mask = tensor_mask.flat<int32>().data();
        vector<int> v_mask(mask, mask + batch*length);

        const Tensor& tensor_decode_length = context->input(3);
	const int* decode_length = tensor_decode_length.flat<int32>().data();

        //init weight
        if(!initialized)
        {
            int idx = 4;
	    //encode
            float* w_embedding = (float*)context->input(idx++).tensor_data().data();
            float* l_embedding = (float*)context->input(idx++).tensor_data().data();
            for (int i = 0; i < LAYERS; i++)
            {
                  float *self_scale = (float *)context->input(idx++).tensor_data().data(); 
                  en_weight[i].push_back(self_scale);
 
                  float *self_bias = (float *)context->input(idx++).tensor_data().data();
                  en_weight[i].push_back(self_bias);

                  float *self_q = (float *)context->input(idx++).tensor_data().data();
                  en_weight[i].push_back(self_q);

                  float *self_k = (float *)context->input(idx++).tensor_data().data();
                  en_weight[i].push_back(self_k);

                  float *self_v = (float *)context->input(idx++).tensor_data().data();
                  en_weight[i].push_back(self_v);

                  float *self_last = (float *)context->input(idx++).tensor_data().data();
                  en_weight[i].push_back(self_last);

                  // encdec attention
                  //ffn
                  float *ffn_scale = (float *)context->input(idx++).tensor_data().data();
                  en_weight[i].push_back(ffn_scale);
 
                  float *ffn_bias = (float *)context->input(idx++).tensor_data().data();
                  en_weight[i].push_back(ffn_bias);
 
                  float *ffn_first_weight = (float *)context->input(idx++).tensor_data().data();
                  en_weight[i].push_back(ffn_first_weight);
 
                  float *ffn_first_bias = (float *)context->input(idx++).tensor_data().data();
                  en_weight[i].push_back(ffn_first_bias);
 
                  float *ffn_second_weight = (float *)context->input(idx++).tensor_data().data();
                  en_weight[i].push_back(ffn_second_weight);
 
                  float *ffn_second_bias = (float *)context->input(idx++).tensor_data().data();
                  en_weight[i].push_back(ffn_second_bias);

                  //self
                  float *self_position_key = (float *)context->input(idx++).tensor_data().data();
                  en_weight[i].push_back(self_position_key);

                  float *self_position_value = (float *)context->input(idx++).tensor_data().data();
                  en_weight[i].push_back(self_position_value);
 
            }
            float* scale = (float*)context->input(idx++).tensor_data().data();
            float* bias = (float*)context->input(idx++).tensor_data().data();
	    // set weight
            en_weight_embedding = w_embedding;
            en_language_embedding = l_embedding;
            en_weight_scale = scale;
            en_weight_bias = bias;
            /**********************encode************************/
            float* de_w_embedding = (float*)context->input(idx++).tensor_data().data();
            float* de_l_embedding = (float*)context->input(idx++).tensor_data().data();
            for (int i = 0; i < LAYERS; i++)
            {
                  float *de_self_scale = (float *)context->input(idx++).tensor_data().data(); 
                  de_weight[i].push_back(de_self_scale);
 
                  float *de_self_bias = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_self_bias);

                  float *de_self_q = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_self_q);

                  float *de_self_k = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_self_k);

                  float *de_self_v = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_self_v);

                  float *de_self_last = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_self_last);

                  // de_encdec attention
                  float *de_encdec_scale = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_encdec_scale);
 
                  float *de_encdec_bias = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_encdec_bias);
 
                  float *de_encdec_q = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_encdec_q);
 
                  float *de_encdec_k = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_encdec_k);
 
                  float *de_encdec_v = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_encdec_v);
 
                  float *de_encdec_last = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_encdec_last);
 
                  //de_ffn
                  float *de_ffn_scale = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_ffn_scale);
 
                  float *de_ffn_bias = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_ffn_bias);
 
                  float *de_ffn_first_de_weight = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_ffn_first_de_weight);
 
                  float *de_ffn_first_bias = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_ffn_first_bias);
 
                  float *de_ffn_second_de_weight = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_ffn_second_de_weight);
 
                  float *de_ffn_second_bias = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_ffn_second_bias);

                  //de_self
                  float *de_self_position_key = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_self_position_key);

                  float *de_self_position_value = (float *)context->input(idx++).tensor_data().data();
                  de_weight[i].push_back(de_self_position_value);
 
            }

            float* de_scale = (float*)context->input(idx++).tensor_data().data();
            float* de_bias = (float*)context->input(idx++).tensor_data().data();
            float* de_logit = (float*)context->input(idx++).tensor_data().data();
	    // set weight
            de_weight_embedding = de_w_embedding;
            de_language_embedding = de_l_embedding ;
            de_weight_scale = de_scale;
            de_weight_bias = de_bias;
            de_weight_logit = de_logit;
            initialized = true;
            encoder = make_shared<Encoder>(HEAD,
                                           HIDDEN,
                                           LAYERS,
					   VOCAB_NUM,
                                           FILTER,
                                           en_weight,
                                           en_weight_embedding,
         			           en_language_embedding,
                                           en_weight_scale,
                                           en_weight_bias);
            decoder = make_shared<Decoder>(HEAD,
                                           HIDDEN,
                                           LAYERS,
					   VOCAB_NUM,
                                           FILTER,
                                           de_weight,
                                           de_weight_embedding,
         			           de_language_embedding,
                                           de_weight_scale,
                                           de_weight_bias,
                                           de_weight_logit);
       }; 
	//input
	const int* input_data = input_tensor.flat<int32>().data();
        vector<int> input(input_data, input_data + batch*length);

        auto time2  = chrono::steady_clock::now();
        vector<float> encode_out = encoder->Encode(input, batch, length, v_mask, v_language);
        vector<vector<size_t>> result = decoder->Translate(encode_out, batch, length, v_language, v_mask, *decode_length);

	//out
	int out_length = result[0].size();
	Tensor* output_tensor = NULL;
	TensorShape out_shape({batch, out_length});// = input_tensor.shape();
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_tensor));
        auto output = output_tensor->flat<int32>();
	//set output
	for(int i=0; i<batch; i++)
	{
		for(int j=0; j<out_length; j++)
		{
			int loc = i * (out_length) + j;
			output(loc) = result[i][j];
		}
	
	}
        auto time3  = chrono::steady_clock::now();
	cout<<"*******translate******: "<<(chrono::duration_cast<chrono::duration<double>>(time3-time2)).count()<<endl;


    }
private:
  bool initialized;
  //encode
  vector<vector<float*>> en_weight;
  float* en_weight_embedding;
  float* en_language_embedding;
  float* en_weight_scale;
  float* en_weight_bias;
  shared_ptr<Encoder> encoder;
  //decode
  vector<vector<float*>> de_weight;
  float* de_weight_embedding;
  float* de_language_embedding;
  float* de_weight_scale;
  float* de_weight_bias;
  float* de_weight_logit;
  shared_ptr<Decoder> decoder;
};

REGISTER_KERNEL_BUILDER(Name("Translate").Device(DEVICE_CPU), TranslateOp);
