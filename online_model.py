import tensorflow as tf
import numpy as np
import time

def get_weight(checkpoint_dir):

    #'input all weight
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph('model.meta')
    sess = tf.Session(graph=graph)
    saver.restore(sess, checkpoint_dir+'/model')
    
    
    #graph = tf.Graph()
    #with graph.as_default():
    #    graph_def = tf.GraphDef()
    #    with tf.gfile.FastGFile('/data/share/liyan/project/Sync/model/model_1.pb', 'rb') as f:
    #        graph_def.ParseFromString(f.read())
    #    tf.import_graph_def(graph_def, name='')
    #sess = tf.Session(graph=graph)
    
    
    ##########################encode weight################################
    w_embedding = sess.graph.get_tensor_by_name('transformer/symbol_modality_32768_1024/shared/weights_0/read:0')#table
    w_embedding = sess.run(w_embedding)
    
    l_embedding = sess.graph.get_tensor_by_name('transformer/body/encoder_target_lang_embedding/kernel/read:0')
    en_l_embedding = sess.run(l_embedding)
    
    
    scale = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_prepostprocess/layer_norm/layer_norm_scale/read:0')
    en_scale = sess.run(scale)
    
    
    bias = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_prepostprocess/layer_norm/layer_norm_bias/read:0')
    en_bias = sess.run(bias)
    
    
    logit = sess.graph.get_tensor_by_name('transformer/symbol_modality_32768_1024/softmax/weights_0/read:0')
    en_logit = sess.run(logit)
    
    en_weight=[]
    # layer_{}
    for i in range(6):
    	##self attention
    	self_scale = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_{}/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale/read:0'.format(i))
    	self_scale = sess.run(self_scale)
    
    	self_bias = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_{}/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias/read:0'.format(i))
    	self_bias = sess.run(self_bias)
    
    	self_q = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_{}/self_attention/multihead_attention/q/kernel/read:0'.format(i))
    	self_q = sess.run(self_q)
    
    	self_k = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_{}/self_attention/multihead_attention/k/kernel/read:0'.format(i))
    	self_k = sess.run(self_k)
    
    	self_v = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_{}/self_attention/multihead_attention/v/kernel/read:0'.format(i))
    	self_v = sess.run(self_v)
    
    	self_last = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_{}/self_attention/multihead_attention/output_transform/kernel/read:0'.format(i))
    	self_last = sess.run(self_last)
    
    
    	## encdec attention
    	## ffn
    	ffn_scale = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_{}/ffn/layer_prepostprocess/layer_norm/layer_norm_scale/read:0'.format(i))
    	ffn_scale = sess.run(ffn_scale)
    
    	ffn_bias = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_{}/ffn/layer_prepostprocess/layer_norm/layer_norm_bias/read:0'.format(i))
    	ffn_bias = sess.run(ffn_bias)
    
    	first_weight = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_{}/ffn/conv1/kernel/read:0'.format(i))
    	first_weight = sess.run(first_weight)
    
    	first_bias = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_{}/ffn/conv1/bias/read:0'.format(i))
    	first_bias = sess.run(first_bias)
    
    	second_weight = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_{}/ffn/conv2/kernel/read:0'.format(i))
    	second_weight = sess.run(second_weight)
    
    	second_bias = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_{}/ffn/conv2/bias/read:0'.format(i))
    	second_bias = sess.run(second_bias)
    
    	self_position_key = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_{}/self_attention/multihead_attention/dot_product_attention_relative/relative_positions_keys/embeddings/read:0'.format(i))
    	self_position_key = sess.run(self_position_key)
    
    	self_position_value = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_{}/self_attention/multihead_attention/dot_product_attention_relative/relative_positions_values/embeddings/read:0'.format(i))
    	self_position_value = sess.run(self_position_value)
    
    	tem = [self_scale,self_bias,self_q,self_k,self_v,self_last,ffn_scale,ffn_bias,first_weight,first_bias,second_weight,second_bias, self_position_key,self_position_value]
    	
    	en_weight.append(tem)
    
    
    ###################decode weight######################
    l_embedding = sess.graph.get_tensor_by_name('transformer/body/decoder_target_lang_embedding/kernel/read:0')
    de_l_embedding = sess.run(l_embedding)
    
    
    scale = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_prepostprocess/layer_norm/layer_norm_scale/read:0')
    de_scale = sess.run(scale)
    
    
    bias = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_prepostprocess/layer_norm/layer_norm_bias/read:0')
    de_bias = sess.run(bias)
    
    
    logit = sess.graph.get_tensor_by_name('transformer/symbol_modality_32768_1024/softmax/weights_0/read:0')
    de_logit = sess.run(logit)
    
    de_weight=[]
    # layer_{}
    for i in range(6):
    	##self attention
    	self_scale = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale/read:0'.format(i))
    	self_scale = sess.run(self_scale)
    
    	self_bias = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias/read:0'.format(i))
    	self_bias = sess.run(self_bias)
    
    	self_q = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/self_attention/multihead_attention/q/kernel/read:0'.format(i))
    	self_q = sess.run(self_q)
    
    	self_k = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/self_attention/multihead_attention/k/kernel/read:0'.format(i))
    	self_k = sess.run(self_k)
    
    	self_v = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/self_attention/multihead_attention/v/kernel/read:0'.format(i))
    	self_v = sess.run(self_v)
    
    	self_last = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/self_attention/multihead_attention/output_transform/kernel/read:0'.format(i))
    	self_last = sess.run(self_last)
    
    
    	## encdec attention
    	encdec_scale = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/encdec_attention/layer_prepostprocess/layer_norm/layer_norm_scale/read:0'.format(i))
    	encdec_scale = sess.run(encdec_scale)
    
    	encdec_bias = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/encdec_attention/layer_prepostprocess/layer_norm/layer_norm_bias/read:0'.format(i))
    	encdec_bias = sess.run(encdec_bias)
    	
    	encdec_q = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/encdec_attention/multihead_attention/q/kernel/read:0'.format(i))
    	encdec_q = sess.run(encdec_q)
    
    	encdec_k = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/encdec_attention/multihead_attention/k/kernel/read:0'.format(i))
    	encdec_k = sess.run(encdec_k)
    
    	encdec_v = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/encdec_attention/multihead_attention/v/kernel/read:0'.format(i))
    	encdec_v = sess.run(encdec_v)
    
    	encdec_last = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/encdec_attention/multihead_attention/output_transform/kernel/read:0'.format(i))
    	encdec_last = sess.run(encdec_last)
    
    	## ffn
    	ffn_scale = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/ffn/layer_prepostprocess/layer_norm/layer_norm_scale/read:0'.format(i))
    	ffn_scale = sess.run(ffn_scale)
    
    	ffn_bias = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/ffn/layer_prepostprocess/layer_norm/layer_norm_bias/read:0'.format(i))
    	ffn_bias = sess.run(ffn_bias)
    
    	first_weight = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/ffn/conv1/kernel/read:0'.format(i))
    	first_weight = sess.run(first_weight)
    
    	first_bias = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/ffn/conv1/bias/read:0'.format(i))
    	first_bias = sess.run(first_bias)
    
    	second_weight = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/ffn/conv2/kernel/read:0'.format(i))
    	second_weight = sess.run(second_weight)
    
    	second_bias = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/ffn/conv2/bias/read:0'.format(i))
    	second_bias = sess.run(second_bias)
    
    	self_position_key = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/self_attention/multihead_attention/dot_product_attention_relative/relative_positions_keys/embeddings/read:0'.format(i))
    	self_position_key = sess.run(self_position_key)
    
    	self_position_value = sess.graph.get_tensor_by_name('transformer/body/decoder/layer_{}/self_attention/multihead_attention/dot_product_attention_relative/relative_positions_values/embeddings/read:0'.format(i))
    	self_position_value = sess.run(self_position_value)
    
    	tem = [self_scale,self_bias,self_q,self_k,self_v,self_last,encdec_scale,encdec_bias,encdec_q,encdec_k,
    	encdec_v,encdec_last,ffn_scale,ffn_bias,first_weight,first_bias,second_weight,second_bias,
    	self_position_key,self_position_value]
    	
    	de_weight.append(tem)
    
    #last
    print('##############get_weight_end##############')
    sess.close()
    return w_embedding, en_l_embedding, en_weight, en_scale, en_bias, de_l_embedding, de_weight, de_scale, de_bias, de_logit


def build_model(model_dir):

    w_embedding, en_l_embedding, en_weight, en_scale, en_bias, de_l_embedding, de_weight, de_scale, de_bias, de_logit = get_weight(model_dir)

    input_word = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_word")
    target_language = tf.placeholder(dtype=tf.int32, name="target_language" )
    mask = tf.placeholder(dtype=tf.int32, name="mask")
    decode_length = tf.placeholder(dtype=tf.int32, name="decode_length")
    
    so_file = './lib/translate_op.so'
    my_tf = tf.load_op_library(so_file)

    output = my_tf.translate(
        input_word, target_language, mask, decode_length, 
        w_embedding, en_l_embedding,
        en_weight[0][0], en_weight[0][1], en_weight[0][2], en_weight[0][3], en_weight[0][4], en_weight[0][5], en_weight[0][6], en_weight[0][7], en_weight[0][8], en_weight[0][9], en_weight[0][10], en_weight[0][11], en_weight[0][12], en_weight[0][13],
        en_weight[1][0], en_weight[1][1], en_weight[1][2], en_weight[1][3], en_weight[1][4], en_weight[1][5], en_weight[1][6], en_weight[1][7], en_weight[1][8], en_weight[1][9], en_weight[1][10], en_weight[1][11], en_weight[1][12], en_weight[1][13],
        en_weight[2][0], en_weight[2][1], en_weight[2][2], en_weight[2][3], en_weight[2][4], en_weight[2][5], en_weight[2][6], en_weight[2][7], en_weight[2][8], en_weight[2][9], en_weight[2][10], en_weight[2][11], en_weight[2][12], en_weight[2][13],
        en_weight[3][0], en_weight[3][1], en_weight[3][2], en_weight[3][3], en_weight[3][4], en_weight[3][5], en_weight[3][6], en_weight[3][7], en_weight[3][8], en_weight[3][9], en_weight[3][10], en_weight[3][11], en_weight[3][12], en_weight[3][13],
        en_weight[4][0], en_weight[4][1], en_weight[4][2], en_weight[4][3], en_weight[4][4], en_weight[4][5], en_weight[4][6], en_weight[4][7], en_weight[4][8], en_weight[4][9], en_weight[4][10], en_weight[4][11], en_weight[4][12], en_weight[4][13],
        en_weight[5][0], en_weight[5][1], en_weight[5][2], en_weight[5][3], en_weight[5][4], en_weight[5][5], en_weight[5][6], en_weight[5][7], en_weight[5][8], en_weight[5][9], en_weight[5][10], en_weight[5][11], en_weight[5][12], en_weight[5][13],
        en_scale, en_bias,
    
    
        w_embedding, de_l_embedding,
        de_weight[0][0], de_weight[0][1], de_weight[0][2], de_weight[0][3], de_weight[0][4], de_weight[0][5], de_weight[0][6], de_weight[0][7], de_weight[0][8], de_weight[0][9], de_weight[0][10], de_weight[0][11], de_weight[0][12], de_weight[0][13], de_weight[0][14], de_weight[0][15], de_weight[0][16], de_weight[0][17],de_weight[0][18],de_weight[0][19],
        de_weight[1][0], de_weight[1][1], de_weight[1][2], de_weight[1][3], de_weight[1][4], de_weight[1][5], de_weight[1][6], de_weight[1][7], de_weight[1][8], de_weight[1][9], de_weight[1][10], de_weight[1][11], de_weight[1][12], de_weight[1][13], de_weight[1][14], de_weight[1][15], de_weight[1][16], de_weight[1][17],de_weight[1][18],de_weight[1][19],
        de_weight[2][0], de_weight[2][1], de_weight[2][2], de_weight[2][3], de_weight[2][4], de_weight[2][5], de_weight[2][6], de_weight[2][7], de_weight[2][8], de_weight[2][9], de_weight[2][10], de_weight[2][11], de_weight[2][12], de_weight[2][13], de_weight[2][14], de_weight[2][15], de_weight[2][16], de_weight[2][17],de_weight[2][18],de_weight[2][19],
        de_weight[3][0], de_weight[3][1], de_weight[3][2], de_weight[3][3], de_weight[3][4], de_weight[3][5], de_weight[3][6], de_weight[3][7], de_weight[3][8], de_weight[3][9], de_weight[3][10], de_weight[3][11], de_weight[3][12], de_weight[3][13], de_weight[3][14], de_weight[3][15], de_weight[3][16], de_weight[3][17],de_weight[3][18],de_weight[3][19],
        de_weight[4][0], de_weight[4][1], de_weight[4][2], de_weight[4][3], de_weight[4][4], de_weight[4][5], de_weight[4][6], de_weight[4][7], de_weight[4][8], de_weight[4][9], de_weight[4][10], de_weight[4][11], de_weight[4][12], de_weight[4][13], de_weight[4][14], de_weight[4][15], de_weight[4][16], de_weight[4][17],de_weight[4][18],de_weight[4][19],
        de_weight[5][0], de_weight[5][1], de_weight[5][2], de_weight[5][3], de_weight[5][4], de_weight[5][5], de_weight[5][6], de_weight[5][7], de_weight[5][8], de_weight[5][9], de_weight[5][10], de_weight[5][11], de_weight[5][12], de_weight[5][13], de_weight[5][14], de_weight[5][15], de_weight[5][16], de_weight[5][17],de_weight[5][18],de_weight[5][19],
        de_scale, de_bias, de_logit)
    return output
