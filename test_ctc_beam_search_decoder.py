from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from ctc_beam_search_decoder import *
import time

vocab_list = ['\'', ' ']+[chr(i) for i in range(97, 101)]
#vocab_list = ['\'', ' ']+[chr(i) for i in range(97, 123)]

def generate_probs(num_time_steps, probs_dim):
    probs_mat = np.random.random(size=(num_time_steps, probs_dim))
    probs_mat = [probs_mat[index]/sum(probs_mat[index]) for index in range(num_time_steps)]
    return probs_mat

def test_beam_search_decoder():
    max_time_steps = 6
    probs_dim = len(vocab_list)+1
    beam_size = 20
    num_results_per_sample = 1
        
    input_prob_matrix_0 = np.asarray(generate_probs(max_time_steps, probs_dim), dtype=np.float32)
    print(input_prob_matrix_0)
    # Add arbitrary offset - this is fine
    input_log_prob_matrix_0 = np.log(input_prob_matrix_0) #+ 2.0

    # len max_time_steps array of batch_size x depth matrices
    inputs = ([
        input_log_prob_matrix_0[t, :][np.newaxis, :] for t in range(max_time_steps)]
     )

    inputs_t = [ops.convert_to_tensor(x) for x in inputs]
    inputs_t = array_ops.stack(inputs_t)
    
    # run CTC beam search decoder in tensorflow
    with tf.Session() as sess:
        decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs_t, 
                                                              [max_time_steps], 
                                                              beam_width=beam_size, 
                                                              top_paths=num_results_per_sample, 
                                                              merge_repeated=False)
        tf_decoded = sess.run(decoded)
	tf_log_probs = sess.run(log_probabilities)        
    

    # run original CTC beam search decoder     
    beam_result = ctc_beam_search_decoder(
			probs_seq=input_prob_matrix_0,
			beam_size=beam_size,
                        vocabulary=vocab_list,
                        blank_id=len(vocab_list),
                        cutoff_prob=1.0, 
			)
    
    # run log- CTC beam search decoder     
    beam_result_log = ctc_beam_search_decoder_log(
			probs_seq=input_prob_matrix_0,
			beam_size=beam_size,
                        vocabulary=vocab_list,
                        blank_id=len(vocab_list),       
                        cutoff_prob=1.0,            
			)
    # compare decoding result
    print("{tf-decoder log probs} \t {org-decoder log probs} \t{log-decoder log probs}:  {tf_decoder result}  {org_decoder result} {log-decoder result}")
    for index in range(num_results_per_sample):
        tf_result = ''.join([vocab_list[i] for i in tf_decoded[index].values])
        print(('%6f\t%f\t%f: ') % (tf_log_probs[0][index], beam_result[index][0], beam_result_log[index][0]), 
               tf_result,'\t', beam_result[index][1], '\t', beam_result_log[index][1])

if __name__ == '__main__':
    test_beam_search_decoder()
