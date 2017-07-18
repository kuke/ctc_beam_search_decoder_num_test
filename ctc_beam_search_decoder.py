import numpy as np
from math import log

def ctc_beam_search_decoder(probs_seq,
                            beam_size,
                            vocabulary,
                            blank_id,
                            cutoff_prob=1.0,
                            ext_scoring_func=None,
                            nproc=False):
    """Beam search decoder for CTC-trained network. It utilizes beam search
    to approximately select top best decoding labels and returning results
    in the descending order. The implementation is based on Prefix
    Beam Search (https://arxiv.org/abs/1408.2873), and the unclear part is
    redesigned. Two important modifications: 1) in the iterative computation
    of probabilities, the assignment operation is changed to accumulation for
    one prefix may comes from different paths; 2) the if condition "if l^+ not
    in A_prev then" after probabilities' computation is deprecated for it is
    hard to understand and seems unnecessary.
    :param probs_seq: 2-D list of probability distributions over each time
                      step, with each element being a list of normalized
                      probabilities over vocabulary and blank.
    :type probs_seq: 2-D list
    :param beam_size: Width for beam search.
    :type beam_size: int
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :param blank_id: ID of blank.
    :type blank_id: int
    :param cutoff_prob: Cutoff probability in pruning,
                        default 1.0, no pruning.
    :type cutoff_prob: float
    :param ext_scoring_func: External scoring function for
                            partially decoded sentence, e.g. word count
                            or language model.
    :type external_scoring_func: callable
    :param nproc: Whether the decoder used in multiprocesses.
    :type nproc: bool
    :return: List of tuples of log probability and sentence as decoding
             results, in descending order of the probability.
    :rtype: list
    """
    # dimension check
    for prob_list in probs_seq:
        if not len(prob_list) == len(vocabulary) + 1:
            raise ValueError("The shape of prob_seq does not match with the "
                             "shape of the vocabulary.")

    # blank_id check
    if not blank_id < len(probs_seq[0]):
        raise ValueError("blank_id shouldn't be greater than probs dimension")

    # If the decoder called in the multiprocesses, then use the global scorer
    # instantiated in ctc_beam_search_decoder_batch().
    if nproc is True:
        global ext_nproc_scorer
        ext_scoring_func = ext_nproc_scorer

    ## initialize
    # prefix_set_prev: the set containing selected prefixes
    # probs_b_prev: prefixes' probability ending with blank in previous step
    # probs_nb_prev: prefixes' probability ending with non-blank in previous step
    prefix_set_prev = {'\t': 1.0}
    probs_b_prev, probs_nb_prev = {'\t': 1.0}, {'\t': 0.0}

    ## extend prefix in loop
    for time_step in xrange(len(probs_seq)):
        # prefix_set_next: the set containing candidate prefixes
        # probs_b_cur: prefixes' probability ending with blank in current step
        # probs_nb_cur: prefixes' probability ending with non-blank in current step
        prefix_set_next, probs_b_cur, probs_nb_cur = {}, {}, {}

        prob_idx = list(enumerate(probs_seq[time_step]))
        cutoff_len = len(prob_idx)
        #If pruning is enabled
        if cutoff_prob < 1.0:
            prob_idx = sorted(prob_idx, key=lambda asd: asd[1], reverse=True)
            cutoff_len, cum_prob = 0, 0.0
            for i in xrange(len(prob_idx)):
                cum_prob += prob_idx[i][1]
                cutoff_len += 1
                if cum_prob >= cutoff_prob:
                    break
            prob_idx = prob_idx[0:cutoff_len]

        for l in prefix_set_prev:
            if not prefix_set_next.has_key(l):
                probs_b_cur[l], probs_nb_cur[l] = 0.0, 0.0

            # extend prefix by travering prob_idx
            for index in xrange(cutoff_len):
                c, prob_c = prob_idx[index][0], prob_idx[index][1]

                if c == blank_id:
                    probs_b_cur[l] += prob_c * (
                        probs_b_prev[l] + probs_nb_prev[l])
                else:
                    last_char = l[-1]
                    new_char = vocabulary[c]
                    l_plus = l + new_char
                    if not prefix_set_next.has_key(l_plus):
                        probs_b_cur[l_plus], probs_nb_cur[l_plus] = 0.0, 0.0

                    if new_char == last_char:
                        probs_nb_cur[l_plus] += prob_c * probs_b_prev[l]
                        probs_nb_cur[l] += prob_c * probs_nb_prev[l]
                    elif new_char == ' ':
                        if (ext_scoring_func is None) or (len(l) == 1):
                            score = 1.0
                        else:
                            prefix = l[1:]
                            score = ext_scoring_func(prefix)
                        probs_nb_cur[l_plus] += score * prob_c * (
                            probs_b_prev[l] + probs_nb_prev[l])
                    else:
                        probs_nb_cur[l_plus] += prob_c * (
                            probs_b_prev[l] + probs_nb_prev[l])
                    # add l_plus into prefix_set_next
                    prefix_set_next[l_plus] = probs_nb_cur[
                        l_plus] + probs_b_cur[l_plus]
            # add l into prefix_set_next
            prefix_set_next[l] = probs_b_cur[l] + probs_nb_cur[l]
        # update probs
        probs_b_prev, probs_nb_prev = probs_b_cur, probs_nb_cur

        ## store top beam_size prefixes
        prefix_set_prev = sorted(
            prefix_set_next.iteritems(), key=lambda asd: asd[1], reverse=True)
        if beam_size < len(prefix_set_prev):
            prefix_set_prev = prefix_set_prev[:beam_size]
        prefix_set_prev = dict(prefix_set_prev)

    beam_result = []
    for seq, prob in prefix_set_prev.items():
        if prob > 0.0 and len(seq) > 1:
            result = seq[1:]
            # score last word by external scorer
            if (ext_scoring_func is not None) and (result[-1] != ' '):
                prob = prob * ext_scoring_func(result)
            if prob > 0.0:
                log_prob = log(prob)
                beam_result.append((log_prob, result))

    ## output top beam_size decoding results
    beam_result = sorted(beam_result, key=lambda asd: asd[0], reverse=True)
    return beam_result

def ctc_beam_search_decoder_log(probs_seq,
                            beam_size,
                            vocabulary,
                            blank_id,
                            cutoff_prob=1.0,
                            ext_scoring_func=None,
                            nproc=False):
    '''
    Beam search decoder computing in log probability, others are kept consistent 
    with ctc_beam_search_decoder().

    :param probs_seq: 2-D list with length num_time_steps, each element
                      is a list of normalized probabilities over vocabulary
                      and blank for one time step.
    :type probs_seq: 2-D list
    :param beam_size: Width for beam search.
    :type beam_size: int
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :param ext_scoring_func: External defined scoring function for
                            partially decoded sentence, e.g. word count
                            and language model.
    :type external_scoring_function: function
    :param blank_id: id of blank, default 0.
    :type blank_id: int
    :param nproc: Whether the decoder used in multiprocesses.
    :type nproc: bool
    :return: Decoding log probability and result string.
    :rtype: list

    '''
    # dimension check
    for prob_list in probs_seq:
        if not len(prob_list) == len(vocabulary) + 1:
            raise ValueError("probs dimension mismatched with vocabulary")
    num_time_steps = len(probs_seq)

    # blank_id check
    probs_dim = len(probs_seq[0])
    if not blank_id < probs_dim:
        raise ValueError("blank_id shouldn't be greater than probs dimension")

    # If the decoder called in the multiprocesses, then use the global scorer
    # instantiated in ctc_beam_search_decoder_nproc().
    if nproc is True:
        global ext_nproc_scorer
        ext_scoring_func = ext_nproc_scorer

    # sum of probabilitis in log format
    def log_sum_exp(x, y):
        if x == FLT64_MIN:
            return y
        if y == FLT64_MIN:
            return x
        xmax = max(x, y)
        z = np.log(np.exp(x-xmax) + np.exp(y-xmax)) + xmax
        return z 

    ## initialize
    FLT64_MIN = np.float64('-inf')
    # the set containing selected prefixes
    prefix_set_prev = {'\t': 0.0}
    log_probs_b_prev, log_probs_nb_prev = {'\t': 0.0}, {'\t': FLT64_MIN}

    ## extend prefix in loop
    for time_step in range(num_time_steps):
        # the set containing candidate prefixes
        prefix_set_next = {}
        log_probs_b_cur, log_probs_nb_cur = {}, {}
        
        prob_idx = list(enumerate(probs_seq[time_step]))
        cutoff_len = len(prob_idx)
        #If pruning is enabled
        if cutoff_prob < 1.0:
            prob_idx = sorted(prob_idx, key=lambda asd: asd[1], reverse=True)
            cutoff_len, cum_prob = 0, 0.0
            for i in xrange(len(prob_idx)):
                cum_prob += prob_idx[i][1]
                cutoff_len += 1
                if cum_prob >= cutoff_prob:
                    break
            prob_idx = prob_idx[0:cutoff_len]
        # convert prob into log-prob
        log_prob_idx = [(prob_idx[i][0], log(prob_idx[i][1])) for i in xrange(cutoff_len)]

        for l in prefix_set_prev:
            if not prefix_set_next.has_key(l):
                log_probs_b_cur[l], log_probs_nb_cur[l] = FLT64_MIN, FLT64_MIN

            # extend prefix by travering vocabulary
            # for c in range(0, probs_dim):
            for index in xrange(cutoff_len):
                c, log_prob_c = log_prob_idx[index][0], log_prob_idx[index][1]
                if c == blank_id:
                    log_probs_prev = log_sum_exp(log_probs_b_prev[l], log_probs_nb_prev[l])
                    log_probs_b_cur[l] = log_sum_exp(log_probs_b_cur[l], 
                                                 log_prob_c+log_probs_prev)
                else:
                    last_char = l[-1]
                    new_char = vocabulary[c]
                    l_plus = l + new_char
                    if not prefix_set_next.has_key(l_plus):
                        log_probs_b_cur[l_plus], log_probs_nb_cur[l_plus] = FLT64_MIN, FLT64_MIN

                    if new_char == last_char:
                        log_probs_nb_cur[l_plus] = log_sum_exp(log_probs_nb_cur[l_plus], log_prob_c+log_probs_b_prev[l])
                        log_probs_nb_cur[l] = log_sum_exp(log_probs_nb_cur[l], log_prob_c+log_probs_nb_prev[l])
                    elif new_char == ' ':
                        if (ext_scoring_func is None) or (len(l) == 1):
                            score = 0.0
                        else:
                            prefix = l[1:]
                            score = ext_scoring_func(prefix, True)
                        log_probs_prev = log_sum_exp(log_probs_b_prev[l], log_probs_nb_prev[l])
                        log_probs_nb_cur[l_plus] = log_sum_exp(log_probs_nb_cur[l_plus], score+log_prob_c+log_probs_prev )
                    else:
                        log_probs_prev = log_sum_exp(log_probs_b_prev[l], log_probs_nb_prev[l])
                        log_probs_nb_cur[l_plus] = log_sum_exp(log_probs_nb_cur[l_plus], log_prob_c+log_probs_prev)
       
                    # add l_plus into prefix_set_next
                    prefix_set_next[l_plus] = log_sum_exp(log_probs_nb_cur[
                        l_plus], log_probs_b_cur[l_plus])
            # add l into prefix_set_next
            prefix_set_next[l] = log_sum_exp(log_probs_b_cur[l], log_probs_nb_cur[l])
        # update probs
        log_probs_b_prev, log_probs_nb_prev = log_probs_b_cur, log_probs_nb_cur

        ## store top beam_size prefixes
        prefix_set_prev = sorted(
            prefix_set_next.iteritems(), key=lambda asd: asd[1], reverse=True)
        if beam_size < len(prefix_set_prev):
            prefix_set_prev = prefix_set_prev[:beam_size]
        prefix_set_prev = dict(prefix_set_prev)

    beam_result = []
    for (seq, log_prob) in prefix_set_prev.items():
        if log_prob > FLT64_MIN:
            result = seq[1:]
            if (ext_scoring_func is not None) and (result[-1] != ' '):
                log_prob = log_prob + ext_scoring_func(result, True)
            if log_prob > FLT64_MIN:
                beam_result.append([log_prob, result])

    ## output top beam_size decoding results
    beam_result = sorted(beam_result, key=lambda asd: asd[0], reverse=True)
    return beam_result

