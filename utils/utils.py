

def get_pair_input(tokenizer, sent1, sent2, model_type):
    if 'roberta' in model_type:
        text = "<s> {} </s></s> {} </s>".format(sent1, sent2)
    else:
        text = "[CLS] {} [SEP] {} [SEP]".format(sent1,sent2)

    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.encode(text)[1:-1]
    assert len(tokenized_text) == len(indexed_tokens)

    if len(tokenized_text) > 500:
        return None, None
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = []
    sep_flag = False
    for i in range(len(indexed_tokens)):
        if 'roberta' in model_type and tokenized_text[i] == '</s>' and not sep_flag:
            segments_ids.append(0)
            sep_flag = True
        elif 'bert-' in model_type and tokenized_text[i] == '[SEP]' and not sep_flag:
            segments_ids.append(0)
            sep_flag = True
        elif sep_flag:
            segments_ids.append(1)
        else:
            segments_ids.append(0)
    return indexed_tokens, segments_ids


def build_batch(tokenizer, text_list, model_type):
    token_id_list = []
    segment_list = []
    attention_masks = []
    longest = -1

    for pair in text_list:
        sent1, sent2 = pair 
        ids, segs = get_pair_input(tokenizer,sent1,sent2,model_type)
        if ids is None or segs is None: continue
        token_id_list.append(ids)
        segment_list.append(segs)
        attention_masks.append([1]*len(ids))
        if len(ids) > longest: longest = len(ids)

    if len(token_id_list) == 0: return None, None, None

    # padding
    assert(len(token_id_list) == len(segment_list))
    for ii in range(len(token_id_list)):
        token_id_list[ii] += [0]*(longest-len(token_id_list[ii]))
        attention_masks[ii] += [0]*(longest-len(attention_masks[ii]))
        segment_list[ii] += [1]*(longest-len(segment_list[ii]))

    return token_id_list, segment_list, attention_masks


