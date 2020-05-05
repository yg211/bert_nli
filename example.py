from bert_nli import BertNLIModel

if __name__ == '__main__':
    bert_type = 'bert-base'
    model = BertNLIModel('output/{}.state_dict'.format(bert_type), bert_type=bert_type)

    sent_pairs = [('The lecturer committed plagiarism.','He was promoted.')]
    labels, probs = model(sent_pairs)
    print(labels)
    print(probs)
