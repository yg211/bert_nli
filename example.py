from bert_nli import BertNLIModel

if __name__ == '__main__':
    model = BertNLIModel('output/sample_model.state_dict')

    sent_pairs = [('The lecturer committed plagiarism.','He was promoted.'),
                  ('The lecturer got a big funding award','He was promoted.'),
                  ('The lecturer won the Novel prize.','He was promoted.'),
                  ('The lecturer became a professor last June.','He was promoted.'),
                  ('A man inspects the uniform of a figure in some East Asian country.','The man is sleeping.'),
                  ('An older and younger man smiling.','Two men are smiling and laughing at the cats playing on the floor.'),
                  ('A black race car starts up in front of a crowd of people.','A man is driving down a lonely road.'),
                  ('A soccer game with multiple males playing.','Some men are playing a sport.'),
                  ('A smiling costumed woman is holding an umbrella.','A happy woman in a fairy costume holds an umbrella.')
                  ]
    labels, probs = model(sent_pairs)
    print(labels)
    print(probs)
