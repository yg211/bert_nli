# BERT-based NLI model

This project includes a natural language inference (NLI) model, developed
by fine-tuning BERT-base on the SNLI and MultiNLI datasets. 
The pre-trained model is provided, as well as an 
easy-to-use interface for using the trained model.
Code for training and testing the model is also provided.

Contact person: Yang Gao, yang.gao@rhul.ac.uk

https://sites.google.com/site/yanggaoalex/home

Don't hesitate to send me an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions


## Prerequisties
* Python3 
* Install all packages in requirement.txt.
```shell script
pip3 install -r requirements.txt
```
* Download the SNLI and MultiNLI data as well as the trained model with the commands below
```shell script
cd datasets/
python get_data.py
```
* (Optional) If you would like to run mixed-precision training 
(which can save the GPU memory consumption by roughly 50%), 
install the nvidia-apex package inside this project
by running the following commands (copied from the official 
instructions at [here](https://github.com/NVIDIA/apex)):
```shell script
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
* All code is tested on a desktop with nVidia RTX 2080,
running Python 3.7 on Ubuntu 18.04 LTS.

## Use the trained NLI model 
* The pretrained model is at *output/sample_model.state_dict* 
* An example usage is provided at *example.py*:
```python
from bert_nli import BertNLIModel

model = BertNLIModel('output/sample_model.state_dict')
sent_pairs = [('The lecturer committed plagiarism.','He was promoted.'),
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
```        
The output of the above example is:
```text
['contradiction', 'entail', 'contradiction', 'neutral', 'contradiction', 'entail', 'neutral']
[[-2.7058125e-02 -5.6325264e+00 -3.7672405e+00]
 [-4.1472511e+00 -4.2099822e-01 -1.1153491e+00]
 [-1.2617111e-03 -8.5965118e+00 -6.8346128e+00]
 [-3.7278435e+00 -6.5714045e+00 -2.5773764e-02]
 [-1.8420219e-03 -7.9999475e+00 -6.4988966e+00]
 [-7.6573701e+00 -3.1454802e-02 -3.4902668e+00]
 [-6.2880807e+00 -3.7793152e+00 -2.5006771e-02]]
```

## Train the NLI model
* Run *train_text.py* with the default settings (batch size 8, epoch num
1, using gpu, using mixed-precision training, 90% of the training set
is used as train and 10% used as dev):
```shell script
python train_test.py
```
On the machine with one RTX 2080 GPU card, 
it takes around three hours to finish the training.
* The trained model (that has the best performance on the dev set)
will be saved to directory *output/*.

## Test the performance of the trained
* To test the performance of a trained model, run the command below:
```shell script
python test_trained_model.py
```
You can specify the model you want to test when you initialise the
BertNLIModel (line 46 in test_trained_model.py). Performance of the 
current sample model is summarised below: 

|  | Contradiction | Entail | Neutral |
|-------|-----------|--------|----|
| Precision | 0.8791 | 0.8955 | 0.8080 |
| Recall | 0.8755 | 0.8658 | 0.8403 |
| F1 | 0.8773 | 0.8804 | 0.8239 |

The overall accuracy is 0.8608.
## License
Apache License Version 2.0




