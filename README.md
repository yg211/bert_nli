# BERT-based NLI model

This project includes a natural language inference (NLI) model, developed
by fine-tuning Transformers on the SNLI, MultiNLI and Hans datasets. 

**Highlighted Features**

* Models based on BERT-(base, large) and ALBERT-(base,large)
* Implemented using *PyTorch* (1.5.0)
* *Low memory requirements*: Using *mixed-precision* (nvidia apex) and [checkpoint](https://pytorch.org/docs/stable/checkpoint.html) to reduce the GPU memory consumption; training the bert/albert-large model only requires around **6GB** GPU memory (with batch size 8).
* *Easy inerface*: A user-friendly interface is provided to use the trained models
* *All source code*: All source code for training and testing the models is provided

Contact person: Yang Gao, yang.gao@rhul.ac.uk

https://sites.google.com/site/yanggaoalex/home

Don't hesitate to send me an e-mail or report an issue, if something is broken or if you have further questions.


## Use the trained NLI model 
* The pretrained models are downloaded to *output/* (after you run *get_data.py* in datasets/)
* An example is presented in *example.py*:
```python
from bert_nli import BertNLIModel

model = BertNLIModel('output/bert-base.state_dict')
sent_pairs = [('The lecturer committed plagiarism.','He was promoted.')]
label, _= model(sent_pairs)
print(label)
```        
The output of the above example is:
```text
['contradiction']
```

## How to set up
* Python3.7 
* Install all packages in requirement.txt.
```shell script
pip3 install -r requirements.txt
```
* Download the SNLI and MultiNLI data as well as the trained model with the commands below
```shell script
cd datasets/
python get_data.py
```
* (Optional) Our code supports the use of the [Hans](https://arxiv.org/abs/1902.01007) dataset to train the model, in order to prevent the BERT model from exploiting spurious features to make NLI predictions. To use the Hans dataset, download *heuristics_train_set.txt* and *heuristics_evaluation_set.txt* from [here](https://github.com/tommccoy1/hans), and put them to *datasets/Hans/*.
During training/test, add argument *--hans 1*.
* (Optional) To use mixed precision training (nvidia apex), run the code below:
```shell script
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
* All code is tested on a desktop with a single nVidia RTX 2080 card (8GB memory),
running Python 3.7 on Ubuntu 18.04 LTS.


## Train NLI models
* Run *train.py* and specify what Transformer model you would like to fine tune:
```shell script
python train.py --bert_type bert-large --check_point 1
```
Option "--check_point 1" means that we will use the checkpoint technique
during training. Without using it, the RTX2080 card (8GB memory) is not 
able to accommodate the bert-large model. But note that, by using
checkpoint, it usually takes longer time to train the model.

The trained model (that has the best performance on the dev set)
will be saved to directory *output/*.

## Test the performance of the trained models
* To test the performance of a trained model on the MNLI and SNLI
dev sets, run the command below:
```shell script
python test_trained_model.py --bert_type bert-large
```

## Performance of Trained Models
----
**BERT-base**

Accuracy: 0.8608.
|  | Contradiction | Entail | Neutral |
|-------|-----------|--------|----|
| Precision | 0.8791 | 0.8955 | 0.8080 |
| Recall | 0.8755 | 0.8658 | 0.8403 |
| F1 | 0.8773 | 0.8804 | 0.8239 |

----
**BERT-large**

Accuracy: 0.8739
|  | Contradiction | Entail | Neutral |
|-------|-----------|--------|----|
| Precision | 0.8992 | 0.8988 | 0.8233 |
| Recall | 0.8895 | 0.8802 | 0.8508 |
| F1 | 0.8944 | 0.8894 | 0.8369 |

----
**ALBERT-large**

Accuracy: 0.8743
|  | Contradiction | Entail | Neutral |
|-------|-----------|--------|----|
| Precision | 0.8907 | 0.8967 | 0.8335 |
| Recall | 0.9006 | 0.8812 | 0.8397 |
| F1 | 0.8957 | 0.8889 | 0.8366 |



## License
Apache License Version 2.0




