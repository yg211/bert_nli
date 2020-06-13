import urllib.request
import zipfile
import os
import subprocess
import shutil
folder_path = os.path.dirname(os.path.realpath(__file__))
print('Beginning download of datasets')

dataset = 'AllNLI.zip'
server = "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/"

print("Download", dataset)
url = server+dataset
dataset_path = os.path.join(folder_path, dataset)
urllib.request.urlretrieve(url, dataset_path)

print("Extract", dataset)
with zipfile.ZipFile(dataset_path, "r") as zip_ref:
    zip_ref.extractall(folder_path)
os.remove(dataset_path)

model_url = 'https://drive.google.com/uc?id=1YlS3wuwlhzGRPMyJr8SE4IP89U0W7yVU'
print('Begin downloading trained model: bert-base')
subprocess.call(['gdown', model_url])
shutil.move('bert-base.state_dict','../output/')

model_url = 'https://drive.google.com/uc?id=1i-YzLRM7MJ1bOL5mlicTamwuhPpT8igK'
print('Begin downloading trained model: bert-large')
subprocess.call(['gdown', model_url])
shutil.move('bert-large.state_dict','../output/')

model_url = 'https://drive.google.com/uc?id=1U3h_KSLQn51bSYOfXHIXFm2cV9PVqldH'
print('Begin downloading trained model: albert-base')
subprocess.call(['gdown', model_url])
shutil.move('albert-base-v1.state_dict','../output/')

model_url = 'https://drive.google.com/uc?id=15RsC8iyo_CsglQ6xcSzGBn28M1jybzx4'
print('Begin downloading trained model: albert-large')
subprocess.call(['gdown', model_url])
shutil.move('albert-large-v2.state_dict','../output/')

model_url = 'https://drive.google.com/uc?id=1N2WhpQmdNrDiMEV6aOF8bxWsnPGi4_rn'
print('Begin downloading trained model: bert-base-hans')
subprocess.call(['gdown', model_url])
shutil.move('bert-base_hans.state_dict','../output/')


print("All datasets downloaded and extracted. Trained models have been downloaded and put to output/")
