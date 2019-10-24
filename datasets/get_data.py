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

model_path = 'https://drive.google.com/uc?id=1YlS3wuwlhzGRPMyJr8SE4IP89U0W7yVU'
print('Bedinning download of trained model')
subprocess.call(['gdown', model_path])
shutil.move('sample_model.state_dict','../output/')


print("All datasets downloaded and extracted; trained model has been downloaded and put to output/sample_model.state_dict")
