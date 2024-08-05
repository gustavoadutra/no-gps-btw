# -*- coding: utf-8 -*-
### 1 Importação das bibliotecas e Obtenção dos dados

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import time
from torchsummary import summary
import imageio.v2 as imageio
import csv
import os
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import zipfile
from IPython.display import display
import time
from glob import glob
from PIL import Image
import psutil
import math
import random

diretorio_origem = './Origem'
diretorio_destino = './Destino'


# Transforma as cores da imagem para escala de cinza
def to_gray(folder):
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img = cv2.imread(img_path)
        if img is not None:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(img_path, gray_img)


diretorio_origem = './Origem'
diretorio_destino = './Destino'


#arquivos_zip = [f for f in os.listdir(diretorio_origem) if f.endswith('.zip')]
arquivos_zip = [] # Para não extrair os arquivos zipados novamente

# Extrái arquivos
for arquivo in arquivos_zip:
    caminho_arquivo = os.path.join(diretorio_origem, arquivo)

    with zipfile.ZipFile(caminho_arquivo, 'r') as zip_ref:
        zip_ref.extractall(diretorio_destino)

    print(f'Arquivo {arquivo} extraído para {diretorio_destino}')

print('Extração concluída.')

caminho_diretorio = "./Destino/KNOW-REFERENCEMAP-2022"
arquivos = os.listdir(caminho_diretorio)

for arquivo in arquivos:
    if arquivo.endswith(".tif"):
        caminho_original = os.path.join(caminho_diretorio, arquivo)
        caminho_destino = os.path.join(caminho_diretorio, arquivo.replace(".tif", ".tiff"))
        os.rename(caminho_original, caminho_destino)
print("Renomeação concluída.")

to_gray(caminho_diretorio)
print("Transformação de cores concluída.")

"""### 2 Normalização, Embaralhamento de Divisão"""

class MakeData(Dataset):
  def __init__(self, root_dir, transform):
    self.root_dir = root_dir
    self.transform = transform
    self.image_list = sorted(os.listdir(root_dir))

  def __len__(self):
    return len(self.image_list)

  def __getitem__(self, idx):
    img_name = os.path.join(self.root_dir, self.image_list[idx])
    #image =  Image.open(img_name)
    image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)  # Ensure image is read in grayscale


    if self.transform:
      image = self.transform(image)
      image = torch.rot90(image, k=1, dims=(1, 2))
    return image

  def get_batch(self,indice,size):
    init = indice * size
    img_names = [os.path.join(self.root_dir, self.image_list[init+i]) for i in range(size)]
    
    if self.transform:
      return [self.transform(cv2.imread(img)) for img in img_names]

def split_data(dataset,img_list):

  size_train = round(0.8*dataset.__len__())
  size_test  = round(0.15*dataset.__len__())
  size_val   = dataset.__len__() - size_train - size_test

  pares = list(zip(dataset, img_list))
  #random.shuffle(pares) seria mais eficiente?
  conjunto_treino, conjunto_teste, conjunto_validacao = random_split(
      pares,
      [size_train,
       size_test,
       size_val])

  train_tensor, train_img = map(list, zip(*conjunto_treino))
  test_tensor, test_img = map(list, zip(*conjunto_teste))
  val_tensor, val_img = map(list, zip(*conjunto_validacao))

  return train_tensor,train_img, test_tensor,test_img, val_tensor,val_img

# Transforma a imagem em um tensor
# Reduz o tamanho da imagem para 160x320
# Transforma as cores para uma escala de 0 a 1
transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((160,320))])
path = "./Destino/KNOW-REFERENCEMAP-2022"

dataset = MakeData(root_dir=path, transform=transform)
conjunto_treino, data1_img, conjunto_teste, data2_img, conjunto_validacao, data3_img = split_data(dataset,dataset.image_list)

print('==============================================')
print(f'Tamanho total do conjunto de dados: {dataset.__len__()}')
print(f'Dados de treinamento: {len(conjunto_treino)}')
print(f'Dados de teste: {len(conjunto_teste)}')
print(f'Dados de validação: {len(conjunto_validacao)}')
print('==============================================')

"""### 3 Modelagem do Autoencoder"""

class AutoEncoder(nn.Module):
  def __init__(self, channels, init_output_size, latent_variable_size):
    super(AutoEncoder, self).__init__()
    self.channels = channels
    self.init_output_size = init_output_size
    self.latent_variable_size = latent_variable_size

    #ENCODER
    self.conv1 = nn.Conv2d(self.channels, init_output_size, kernel_size=4, stride=2, padding=1)
    self.bn1 = nn.BatchNorm2d(init_output_size)

    self.conv2 = nn.Conv2d(init_output_size, init_output_size*2, kernel_size=4, stride=2, padding=1)
    self.bn2 = nn.BatchNorm2d(init_output_size*2)

    self.conv3 = nn.Conv2d(init_output_size*2, init_output_size*4, kernel_size=4, stride=2, padding=1)
    self.bn3 = nn.BatchNorm2d(init_output_size*4)

    self.conv4 = nn.Conv2d(init_output_size*4, init_output_size*8, kernel_size=4, stride=2, padding=1)
    self.bn4 = nn.BatchNorm2d(init_output_size*8)

    self.conv5 = nn.Conv2d(init_output_size*8, init_output_size*8, kernel_size=4, stride=2, padding=1)
    self.bn5 = nn.BatchNorm2d(init_output_size*8)

    self.lin1 = nn.Linear(init_output_size*8*10*5, latent_variable_size)


    #DECODER
    self.lin2 = nn.Linear(latent_variable_size, init_output_size*8*10*5)
    self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
    self.pd1 = nn.ReplicationPad2d(1)
    self.d2 = nn.Conv2d(1024, 1024, 3, 1)
    self.bn6 = nn.BatchNorm2d(1024, 1.e-3)

    self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
    self.pd2 = nn.ReplicationPad2d(1)
    self.d3 = nn.Conv2d(1024, 512, 3, 1)
    self.bn7 = nn.BatchNorm2d(512, 1.e-3)

    self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
    self.pd3 = nn.ReplicationPad2d(1)
    self.d4 = nn.Conv2d(512, 256, 3, 1)
    self.bn8 = nn.BatchNorm2d(256, 1.e-3)

    self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
    self.pd4 = nn.ReplicationPad2d(1)
    self.d5 = nn.Conv2d(256, 128, 3, 1)
    self.bn9 = nn.BatchNorm2d(128, 1.e-3)

    self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
    self.pd5 = nn.ReplicationPad2d(1)
    self.d6 = nn.Conv2d(128, 1, 3, 1)

    self.leakyrelu = nn.LeakyReLU(0.2)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def encode(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.leakyrelu(x)
    l1 = x

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.leakyrelu(x)
    l2 = x

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.leakyrelu(x)
    l3 = x

    x = self.conv4(x)
    x = self.bn4(x)
    x = self.leakyrelu(x)
    l4 = x

    x = self.conv5(x)
    x = self.bn5(x)
    x = self.leakyrelu(x)
    l5 = x

    x = x.reshape(-1,51200)
    x = self.lin1(x)

    return x,l1,l2,l3,l4,l5

  def decode(self, z):
    h1 = self.relu(self.lin2(z))
    h1 = h1.view(-1,1024,10,5)
    dec1 = h1
    h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
    dec2 = h2
    h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
    dec3 = h3
    h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
    dec4 = h4
    h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))
    dec5 = h5

    return self.sigmoid(self.d6(self.pd5(self.up5(h5)))), dec1, dec2, dec3, dec4, dec5

  def get_latent_var(self, x):
      mu, logvar = self.encode(x) #x de fato é igual à 'x.view(-1, self.nc, self.ndf, self.ngf)'
      z = self.reparametrize(mu, logvar)
      return z

  def loss_layer(self,e1,e2,e3,e4,e5,d1,d2,d3,d4,d5,alfa=0.01):

    encode_layers = [e1,e2,e3,e4,e5]
    decode_layers = [d5,d4,d3,d2,d1]
    loss=0
    criterion = nn.MSELoss()

    for i in range(5):
      loss+= criterion(encode_layers[i],decode_layers[i])

    loss = loss*alfa

    return loss

  def forward(self, x):
    z,e1,e2,e3,e4,e5 = self.encode(x)
    res, d1,d2,d3,d4,d5= self.decode(z)
    loss = self.loss_layer(e1,e2,e3,e4,e5,d1,d2,d3,d4,d5)
    return res,loss

"""### 4 Função de Perda"""

def loss_function(recon_x, x):

  MSE = reconstruction_function(recon_x, x)

  return MSE

reconstruction_function = nn.MSELoss()

"""### 5 Loop de treinamento

#### Train
"""

def train(epoch,data,batch_size):

  model.train()

  train_loss = 0
  gpu = torch.device("cuda")
  cpu = torch.device("cpu")

  batch_num = int(len(data) / batch_size)
  ciclos = 0

  min_loss = [0,0,1]
  max_loss = [0,0,0]

  it = iter(DataLoader(data, batch_size=batch_size, shuffle=False))

  for _ in range(batch_num):

    if torch.cuda.is_available():

      x = next(it).to(gpu)

      optimizer.zero_grad()

      y_hat, layer_loss = model(x)

      loss = loss_function(y_hat, x) + layer_loss

      loss.backward()

      train_loss += loss.item()

      optimizer.step()

      min_loss = [y_hat,x,loss.item()] if loss.item() <= min_loss[2] else min_loss
      max_loss = [y_hat,x,loss.item()] if loss.item() >= max_loss[2] else max_loss

      ciclos+=1

      torch.cuda.synchronize()
      torch.cuda.empty_cache()

  avg_loss = train_loss/(ciclos)

  print(f'======>Epoch: {epoch}')
  print(f'Loss: { avg_loss:.4f}')

  return min_loss, max_loss, avg_loss

"""#### Test"""

def test(epoch,data,batch_size):

  model.eval()
  torch.no_grad()

  test_loss = 0
  gpu = torch.device("cuda")
  cpu = torch.device("cpu")

  batch_size = 1#int(batch_size/4)
  batch_num = int(len(data) / batch_size)
  ciclos = 0

  min_loss = [0,0,1]
  max_loss = [0,0,0]

  it = iter(DataLoader(data, batch_size=batch_size, shuffle=False))

  for _ in range(batch_num):

    if torch.cuda.is_available():

      x = next(it).to(gpu)

      y_hat, layer_loss = model(x)

      loss = loss_function(y_hat, x) + layer_loss

      test_loss += loss.item()

      min_loss = [y_hat,x,loss.item()] if loss.item() <= min_loss[2] else min_loss
      max_loss = [y_hat,x,loss.item()] if loss.item() >= max_loss[2] else max_loss

      ciclos+=1

      torch.cuda.synchronize()
      torch.cuda.empty_cache()

  avg_loss = test_loss/(ciclos)

  print(f'======>Epoch: {epoch}')
  print(f'Loss: { avg_loss:.4f}')

  return min_loss, max_loss, avg_loss

def load_last_model():
  #models = glob('./Modelos/Epoch_200_Train_loss_0.0008.pth')
  models = glob('./Modelos/Epocas/Epoch_*.pth')
  if not models:
    print("Nenhum arquivo de modelo encontrado. Certifique-se de que os arquivos estejam no diretório especificado.")
    return 0, 0

  else:
    model_ids = [(int(f.split('_')[1]), f) for f in models]
    start_epoch, last_cp = max(model_ids, key=lambda item:item[0])
    print('Last checkpoint: ', last_cp)
    model.load_state_dict(torch.load(last_cp))

  return start_epoch, last_cp

"""### 5 Treinamento"""

def resume_training(train_data,test_data,epochs,batch_size):
  start_epoch, _ = load_last_model()
  result = []
  result2= []
  e = 0
  loss = 0

  for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
    train_min_loss, train_max_loss, train_loss = train(epoch,train_data,batch_size)
    test_min_loss, test_max_loss, test_loss = test(epoch,test_data,batch_size)
    result.append([train_min_loss, train_max_loss])
    result2.append([test_min_loss, test_max_loss])
    e = epoch
    loss = train_loss
    if loss <= 0.0008 or epoch % 10 == 0:
      torch.save(model.state_dict(), './Modelos/Epocas/Epoch_'+str(e)+'_Train_loss_'+str(round(loss,4))+'.pth')#, test_loss))
  #result = 0
  #result2 = 0
  return result,result2

gpu = torch.device("cuda")
model = AutoEncoder(channels=1, init_output_size=128, latent_variable_size=1000).to(gpu)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

summary(model,(1,320,160))

"""### 6 Análise"""
def decode_real_img(img):

  if load_last_model() == 0:
    print("Nenhum arquivo de modelo encontrado. Certifique-se de que os arquivos estejam no diretório especificado.")
    return
  else:
    model.eval()
    gpu = torch.device("cuda")

    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((160,320))])
    x = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    x = transform(x)
    x = torch.rot90(x, k=1, dims=(1, 2))
    x = x.unsqueeze(0)

    with torch.inference_mode():
      if torch.cuda.is_available():
        x = x.to(gpu)
        #embedding = model.get_embeddings(x)
        img_recon = model(x)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    #return embedding, img['id'], img['img'][1:], img['coord'], img_recon[0]
    #return [embedding, img['id'], img['coord']]
    # Tratamento para voltar a ser imagem
    #img_recon = torch.rot90(img_recon[0], k=-1, dims=(1,2))
    img_recon = img_recon[0].detach().cpu().numpy()
    img_recon = img_recon.squeeze()  # Ensure the image is in the correct shape

    return img_recon
  
#z_img,test_img = resume_training(train_data=conjunto_treino, test_data=conjunto_teste, epochs=10, batch_size=8)

img_real = '/src/Destino/KNOW-REFERENCEMAP-2022/00001_2022_01.tiff'
img_recon = decode_real_img(img_real)
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


# Load the real image
real_image = cv2.imread(img_real, cv2.IMREAD_GRAYSCALE)
real_image_rotated = rotate(real_image, 90)

# Plot the real image and reconstructed image side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(real_image_rotated, cmap='gray')
axes[0].set_title('Real Image')
axes[0].axis('off')
axes[1].imshow(img_recon, cmap='gray')
axes[1].set_title('Reconstructed Image')
axes[1].axis('off')

# Show the plot
plt.show()
# Fazer a imagem real aparecer do lado da imagem reconstruída

