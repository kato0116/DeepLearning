import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
import matplotlib.pyplot as plt
import numpy as np

# 乱数固定
def torch_seed(seed=123):   
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    
data_root = "./pytorch/data" # ダウンロード先のディレクト名
# データの前処理
transform = transforms.Compose([
    transforms.ToTensor(), # データをテンソル化処理
    transforms.Normalize(0.5,0.5), # データの正規化 範囲を[-1,1]に変更できる
])

train_set = datasets.MNIST(
    root  = data_root,
    train = True,     # 訓練データかテストデータか
    download  = True, # 元データがない場合ダウンロードする
    transform = transform
)

batch_size = 64 # ミニバッチのサイズ指定
# 訓練用データローダ―
train_loader = DataLoader(
    train_set,
    batch_size = batch_size,
    shuffle    = True
)

class Generator(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size,1024)
        self.fc2 = nn.Linear(1024,2048)
        self.fc3 = nn.Linear(2048,784)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self,z):
        z1 = self.relu(self.fc1(z))
        z2 = self.relu(self.fc2(z1))
        z3 = self.fc3(z2)
        x  = z3.view(-1,1,28,28)
        return self.tanh(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,512)
        self.fc2 = nn.Linear(512,1)
        self.relu    = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x  = x.view(-1,784)
        x1 = self.relu(self.fc1(x))
        x2 = self.fc2(x1)
        return self.sigmoid(x2)
    
torch_seed()
lr = 2e-4
epochs = 30
noise_size = 100
net_G = Generator(noise_size)
net_D = Discriminator()
criterion = nn.BCELoss() # 損失関数: 2値分類交差エントロピー
# G,Dの最適化
optimizer_G = optim.Adam(net_G.parameters(),lr=lr)
optimizer_D = optim.Adam(net_D.parameters(),lr=lr)
history = np.zeros((epochs,3))
for epoch in range(epochs):
    print(f'{epoch+1}エポック')
    run_loss_G, run_loss_D = 0, 0
    
    # 学習
    for real_img, _ in train_loader:
        batch_size = real_img.size(0)
        # 勾配の初期化
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        
        z = (torch.randn(batch_size,noise_size)-0.5)/0.5 # 100次元のランダムな乱数を生成
        
        fake_labels = torch.zeros(batch_size,1) # 偽物の画像のラベル
        real_labels = torch.ones(batch_size,1)  # 本物の画像のラベル
        
        fake_img = net_G(z)
        fake_outputs = net_D(fake_img)
        loss_G = criterion(fake_outputs,fake_labels)     
        loss_G.backward()       # 勾配計算
        optimizer_G.step()      # パラメータの更新
        
        real_outputs   = net_D(real_img)
        loss_D_real    = criterion(real_outputs,real_labels)
        fake_outputs   = net_D(fake_img.detach()) # .detachは勾配情報の切り離し。 Generatorへの勾配の更新を防ぐ
        loss_D_fake    = criterion(fake_outputs,fake_labels)
        loss_D = (loss_D_fake+loss_D_real) / 2
        loss_D.backward()
        optimizer_D.step()
        
        run_loss_G += loss_G.item()
        run_loss_D += loss_D.item()
    run_loss_G /= len(train_loader)
    run_loss_D /= len(train_loader)     
    print(f"loss_G :{run_loss_G}")
    print(f"loss_D :{run_loss_D}")
    history[epoch] = [epoch + 1, run_loss_G, run_loss_D]

    # 生成した画像を表示・保存
    with torch.no_grad():
        z = torch.randn(16, noise_size)  # 16枚の画像を生成
        generated_images = net_G(z)
        
    plt.figure(figsize=(4, 4))
    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i+1)
        # plt.imshow(generated_images[i,:,:, 0].cpu().numpy(), cmap='gray')
        plt.imshow(generated_images[i,:,:, 0]*127.5+127.5, cmap='gray')
        plt.axis('off')
    plt.savefig('1111_image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


plt.plot(history[:,0],history[:,1],c='r',label="generator")
plt.plot(history[:,0],history[:,2],c="b",label="discriminator")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN")
plt.legend()
plt.show()