import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import os
import matplotlib.pyplot as plt
import numpy as np
  
# 乱数固定
def torch_seed(seed=123):   
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

# 画像を表示する
def show_img(img,epoch,nrow=4):
  img = img.detach()
  img_grid = make_grid(img, nrow) # 画像を並べる
  plt.figure(figsize=(4, 4))
  plt.imshow(img_grid.permute(1, 2, 0)) # 画像の表示
  plt.axis("off")
  # plt.savefig('1111_image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  
# 重みの初期化
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
    
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

batch_size = 128 # ミニバッチのサイズ指定
# 訓練用データローダ―
train_loader = DataLoader(
    train_set,
    batch_size = batch_size,
    shuffle    = True
)

class Generator(nn.Module):
    """
    z_dim: zの
    hidden_channel: 中間層のチャネル数
    img_dim: 生成する画像の次元数
    """
    def __init__(self,z_dim,hidden_channel=128,img_dim=1):
        super().__init__()
        self.z_dim = z_dim
        self.convt1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim,hidden_channel*4,kernel_size=3,stride=2),
            nn.BatchNorm2d(hidden_channel*4),
            nn.ReLU(inplace=True)   
        )
        self.convt2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channel*4,hidden_channel*2,kernel_size=4,stride=1),
            nn.BatchNorm2d(hidden_channel*2),
            nn.ReLU(inplace=True)
        )
        self.convt3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channel*2,hidden_channel,kernel_size=3,stride=2),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True)
        )
        self.convt4 = nn.ConvTranspose2d(hidden_channel,img_dim,kernel_size=4,stride=2)
        self.tanh   = nn.Tanh()

    def forward(self,z):
        z = z.view(len(z),self.z_dim,1,1)
        x = self.convt1(z)
        x = self.convt2(x)
        x = self.convt3(x)
        x = self.convt4(x)
        # x = self.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,img_chanel,hidden_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_chanel,hidden_channel,kernel_size=4,stride=2),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channel, hidden_channel*2, kernel_size=4, stride=2),
            nn.BatchNorm2d(hidden_channel*2),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.3)
        )
        self.conv3 = nn.Conv2d(hidden_channel*2,1,kernel_size=4,stride=2)
        # self.flatten = nn.Flatten()
        # self.fc = nn.Linear(7*7*128,1)
        # self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
       
        return x.view(len(x),-1)
    
# torch_seed()
lr_rate       = 2e-4
beta_1,beta_2 = 0.5,0.999
epochs     = 30
noise_size = 64
test_size  = 16
image_chanel   = 1
hidden_channel = 16
net_G = Generator(noise_size)
net_D = Discriminator(image_chanel,hidden_channel)
# 重みの初期化
net_G = net_G.apply(weights_init)
net_D = net_D.apply(weights_init)

criterion = nn.BCEWithLogitsLoss() # 損失関数: 2値分類交差エントロピー

# G,Dの最適化
optimizer_G = optim.Adam(net_G.parameters(),lr=lr_rate,betas=(beta_1,beta_2))
optimizer_D = optim.Adam(net_D.parameters(),lr=lr_rate,betas=(beta_1,beta_2))
history = np.zeros((epochs,3))
for epoch in range(epochs):
    print(f'{epoch+1}エポック')
    run_loss_G, run_loss_D = 0, 0
    count = 0
    
    # 学習
    for real_imgs, _ in train_loader:
        count += 1
        batch_size = real_imgs.size(0)
        
        fake_labels = torch.zeros(batch_size,1) # 偽物の画像のラベル
        real_labels = torch.ones(batch_size,1)  # 本物の画像のラベル
        
        # ---識別器(Discriminator)---
        optimizer_D.zero_grad()                 # 勾配の初期化
        
        z = torch.randn(batch_size,noise_size)   # 100次元のランダムな乱数を生成
        fake_imgs = net_G(z)                     # 偽画像の生成
        
        fake_outputs = net_D(fake_imgs.detach()) # .detach()は勾配情報の切り離し。 Generatorへの勾配の更新を防ぐ
        real_outputs = net_D(real_imgs)
        
        # 識別器(Discriminator)の損失計算
        loss_D_real = criterion(real_outputs,real_labels) # 本物の画像を本物と判断したとき
        loss_D_fake = criterion(fake_outputs,fake_labels) # 偽物の画像を偽物と判断したとき (誤認識)
        loss_D = (loss_D_fake+loss_D_real)/2
        
        # 識別器(Discriminator)の勾配計算
        loss_D.backward()
        optimizer_D.step()
        
        # ---生成器(Generator)---
        optimizer_G.zero_grad() # 勾配の初期化
        
        z = torch.randn(batch_size,noise_size)   # 100次元のランダムな乱数を生成
        fake_imgs = net_G(z)                     # 偽画像の生成
        
        # 生成器(Generator)の勾配計算
        fake_outputs = net_D(fake_imgs)
        loss_G = criterion(fake_outputs,real_labels)        # 偽物の画像を本物と判断したとき (誤認識)
        
        # 生成器(Generator)の勾配計算
        loss_G.backward()
        optimizer_G.step()
        
        run_loss_G += loss_G.item()
        run_loss_D += loss_D.item()

    run_loss_G /= count
    run_loss_D /= count
    print(f"loss_G :{run_loss_G}")
    print(f"loss_D :{run_loss_D}")
    history[epoch] = [epoch + 1, run_loss_G, run_loss_D]
    # # 生成した画像を表示・保存
    with torch.no_grad():
        z = torch.randn(test_size, noise_size)  # 16枚の画像を生成
        test_imgs = net_G(z)
        show_img(test_imgs,epoch)
        
    # plt.figure(figsize=(4, 4))
    # for i in range(generated_images.shape[0]):
    #     plt.subplot(4, 4, i+1)
    #     plt.imshow(generated_images[i,:,:, 0], cmap='gray')
    #     plt.axis('off')
    # 
    # plt.show()


plt.plot(history[:,0],history[:,1],c='r',label="generator")
plt.plot(history[:,0],history[:,2],c="b",label="discriminator")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN")
plt.legend()
plt.show()