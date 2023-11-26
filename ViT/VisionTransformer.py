import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from einops.layers.torch import Rearrange # パッチ分割用

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
  plt.figure(figsize=(5,4))
  plt.imshow(img_grid.permute(1, 2, 0)) # 画像の表示
  plt.axis("off")
  plt.savefig('1111_image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

# torch_seed()
data_root = "./pytorch/data" # ダウンロード先のディレクト名
# データの前処理
transform = transforms.Compose([
    transforms.ToTensor(), # データをテンソル化処理
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # データの正規化 範囲を[-1,1]に変更できる
])
train_set = datasets.CIFAR10(
    root  = data_root,
    # split = "train", 
    train = True,
    download  = True,
    transform = transform
)

batch_size = 64  # ミニバッチのサイズ指定

# 訓練用データローダ―
train_loader = DataLoader(
    train_set,
    batch_size = batch_size,
    shuffle    = True
)

# パッチ分割
class Patching(nn.Module):
  """
  論文: (C,H,W) --> (N, (P^2)*C) 
  上記のNはパッチ数, Pはパッチサイズ
  
  P: パッチの高さ,幅 (パッチは正方形)
  Rearrange: パッチ分割を行う. (N, C, H, W) -> (N, (H*W)/(P^2), (P^2)*C))
  上記のNはバッチ数。ミニバッチ処理を想定.
  """
  def __init__(self,patch_size):
    super().__init__()
    self.split_patch = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=patch_size, pw=patch_size)
  def forward(self,x):
    x_p = self.split_patch(x) # パッチ分割
    return x_p

# 埋め込み処理
class Embedding(nn.Module):
  """
  patch_dim : パッチ分割後のパッチの次元数 (=P^2*C)
  patch_num : パッチ数 (=H*W/P^2)
  token_dim : Transformer内で扱われる潜在ベクトルの次元
  
  patch_embedding: x_p^i * E {(i=1,2,...,N), E(∈R^(P^2*C, D))}
  """
  def __init__(self,patch_dim,patch_num,token_dim):
    super().__init__()
    self.patch_embedding   = nn.Linear(patch_dim,token_dim) # P^2*C次元をD次元に変換
    # nn.Parameter: Parameterオブジェクトに変換する → 学習可能なパラメータにする
    self.class_token       = nn.Parameter(torch.randn(1,1,token_dim))           # [class]トークンを作成
    self.postion_embedding = nn.Parameter(torch.randn(1,patch_num+1,token_dim)) # E_posの作成
  def forward(self,x):
    z0 = self.patch_embedding(x)
    batch_size = z0.shape[0]
    # [class]トークンを先頭に挿入
    class_tokens = []
    for _ in range(batch_size):
        class_tokens.append(self.class_token)     # cls_tokenを追加してリストに追加
    class_tokens = torch.cat(class_tokens, dim=0) # リストを結合して新しいテンソルに変換
    z0 = torch.cat([z0,class_tokens],dim=1)
    # 位置埋め込み (仮)
    z0 += self.postion_embedding
    return z0

# 自己注意のマルチヘッドアテンション
class MultiHeadAttention(nn.Module):
  """
  token_dim: Transformer内で扱われる潜在ベクトルサイズ
  head_num: headの数
  head_dim: head分割時のD次元からD'次元に線形写像するときのD'のこと
  """
  def __init__(self,token_dim,head_num):
    super().__init__()
    self.head_num = head_num
    self.head_dim = token_dim//head_num
    
    self.W_q = nn.Linear(token_dim,token_dim,bias=False) # query
    self.W_k = nn.Linear(token_dim,token_dim,bias=False) # key
    self.W_v = nn.Linear(token_dim,token_dim,bias=False) # value
    self.split_head = Rearrange("b n (d h) -> b h n d", h=self.head_num) # headに分割
    self.softmax = nn.Softmax(dim=-1) # 各行ベクトルに対して独立にsoftmaxを適用
    self.concat  = Rearrange("b h n d -> b n (h d)", h=self.head_num)
    self.W_o = nn.Linear(token_dim,token_dim,bias=False)
  def forward(self,x):
    Q = self.W_q(x) # Q*W_q
    K = self.W_k(x) # K*W_k
    V = self.W_v(x) # V*W_v
    
    Q = self.split_head(Q)
    K = self.split_head(K)
    V = self.split_head(V)
    
    K_t = K.transpose(-1,-2)
    attention_w = self.softmax(torch.matmul(Q,K_t)/np.sqrt(self.head_dim))
    heads = torch.matmul(attention_w,V)
    heads = self.concat(heads)
    out   = self.W_o(heads)
    return out
    
# 多層パーセプトロン
class MultiLayerPerceptron(nn.Module):
  """
  token_dim: Transformer内で扱われる潜在ベクトルサイズ
  hidden_dim: 隠れ層のパーセプトロン
  """
  def __init__(self,token_dim,hidden_dim):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(token_dim,hidden_dim),
      nn.GELU(), # ReLUのパワーアップ版？、高速なやつ
      nn.Linear(hidden_dim,token_dim)
    )
  def forward(self,x):
    x = self.net(x)
    return x

class TransformerEncoder(nn.Module):
  """
  token_dim: Transformer内で扱われる潜在ベクトルのサイズ
  L: Transformerブロックの繰り返し回数
   
  """
  def __init__(self,token_dim,head_num,hidden_dim,depth):
    super().__init__()
    self.norm = nn.LayerNorm(token_dim)
    self.MSA_block = MultiHeadAttention(token_dim,head_num) 
    self.MLP_block = MultiLayerPerceptron(token_dim,hidden_dim)
    self.depth = depth
    
  def forward(self,x):
    for _ in range(self.depth):
      x = self.MSA_block(self.norm(x)) + x
      x = self.MLP_block(self.norm(x)) + x
    return x

class MLPHead(nn.Module):
  def __init__(self,token_dim,out_dim):
    super().__init__()
    self.net = nn.Sequential(
      nn.LayerNorm(token_dim),
      nn.Linear(token_dim,out_dim)
    )
  def forward(self,x):
    x = self.net(x)
    return x

class VisionTransformer(nn.Module):
  def __init__(self,H,W,C,patch_size,token_dim,hidden_dim,output_dim,head_num,depth):
    super().__init__()
    self.patch_size = patch_size
    self.patch_num  = (H*W)//patch_size**2
    self.patch_dim  = (patch_size**2) * C
    self.token_dim  = token_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.head_num   = head_num
    self.depth      = depth
    self.patching  = Patching(self.patch_size)
    self.embedding = Embedding(self.patch_dim,self.patch_num,self.token_dim)
    self.encoder   = TransformerEncoder(self.token_dim,self.head_num,self.hidden_dim,self.depth)
    self.mlphead   = MLPHead(self.token_dim,self.output_dim)
  
  def forward(self,x):
    x_p = self.patching(x)    # (batch_size,patch_num,patch_dim)
    z0 = self.embedding(x_p)  # (batch_size,patch_num,token_dim) <--[cls]トークンを追加してるためpatch_numの数が1つ増えている
    zL = self.encoder(z0)     # (batch_size,patch_num,token_dim)
    zL_0 = zL[:,0]            # (batch_size,patch_num,1)         <--[class]トークンを取り出す
    y    = self.mlphead(zL_0) # (batch_size,10)
    return y
  
for x, _ in train_loader:
  break

C,H,W = (3,32,32)
patch_size = 2
patch_dim  = patch_size**2*C
patch_num  = (H*W)//patch_size**2
token_dim  = 100
hidden_dim = 80
output_dim = 10
head_num   = 10
depth      = 5


vit = VisionTransformer(H,W,C,patch_size,token_dim,hidden_dim,output_dim,head_num,depth)
y   = vit(x)