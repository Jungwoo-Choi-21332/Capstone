from msilib.schema import Class
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SimpleCNN(nn.Module):

    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        # feature extractor
        self.features = nn.Sequential(
            # 32x32 → 16x16
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 16x16 → 8x8
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        # CNN feature extraction
        x = self.features(x)
        # flatten
        x = x.view(x.size(0), -1)
        # classification
        x = self.classifier(x)
        return x

class CNNFourLayer(nn.Module):

    def __init__(self, num_classes=3):
        super(CNNFourLayer, self).__init__()

        self.features = nn.Sequential(

            # 32x32
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            # 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # 16x16
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 8x8
        )

        self.classifier = nn.Sequential(

            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

class CNNFiveLayer(nn.Module):

    def __init__(self, num_classes=3):
        super(CNNFiveLayer, self).__init__()

        self.features = nn.Sequential(

            # 32x32
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            # 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # 16x16
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return
#사용방법
# from models.custom_cnn import CNNFourLayer, CNNFourLayer
#
# model = CNNFourLayer(num_classes=len(classes))
#
# model = CNNFourLayer(num_classes=len(classes))

#MLP network

class WaferMLP(nn.Module):
    def __init__(self, num_classes=3):
        super(WaferMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#residual Layer
class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

class ResidualCNN(nn.Module):

    def __init__(self, num_classes=3):
        super(ResidualCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(32),
            nn.MaxPool2d(2),  # 16x16
            ResidualBlock(32),
            nn.MaxPool2d(2)  # 8x8
        )

        self.classifier = nn.Sequential(

            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            padding=1,
            groups=in_ch
        )

        self.pointwise = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=1
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu(x)
        return x


class DepthwiseCNN(nn.Module):
    """
    Depthwise separable convolution 사용

    특징:
    - 파라미터 수 감소
    - crossbar 효율 증가 가능
    - MobileNet 스타일 구조
    """

    def __init__(self, num_classes=3):
        super(DepthwiseCNN, self).__init__()

        self.features = nn.Sequential(
            DepthwiseSeparableConv(1, 16),
            nn.MaxPool2d(2),  # 16x16
            DepthwiseSeparableConv(16, 32),
            nn.MaxPool2d(2)  # 8x8
        )

        self.classifier = nn.Sequential(

            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 사용방법
# from models.other_networks import (
#     WaferMLP,
#     ResidualCNN,
#     DepthwiseCNN
# )
#
# model = WaferMLP(num_classes=len(classes))
# model = ResidualCNN(num_classes=len(classes))
# model = DepthwiseCNN(num_classes=len(classes))

class AlexNetSmall(nn.Module):

    def __init__(self, num_classes=3):
        super(AlexNetSmall, self).__init__()

        self.features = nn.Sequential(

            # 32x32 → 16x16
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 16x16 → 8x8
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 8x8
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2)  # 4x4
        )

        self.classifier = nn.Sequential(

            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        # 차원 맞추기용 shortcut
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out += identity

        out = self.relu(out)

        return out


# ---------------------------------
# ResNet (small version)
# ---------------------------------
class ResNetSmall(nn.Module):

    def __init__(self, num_classes=3):
        super(ResNetSmall, self).__init__()

        self.initial = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.layer1 = BasicBlock(32, 32)

        self.layer2 = BasicBlock(
            32,
            64,
            stride=2  # 32x32 → 16x16
        )

        self.layer3 = BasicBlock(
            64,
            128,
            stride=2  # 16x16 → 8x8
        )

        self.classifier = nn.Sequential(

            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 사용방법
# from models.classic_networks import AlexNetSmall, ResNetSmall
#
# model = AlexNetSmall(num_classes=len(classes))
#
# 또는
# model = ResNetSmall(num_classes=len(classes))

class EchoStateNetwork(nn.Module):

    def __init__(
        self,
        input_size=32*32,
        reservoir_size=500,
        num_classes=3,
        spectral_radius=0.9
    ):
        super(EchoStateNetwork, self).__init__()
        self.reservoir_size = reservoir_size
        # 입력 → reservoir
        self.Win = nn.Parameter(
            torch.randn(reservoir_size, input_size) * 0.1,
            requires_grad=False
        )
        # reservoir recurrent weight
        W = torch.randn(reservoir_size, reservoir_size)
        # spectral radius 조정
        eigvals = torch.linalg.eigvals(W)
        max_eig = torch.max(torch.abs(eigvals))
        W = W * (spectral_radius / max_eig)
        self.Wres = nn.Parameter(
            W.real,
            requires_grad=False
        )
        # readout layer (학습됨)
        self.readout = nn.Linear(
            reservoir_size,
            num_classes
        )

    def forward(self, x):
        batch_size = x.size(0)
        # 32x32 → 1024 vector
        x = x.view(batch_size, -1)
        # reservoir 초기 상태
        h = torch.zeros(
            batch_size,
            self.reservoir_size,
            device=x.device
        )
        # 단일 step reservoir 업데이트
        h = torch.tanh(
            torch.matmul(x, self.Win.T) +
            torch.matmul(h, self.Wres.T)
        )
        # classification
        out = self.readout(h)

        return out
#사용방법
# from models.esn import EchoStateNetwork
#
# model = EchoStateNetwork(
#     input_size=32*32,
#     reservoir_size=500,
#     num_classes=len(classes)
# )

class WaferLSTM(nn.Module):
    """
    LSTM 기반 모델

    32x32 이미지를 sequence로 변환하여 처리
    """

    def __init__(
        self,
        input_size=32,
        hidden_size=128,
        num_layers=1,
        num_classes=3
    ):

        super(WaferLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )


    def forward(self, x):

        # (batch,1,32,32)
        x = x.squeeze(1)
        # (batch, 32, 32)
        # 32 step sequence
        out, _ = self.lstm(x)
        # 마지막 step 사용
        out = out[:, -1, :]
        out = self.classifier(out)

        return out

class WaferGRU(nn.Module):
    """
    GRU 기반 모델

    LSTM보다 단순한 recurrent 구조
    """
    def __init__(
        self,
        input_size=32,
        hidden_size=128,
        num_layers=1,
        num_classes=3
    ):
        super(WaferGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )
    def forward(self, x):
        x = x.squeeze(1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out

class LIFNeuron(nn.Module):

    def __init__(
        self,
        size,
        threshold=1.0,
        decay=0.9
    ):

        super(LIFNeuron, self).__init__()

        self.threshold = threshold
        self.decay = decay
        self.size = size


    def forward(self, input_current):
        batch_size = input_current.size(0)
        membrane = torch.zeros(
            batch_size,
            self.size,
            device=input_current.device
        )

        membrane = self.decay * membrane + input_current
        spikes = (membrane > self.threshold).float()
        membrane = membrane * (1 - spikes)

        return spikes

class SimpleSNN(nn.Module):

    def __init__(
        self,
        input_size=32*32,
        hidden_size=256,
        num_classes=3
    ):

        super(SimpleSNN, self).__init__()

        self.fc1 = nn.Linear(
            input_size,
            hidden_size
        )

        self.spike1 = LIFNeuron(hidden_size)

        self.fc2 = nn.Linear(
            hidden_size,
            num_classes
        )


    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        spikes = self.spike1(x)
        out = self.fc2(spikes)
        return out

# from models.recurrent_and_snn import (
#     WaferLSTM,
#     WaferGRU,
#     SimpleSNN
# )
#
# model = WaferLSTM(num_classes=len(classes))
# model = WaferGRU(num_classes=len(classes))
# model = SimpleSNN(num_classes=len(classes))

class PatchEmbedding(nn.Module):

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        embed_dim=64
    ):

        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # (B,1,32,32)
        x = self.proj(x)
        # (B,embed_dim,8,8)
        x = x.flatten(2)
        # (B,embed_dim,64)
        x = x.transpose(1,2)
        # (B,64,embed_dim)
        return x

class TransformerEncoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim=64,
        num_heads=4,
        mlp_dim=128,
        dropout=0.1
    ):

        super(TransformerEncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # attention
        h = self.norm1(x)
        h,_ = self.attn(h,h,h)
        x = x + self.dropout(h)
        # MLP
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + self.dropout(h)
        return x

class VisionTransformerSmall(nn.Module):

    def __init__(
        self,
        num_classes=3,
        embed_dim=64,
        depth=4,
        num_heads=4,
        mlp_dim=128
    ):

        super(VisionTransformerSmall, self).__init__()

        self.patch_embed = PatchEmbedding(
            img_size=32,
            patch_size=4,
            embed_dim=embed_dim
        )
        num_patches = 64
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, embed_dim)
        )

        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim,
                    num_heads,
                    mlp_dim
                )
                for _ in range(depth)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.patch_embed(x)
        x = x + self.pos_embedding
        x = self.encoder(x)
        # global average pooling
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

# from models.transformer_models import VisionTransformerSmall
#
# model = VisionTransformerSmall(
#     num_classes=len(classes),
#     embed_dim=64,
#     depth=4
# )

class SimpleBERT(nn.Module):
    def __init__(self,num_classes=3,seq_len=32,embed_dim=64,num_heads=4,depth=4):
        super(SimpleBERT,self).__init__()
        self.embedding=nn.Linear(32,embed_dim)
        self.pos_embedding=nn.Parameter(torch.randn(1,seq_len,embed_dim))
        encoder_layer=nn.TransformerEncoderLayer(d_model=embed_dim,nhead=num_heads,batch_first=True)
        self.encoder=nn.TransformerEncoder(encoder_layer,num_layers=depth)
        self.classifier=nn.Sequential(nn.Linear(embed_dim,128),nn.ReLU(),nn.Linear(128,num_classes))

    def forward(self,x):
        x=x.squeeze(1)
        x=self.embedding(x)
        x=x+self.pos_embedding
        x=self.encoder(x)
        x=x.mean(dim=1)
        x=self.classifier(x)
        return x


class Generator(nn.Module):
    def __init__(self,latent_dim=100):
        super(Generator,self).__init__()
        self.net=nn.Sequential(nn.Linear(latent_dim,256),nn.ReLU(),nn.Linear(256,512),nn.ReLU(),nn.Linear(512,32*32),nn.Tanh())

    def forward(self,z):
        x=self.net(z)
        x=x.view(-1,1,32,32)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.net=nn.Sequential(nn.Linear(32*32,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,1),nn.Sigmoid())

    def forward(self,x):
        x=x.view(x.size(0),-1)
        return self.net(x)


class SimpleGAN(nn.Module):
    def __init__(self,latent_dim=100):
        super(SimpleGAN,self).__init__()
        self.generator=Generator(latent_dim)
        self.discriminator=Discriminator()


class GCNLayer(nn.Module):
    def __init__(self,in_features,out_features):
        super(GCNLayer,self).__init__()
        self.linear=nn.Linear(in_features,out_features)

    def forward(self,x,adj):
        x=torch.matmul(adj,x)
        x=self.linear(x)
        return F.relu(x)


class SimpleGNN(nn.Module):
    def __init__(self,num_nodes=32*32,hidden_dim=64,num_classes=3):
        super(SimpleGNN,self).__init__()
        self.gcn1=GCNLayer(1,hidden_dim)
        self.gcn2=GCNLayer(hidden_dim,hidden_dim)
        self.classifier=nn.Sequential(nn.Linear(hidden_dim,128),nn.ReLU(),nn.Linear(128,num_classes))

    def forward(self,x):
        batch_size=x.size(0)
        x=x.view(batch_size,1024,1)
        adj=torch.eye(1024,device=x.device)
        x=self.gcn1(x,adj)
        x=self.gcn2(x,adj)
        x=x.mean(dim=1)
        x=self.classifier(x)
        return x

#사용방법
# from models.advanced_architectures import (
#     SimpleBERT,
#     SimpleGAN,
#     SimpleGNN
# )
# model = SimpleBERT(num_classes=len(classes))
# model = SimpleGNN(num_classes=len(classes))
# gan = SimpleGAN()
# generator = gan.generator
# discriminator = gan.discriminator

class VAE(nn.Module):
    def __init__(self,latent_dim=64):
        super(VAE,self).__init__()
        self.encoder=nn.Sequential(nn.Linear(32*32,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.mu=nn.Linear(128,latent_dim)
        self.logvar=nn.Linear(128,latent_dim)
        self.decoder=nn.Sequential(nn.Linear(latent_dim,128),nn.ReLU(),nn.Linear(128,256),nn.ReLU(),nn.Linear(256,32*32),nn.Sigmoid())

    def reparameterize(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        return mu+eps*std

    def forward(self,x):
        x=x.view(x.size(0),-1)
        h=self.encoder(x)
        mu=self.mu(h)
        logvar=self.logvar(h)
        z=self.reparameterize(mu,logvar)
        recon=self.decoder(z)
        recon=recon.view(-1,1,32,32)
        return recon,mu,logvar



class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim=dim

    def forward(self,t):
        device=t.device
        half=self.dim//2
        emb=torch.exp(torch.arange(half,device=device)*-(torch.log(torch.tensor(10000.0))/ (half-1)))
        emb=t[:,None]*emb[None,:]
        emb=torch.cat((emb.sin(),emb.cos()),dim=1)
        return emb



class DiffusionMLP(nn.Module):
    def __init__(self,time_dim=64):
        super().__init__()
        self.time_embed=SinusoidalTimeEmbedding(time_dim)
        self.net=nn.Sequential(nn.Linear(32*32+time_dim,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,32*32))

    def forward(self,x,t):
        x=x.view(x.size(0),-1)
        t_emb=self.time_embed(t)
        x=torch.cat([x,t_emb],dim=1)
        x=self.net(x)
        return x.view(-1,1,32,32)



class ConditionalDiffusion(nn.Module):
    def __init__(self,num_classes=3,time_dim=64):
        super().__init__()
        self.label_embed=nn.Embedding(num_classes,32)
        self.time_embed=SinusoidalTimeEmbedding(time_dim)
        self.net=nn.Sequential(nn.Linear(32*32+time_dim+32,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,32*32))

    def forward(self,x,t,y):
        x=x.view(x.size(0),-1)
        t_emb=self.time_embed(t)
        y_emb=self.label_embed(y)
        x=torch.cat([x,t_emb,y_emb],dim=1)
        x=self.net(x)
        return x.view(-1,1,32,32)



class LatentDiffusion(nn.Module):
    def __init__(self,latent_dim=64,time_dim=64):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(32*32,256),nn.ReLU(),nn.Linear(256,latent_dim))
        self.decoder=nn.Sequential(nn.Linear(latent_dim,256),nn.ReLU(),nn.Linear(256,32*32))
        self.time_embed=SinusoidalTimeEmbedding(time_dim)
        self.diffusion=nn.Sequential(nn.Linear(latent_dim+time_dim,256),nn.ReLU(),nn.Linear(256,latent_dim))

    def forward(self,x,t):
        x=x.view(x.size(0),-1)
        z=self.encoder(x)
        t_emb=self.time_embed(t)
        zt=torch.cat([z,t_emb],dim=1)
        z_pred=self.diffusion(zt)
        recon=self.decoder(z_pred)
        return recon.view(-1,1,32,32)

# from models.vae_and_diffusion_clean import (
#     VAE,
#     DiffusionMLP,
#     ConditionalDiffusion,
#     LatentDiffusion
# )
# model = VAE(latent_dim=64)
# model = DiffusionMLP()
# model = ConditionalDiffusion(num_classes=len(classes))
# model = LatentDiffusion()


#memtorch 사용은 모든 network 동일
# memristive_model = memtorch.patch_model(
#     model,
#     memristor_model=reference_memristor,
#     tile_shape=(128,128),
#     adc_bitwidth=8,
#     dac_bitwidth=8
# )