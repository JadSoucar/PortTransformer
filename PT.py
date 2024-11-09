import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import pandas as pd
import numpy as np

# Time2Vec Layer as defined by Kazemi
def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class Time2Vec(nn.Module):
    def __init__(self, activation, input_dim, embed_dim):
        super(Time2Vec, self).__init__()
        self.l1 = SineActivation(input_dim, embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.fc1(x)
        return x


# Other Network Components (Gated Residual Network, Attention Layers, etc.)
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        residual = x
        x = F.elu(self.fc1(x))  # Apply ELU activation
        x = self.fc2(x)
        gate = torch.sigmoid(self.gate(residual))
        x = x * gate  # Gated Linear Unit
        return self.layer_norm(x + residual)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.grn = GatedResidualNetwork(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.layer_norm(attn_output + x)
        grn_output = self.grn(x)
        return self.layer_norm(grn_output + x)

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(DecoderLayer, self).__init__()
        self.masked_attention = MultiHeadAttention(embed_dim, num_heads)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.grn = GatedResidualNetwork(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, enc_output):
        masked_attn_output = self.masked_attention(x)
        x = self.layer_norm(masked_attn_output + x)
        attn_output = self.attention(x)
        x = self.layer_norm(attn_output + x)
        grn_output = self.grn(x)
        return self.layer_norm(grn_output + x)

class Encoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, enc_output):
        for layer in self.layers:
            x = layer(x, enc_output)
        return x

class OutputLayer(nn.Module):
    def __init__(self, embed_dim, num_assets):
        super(OutputLayer, self).__init__()
        self.fc = nn.Linear(embed_dim, num_assets)  # Output weights for each asset
    
    def forward(self, x):
        s = self.fc(x)  
        
        # Apply normalization (sign and softmax for short-selling)
        weights = torch.sign(s) * F.softmax(torch.abs(s), dim=-1)
        return weights  # Shape: [batch_size, num_assets]


class PortfolioTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, num_assets):
        super(PortfolioTransformer, self).__init__()
        self.time2vec = Time2Vec(SineActivation, input_dim, embed_dim)
        self.encoder = Encoder(num_layers, embed_dim, num_heads)
        self.decoder = Decoder(num_layers, embed_dim, num_heads)
        self.output_layer = OutputLayer(embed_dim, num_assets)
    
    def forward(self, x):
        x = self.time2vec(x) 
        x = x.permute(1, 0, 2)
        
        # Pass through encoder and decoder
        enc_output = self.encoder(x)
        dec_output = self.decoder(x, enc_output)
        
        # Take only the last timestep output
        last_timestep = dec_output[-1]
        
        # Generate the final portfolio weights vector
        out = self.output_layer(last_timestep)
        return out


class FinancialDataset(Dataset):
    def __init__(self, returns, seq_length=30):
        self.returns = returns.astype(np.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.returns) - self.seq_length

    def __getitem__(self, idx):
        # Extract sequences and ensure shape [seq_length, num_assets]
        x = self.returns[idx:idx + self.seq_length, :]  
        y = self.returns[idx + 1:idx + 1 + self.seq_length, :]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

def sharpe_loss(weights, returns, prev_weights, transaction_cost=0.0002):
    prev_weights = prev_weights.detach()
    portfolio_returns = torch.sum(weights.unsqueeze(1) * returns, dim=-1)
    adjusted_returns = portfolio_returns
    mean_return = torch.mean(adjusted_returns, dim=1)
    std_return = torch.std(adjusted_returns, dim=1)
    epsilon = 1e-6
    sharpe_ratio = mean_return / (std_return + epsilon)
    return -torch.mean(sharpe_ratio)


