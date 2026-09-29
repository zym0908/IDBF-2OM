import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model).to(device)
        self.pos_embed = nn.Embedding.from_pretrained(self.position_encoding(max_len, d_model), freeze=True).to(device)
        self.norm = nn.LayerNorm(d_model).to(device)

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len]
        pos = torch.arange(seq_len, dtype=torch.long, device=device)  # [seq_len]
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        return self.norm(self.tok_embed(x) + self.pos_embed(pos))

    @staticmethod
    def position_encoding(max_len, d_model):
        """
        Position encoding feature introduced in "Attention is all you need",
        the b is changed to 1000 for the short length of sequence.
        """
        b = 1000
        pos_encoding = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = np.sin(pos / (b ** (2 * i / d_model)))
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = np.cos(pos / (b ** (2 * i / d_model)))
        return torch.FloatTensor(pos_encoding)


class CA1D(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(CA1D, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y1 = self.avg_pool(x)
        y1 = y1.transpose(-1, -2)
        y1 = self.conv(y1)
        y1 = y1.transpose(-1, -2)
        y2 = self.max_pool(x)
        y2 = y2.transpose(-1, -2)
        y2 = self.conv(y2)
        y2 = y2.transpose(-1, -2)
        y = self.sigmoid(y1 + y2)
        return y.expand_as(x)

class SA1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA1D, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def depthwise_separable_conv1d(in_channel, out_channel, kernel_size):
    return nn.Sequential(
        nn.Conv1d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            groups=in_channel,
            bias=False
        ),
        nn.BatchNorm1d(in_channel),
        nn.GELU(), 
        nn.Conv1d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        ),
        nn.BatchNorm1d(out_channel),
        nn.GELU()
    )

class Res_CS_block_1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_CS_block_1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.dirate1_conv = depthwise_separable_conv1d(out_channels, out_channels, kernel_size=3)
        self.dirate3_conv = depthwise_separable_conv1d(out_channels, out_channels, kernel_size=5)
        self.dirate5_conv = depthwise_separable_conv1d(out_channels, out_channels, kernel_size=7)

        self.conv2 = nn.Conv1d(out_channels * 3, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.ca = CA1D(out_channels) 
        self.sa = SA1D()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)  # [batch, 100, 41]
        out = self.bn1(out)
        out = self.relu(out)
        out1 = self.dirate1_conv(out)
        out2 = self.dirate3_conv(out) 
        out3 = self.dirate5_conv(out) 
        out = self.conv2(torch.cat((out1, out2, out3), 1))
        out = self.bn2(out)
        out = self.ca(out) * out  
        out = self.sa(out) * out 
        out += residual
        out = self.relu(out)
        return out
        
class BiCrossAttention(nn.Module):
    def __init__(self, d_model=100, dropout=0.1, heads=4, layers=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            cross_block = nn.ModuleDict({
                "g2s": nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=False),
                "s2g": nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=False),
                "ng": nn.LayerNorm(d_model),
                "ns": nn.LayerNorm(d_model),
                "gate_g": nn.Linear(d_model * 2, d_model),
                "gate_s": nn.Linear(d_model * 2, d_model),
                "ffg": nn.Sequential(
                    nn.Linear(d_model, 2 * d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(2 * d_model, d_model)
                ),
                "ffs": nn.Sequential(
                    nn.Linear(d_model, 2 * d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(2 * d_model, d_model)
                ),
                "ng2": nn.LayerNorm(d_model),
                "ns2": nn.LayerNorm(d_model),
                "drop": nn.Dropout(dropout)
            })
            self.layers.append(cross_block)
    def forward(self, g_tokens, s_tokens):
        G = g_tokens.transpose(0, 1)
        S = s_tokens.transpose(0, 1)
        for block in self.layers:
            drug_to_protein, _ = block["g2s"](G, S, S)
            protein_to_drug, _ = block["s2g"](S, G, G)
            G_update = block["ng"](G + block["drop"](drug_to_protein))
            S_update = block["ns"](S + block["drop"](protein_to_drug))
            gate_drug = torch.sigmoid(block["gate_g"](torch.cat([G_update, G], dim=-1)))
            gate_prot = torch.sigmoid(block["gate_s"](torch.cat([S_update, S], dim=-1)))
            G = gate_drug * G_update + (1 - gate_drug) * G
            S = gate_prot * S_update + (1 - gate_prot) * S
            G = block["ng2"](G + block["drop"](block["ffg"](G)))
            S = block["ns2"](S + block["drop"](block["ffs"](S)))
        G = G.transpose(0, 1)
        S = S.transpose(0, 1)
        return G, S
        
class AttnFusion(nn.Module):
    def __init__(self, d_model=100, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.score = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, tokens):
        feat_transform = torch.tanh(self.proj(tokens))
        attn_weight = torch.softmax(self.score(feat_transform), dim=1)
        fused_feat = torch.sum(tokens * attn_weight, dim=1)
        fused_feat = self.drop(self.norm(fused_feat))
        return fused_feat
        
class abcmodel(nn.Module):
    def __init__(self, in_channels=64, feat_channels=100):
        super().__init__()
        self.emb_dim = 64
        self.max_len = 41
        self.embedding = Embedding(5, self.emb_dim, self.max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3,
                                                         norm=nn.LayerNorm(self.emb_dim))
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=50,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.feat_enhance = Res_CS_block_1D(in_channels, feat_channels)
        # self.feat_enhance = Res_CS_block_1D(in_channels, feat_channels)
        # self.dwconv = depthwise_separable_conv1d(64, 100, 1)
        self.bi_cross = BiCrossAttention(d_model=100, dropout=0.2, heads=4, layers=3)
        self.attn_fusion = AttnFusion(d_model=100, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(100, 64), nn.BatchNorm1d(64), nn.Dropout(0.3), nn.LeakyReLU(),
            nn.Linear(64, 2), nn.Softmax(dim=1))

    def forward(self, seqs):
        seqs_feature = self.embedding(seqs) 
        seqs_feature = self.transformer_encoder(seqs_feature)

        output1 = seqs_feature.permute(0, 2, 1)  
        output1 = self.feat_enhance(output1)  
        output1 = output1.permute(0, 2, 1) 

        output2, _ = self.bilstm(seqs_feature) 

        fused1, fused2 = self.bi_cross(output1, output2)

        g1 = torch.mean(fused1, dim=1) 
        g2 = torch.mean(fused2, dim=1)  
        global_feat = torch.stack([g1, g2], dim=1)  
        output = self.attn_fusion(global_feat)  
        output = self.classifier(output)
        return output
