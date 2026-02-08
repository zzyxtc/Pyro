import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, dropout=0.3):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return out, h_n, c_n


class Decoder(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, output_size=1, num_layers=2, dropout=0.3):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out.unsqueeze(-1), hidden


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_size, cnn_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, cnn_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x.permute(0, 2, 1)


class FeatureGroupAttention(nn.Module):
    def __init__(self, input_size, group_sizes, temperature_idx=0):
        super().__init__()
        self.group_sizes = group_sizes
        self.num_groups = len(group_sizes)
        self.temperature_idx = temperature_idx

        assert sum(group_sizes) == input_size, f"Group sizes sum {sum(group_sizes)} != input_size {input_size}"

        # 每个分组的注意力权重学习
        self.group_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(size, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            ) for size in group_sizes
        ])

        # 确保embed_dim能被num_heads整除
        embed_dim = sum(group_sizes)
        num_heads = min(3, embed_dim)

        if embed_dim % num_heads != 0:
            for nh in range(num_heads, 0, -1):
                if embed_dim % nh == 0:
                    num_heads = nh
                    break

        self.cross_group_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        features_split = torch.split(x, self.group_sizes, dim=-1)

        # 计算每个分组的注意力权重
        group_weights = []
        weighted_features = []

        for i, (group_feat, attention) in enumerate(zip(features_split, self.group_attentions)):
            group_mean = group_feat.mean(dim=1, keepdim=True)  # (batch, 1, group_size)
            weight = attention(group_mean)  # (batch, 1, 1)
            group_weights.append(weight)
            weighted_group = group_feat * weight
            weighted_features.append(weighted_group)

        # 拼接加权后的特征
        weighted_concat = torch.cat(weighted_features, dim=-1)

        # 跨组注意力
        cross_weighted, _ = self.cross_group_attention(
            weighted_concat, weighted_concat, weighted_concat
        )

        final_features = weighted_concat + 0.1 * cross_weighted

        return final_features


class EnhancedCNNSeq2Seq(nn.Module):
    def __init__(self, input_size=21, hidden_size=128, cnn_channels=64,
                 num_layers=2, dropout=0.3, pred_length=30):
        super().__init__()
        self.pred_length = pred_length
        self.temperature_idx = 0

        self.group_sizes = [5, 7, 7, 2]
        assert sum(self.group_sizes) == input_size, "Group sizes must sum to input_size"

        self.feature_attention = FeatureGroupAttention(input_size, self.group_sizes,
                                                       temperature_idx=self.temperature_idx)

        self.cnn = CNNFeatureExtractor(input_size, cnn_channels, kernel_size=3)
        encoder_input_size = cnn_channels

        self.encoder = Encoder(
            input_size=encoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        decoder_input_size = cnn_channels
        self.decoder = Decoder(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
            output_size=1,
            num_layers=num_layers,
            dropout=dropout
        )

        if decoder_input_size != input_size:
            self.decoder_input_proj = nn.Linear(input_size, decoder_input_size)
        else:
            self.decoder_input_proj = None

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape

        attended_features = self.feature_attention(x)
        cnn_features = self.cnn(attended_features)
        encoder_input = cnn_features
        encoder_output, h_n, c_n = self.encoder(encoder_input)

        h_n = h_n.view(self.encoder.lstm.num_layers, 2, batch_size, -1)
        c_n = c_n.view(self.encoder.lstm.num_layers, 2, batch_size, -1)

        h_n = torch.mean(h_n, dim=1)
        c_n = torch.mean(c_n, dim=1)

        if self.decoder_input_proj is not None:
            current_input = self.decoder_input_proj(x[:, -1:, :])
        else:
            current_input = encoder_input[:, -1:, :]

        outputs = []

        for _ in range(self.pred_length):
            output, (h_n, c_n) = self.decoder(current_input, (h_n, c_n))
            outputs.append(output)

            if self.decoder_input_proj is not None:
                next_input = current_input.clone()
                next_input = current_input
            else:
                next_input = current_input.clone()

            current_input = next_input
        return torch.cat(outputs, dim=1)
