import torch
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel


class PhoneCNNStack(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # [B,T,H] -> [B,H,T]
        x = x.transpose(1, 2)

        x = self.conv(x)

        # [B,H,T] -> [B,T,H]
        x = x.transpose(1, 2)

        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x


class PhoneRNNStack(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x, _ = self.bilstm(x)

        x = self.norm(x)
        x = self.dropout(x)

        return x


class PhoneticEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.cnn = PhoneCNNStack(hidden_dim)
        self.rnn = PhoneRNNStack(hidden_dim)

        # SAU
    def forward(self, x):
        residual = x
        x = self.cnn(x)
        x = self.rnn(x)
        return x + residual


class LinguisticEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(256, 64)
        self.pos_embedding = nn.Embedding(512,64)

        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            bidirectional=True,
            batch_first=True,
        )

        self.fc_k = nn.Linear(128, 768)
        self.fc_v = nn.Linear(128, 768)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.long()

        x = self.embedding(x)

        # SAU
        positions = torch.arange(
            x.size(1),
            device=x.device
        ).unsqueeze(0).expand(x.size(0), -1)  # explicit expand, tránh broadcasting ngầm
        
        x = x + self.pos_embedding(positions)

        o, _ = self.bilstm(x)

        key = self.fc_k(o)
        value = self.fc_v(o)

        return key, value


class PL(Wav2Vec2PreTrainedModel):
    def __init__(
        self,
        config,
        hidden_dim: int = 768,
        vocab_size: int = 123,
    ):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)

        self.phonetic_encoder = PhoneticEncoder(hidden_dim)

        self.linguistic_encoder = LinguisticEncoder()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=16,
            batch_first=True,
            kdim=768,
            vdim=768,
        )

        self.dropout = nn.Dropout(0.1)
        
        self.gate = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )
        
        self.classifier = nn.Linear(
            hidden_dim,
            vocab_size,
        )

        self.post_init()

        self._init_custom_weights()

    def _init_custom_weights(self):
        """
        Chỉ khởi tạo các layer custom.
        Không đụng tới Wav2Vec2 pretrained.
        """

        for module in [
            self.phonetic_encoder,
            self.linguistic_encoder,
            self.multihead_attn,
            self.gate,
            self.classifier,
        ]:
            for m in module.modules():

                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                elif isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(
                        m.weight,
                        nonlinearity="relu",
                    )

                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(
                        m.weight,
                        mean=0.0,
                        std=0.02,
                    )

                elif isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():

                        if "weight" in name:
                            nn.init.xavier_uniform_(param)

                        elif "bias" in name:
                            nn.init.zeros_(param)

    # SAU
    def freeze_feature_extractor(self, num_freeze_layers: int = 4):
        self.wav2vec2.feature_extractor._freeze_parameters()
    
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            if i < num_freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_values, linguistic):
        phonetic = self.wav2vec2(
            input_values,
            attention_mask=None,
        ).last_hidden_state

        phonetic = self.phonetic_encoder(phonetic)

        h_k, h_v = self.linguistic_encoder(linguistic)

        attn_output, _ = self.multihead_attn(
            query=phonetic,
            key=h_k,
            value=h_v,
        )
        
        attn_output = attn_output + phonetic
        
        gate = torch.sigmoid(
            self.gate(
                torch.cat(
                    [phonetic, attn_output],
                    dim=-1
                )
            )
        )
        
        fused = (
            gate * attn_output
            + (1.0 - gate) * phonetic
        )

        fused = self.dropout(fused)

        logits = self.classifier(fused)

        return logits