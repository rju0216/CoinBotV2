"""DL 모델 클래스 정의. 학습과 추론 양쪽에서 import.

LSTMClassifier: 순차 시퀀스 처리, ~40K 파라미터
TransformerClassifier: Attention 기반, ~80K 파라미터
둘 다 CPU 추론이 밀리초 단위로 가능한 경량 설계.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """경량 LSTM 분류기.

    Input(seq_len, n_features) → LSTM(hidden) → Dropout
    → Dense(32, ReLU) → Dense(num_classes)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_classes: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            logits: (batch, num_classes) — softmax는 loss/추론에서 처리
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        return self.fc2(out)


class PositionalEncoding(nn.Module):
    """사인/코사인 위치 인코딩."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        # 홀수 d_model 대응: sin/cos 슬라이스 크기에 맞게 div_term 조정
        n_sin = len(pe[:, 0::2][0])  # ceil(d_model / 2)
        n_cos = len(pe[:, 1::2][0])  # floor(d_model / 2)
        pe[:, 0::2] = torch.sin(position * div_term[:n_sin])
        pe[:, 1::2] = torch.cos(position * div_term[:n_cos])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    """경량 Transformer 분류기.

    Input(seq_len, n_features) → Linear(→ d_model)
    → PositionalEncoding → TransformerEncoder
    → GlobalAveragePooling → Dense(32, ReLU) → Dense(num_classes)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.3,
        num_classes: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            logits: (batch, num_classes)
        """
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global Average Pooling
        x = self.relu(self.fc1(x))
        return self.fc2(x)
