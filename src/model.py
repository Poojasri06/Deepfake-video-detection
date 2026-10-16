# src/model.py
"""Single source of truth for the trained CNN + LSTM architecture.

This mirrors ``DeepfakeDetector`` in ``src/train.py`` exactly (same layers, same
parameter names), so checkpoints produced by ``train.py`` load with strict key
matching. Do not change layer shapes or names here without retraining.
"""
import torch.nn as nn
from torchvision import models

# Class order used during training (CrossEntropy targets): 0 = REAL, 1 = FAKE.
CLASS_NAMES = ("real", "fake")
REAL_INDEX = 0
FAKE_INDEX = 1


class DeepfakeDetector(nn.Module):
    """EfficientNet-B0 frame encoder + BiLSTM temporal model + 2-class head.

    forward input : (B, T, C, H, W)
    forward output: logits (B, 2); softmax over them gives [P(real), P(fake)].

    ``pretrained=True`` downloads ImageNet weights (training only). Inference
    must use ``pretrained=False``: the checkpoint overwrites every weight, so no
    download is needed.

    Note: the head reads ``lstm_out[:, -1, :]``. For a bidirectional LSTM this
    includes the backward state after seeing only the last frame. That is a
    property of the trained weights and is kept for checkpoint compatibility.
    """

    def __init__(self, cnn_out_dim=256, lstm_hidden=128, bidirectional=True,
                 pretrained=False):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        if isinstance(backbone.classifier, nn.Sequential):
            in_feats = backbone.classifier[1].in_features
            backbone.classifier[1] = nn.Linear(in_feats, cnn_out_dim)
        else:
            in_feats = backbone.classifier.in_features
            backbone.classifier = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(in_feats, cnn_out_dim)
            )

        self.cnn = backbone
        self.lstm = nn.LSTM(input_size=cnn_out_dim,
                            hidden_size=lstm_hidden,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(lstm_hidden * (2 if bidirectional else 1), 2)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feats = self.cnn(x)
        feats = feats.reshape(B, T, -1)
        out, _ = self.lstm(feats)
        return self.fc(out[:, -1, :])
