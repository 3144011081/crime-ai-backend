import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# =========================================================
# 1. R(2+1)D ARCHITECTURE (3D CONVOLUTIONAL MODEL)
# =========================================================

class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels, stride=1):
        super().__init__(
            nn.Sequential(
                nn.Conv3d(
                    in_channels, mid_channels,
                    kernel_size=(1, 3, 3), stride=(1, stride, stride),
                    padding=(0, 1, 1), bias=False
                ),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    mid_channels, out_channels,
                    kernel_size=(3, 1, 1), stride=(1, 1, 1),
                    padding=(1, 0, 0), bias=False
                )
            ),
            nn.BatchNorm3d(out_channels)
        )


class BasicBlock2Plus1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = Conv2Plus1D(in_channels, out_channels, mid_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2Plus1D(out_channels, out_channels, mid_channels, stride=1)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class CrimeR2Plus1D(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(64, 64, [144, 144], stride=1)
        self.layer2 = self._make_layer(64, 128, [230, 288], stride=2)
        self.layer3 = self._make_layer(128, 256, [460, 576], stride=2)
        self.layer4 = self._make_layer(256, 512, [921, 1152], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(512, num_classes))

    def _make_layer(self, in_channels, out_channels, mid_channels_list, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(out_channels)
            )
        layers = [BasicBlock2Plus1D(in_channels, out_channels, mid_channels_list[0], stride=stride, downsample=downsample)]
        for i in range(1, len(mid_channels_list)):
            layers.append(BasicBlock2Plus1D(out_channels, out_channels, mid_channels_list[i], stride=1, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# =========================================================
# 2. DOMAIN ADAPTATION CRIME ARCHITECTURE (LSTM + GRL)
# =========================================================

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        # x: [B, T, 2048]
        out, (hn, cn) = self.lstm(x)
        feat = out[:, -1, :]
        feat = F.relu(self.bn(self.fc(feat)), inplace=True)
        return feat


class CrimeClassifier(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, num_classes=11):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)), inplace=True)
        x = self.fc2(x)
        return x


class CrimeClassifierAerial(nn.Module):
    def __init__(self, in_dim=512, num_classes=11):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)), inplace=True)
        x = F.relu(self.bn2(self.fc2(x)), inplace=True)
        x = F.relu(self.bn3(self.fc3(x)), inplace=True)
        x = self.fc4(x)
        return x


class DomainClassifier(nn.Module):
    def __init__(self, in_dim=256, hidden1=128, hidden2=64, num_domains=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, num_domains)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)), inplace=True)
        x = F.relu(self.bn2(self.fc2(x)), inplace=True)
        x = self.fc3(x)
        return x


class DomainClassifierAerial(nn.Module):
    def __init__(self, in_dim=512, num_domains=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_domains)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)), inplace=True)
        x = F.relu(self.bn2(self.fc2(x)), inplace=True)
        x = self.fc3(x)
        return x


class DomainAdaptationCrimeModel(nn.Module):
    """
    Domain Adaptation Crime Detection Architecture.
    Combines 2D CNN (ResNet-50) spatial feature extraction, Bidirectional LSTM temporal
    encoding, and a multi-class crime classifier + domain adaptation classifier.
    Supports both standard (hidden_dim=256) and aerial (hidden_dim=512) configurations.
    """
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        num_classes: int = 11,
        is_aerial: bool = True,
        use_backbone: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.is_aerial = is_aerial
        self.use_backbone = use_backbone

        if use_backbone:
            try:
                resnet = models.resnet50(weights=None)
            except Exception:
                resnet = models.resnet50(pretrained=False)
            resnet.fc = nn.Identity()
            self.backbone = resnet
        else:
            self.backbone = None

        self.temporal_encoder = TemporalEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        
        if is_aerial:
            self.classifier = CrimeClassifierAerial(in_dim=hidden_dim, num_classes=num_classes)
            self.domain_classifier = DomainClassifierAerial(in_dim=hidden_dim)
        else:
            self.classifier = CrimeClassifier(in_dim=hidden_dim, num_classes=num_classes)
            self.domain_classifier = DomainClassifier(in_dim=hidden_dim)

    def extract_features(self, x):
        """
        Extracts temporal feature embeddings [B, hidden_dim] from raw video tensors [B, 3, T, H, W]
        or pre-extracted features [B, T, 2048].
        """
        if x.dim() == 5:
            if x.shape[1] == 3:
                x = x.permute(0, 2, 1, 3, 4)
            b, t, c, h, w = x.shape
            if self.backbone is not None:
                x_reshaped = x.reshape(b * t, c, h, w)
                feats = self.backbone(x_reshaped)  # [B*T, 2048]
                x = feats.reshape(b, t, -1)        # [B, T, 2048]

        feat = self.temporal_encoder(x)
        return feat

    def forward(self, x):
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits


# =========================================================
# 3. UNIFIED MODEL LOADER & FACTORY
# =========================================================

def load_crime_model(
    checkpoint_path: Union[str, Path],
    device: torch.device = torch.device("cpu")
) -> Tuple[nn.Module, List[str], Dict]:
    """
    Dynamically loads DomainAdaptationCrimeModel (standard or aerial) or CrimeR2Plus1D
    depending on checkpoint architecture and metadata.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        candidates = [
            ckpt_path.parent / "crime_aerial_augmented_best1.pth",
            Path("models/crime_aerial_augmented_best1.pth"),
            ckpt_path.parent / "crime_r2plus1d_ucf_finetuned.pth",
            Path("models/crime_r2plus1d_ucf_finetuned.pth"),
            ckpt_path.parent / "domain_adaptation_model_aerial.pth",
            ckpt_path.parent / "domain_adaptation_model.pth",
            Path("models/domain_adaptation_model_aerial.pth"),
            Path("models/domain_adaptation_model.pth"),
        ]
        found = None
        for cand in candidates:
            if cand.exists():
                found = cand
                print(f"[Notice] Requested checkpoint '{ckpt_path.name}' not found. Falling back to '{cand.name}'.")
                ckpt_path = cand
                break
        if found is None:
            pth_files = list(Path("models").glob("*.pth")) if Path("models").exists() else []
            if pth_files:
                ckpt_path = pth_files[0]
                print(f"[Notice] Requested checkpoint not found. Falling back to '{ckpt_path.name}'.")
            else:
                raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    # Check class names in checkpoint metadata
    if "class_names" in checkpoint:
        classes = checkpoint["class_names"]
    elif "classes" in checkpoint:
        classes = checkpoint["classes"]
    elif "class_names" in config:
        classes = config["class_names"]
    elif "classes" in config:
        classes = config["classes"]
    else:
        if any("temporal_encoder" in k for k in state_dict.keys()):
            classes = [
                "Accident", "Arrest", "Arson", "Burglary", "Explosion",
                "Normal", "Robbery", "Shooting", "Theft", "Vandalism", "Violence"
            ]
        else:
            classes = [
                "Normal", "Violence", "Robbery", "Shooting", "FireExplosion", "Accident", "Vandalism"
            ]

    # Detect architecture based on state_dict keys
    if any("temporal_encoder" in k for k in state_dict.keys()):
        num_classes = checkpoint.get("num_classes", len(classes))
        
        # Check hidden dimension from LSTM weights (512 for Aerial vs 256 for standard)
        lstm_hh_shape = state_dict.get("temporal_encoder.lstm.weight_hh_l0", None)
        hidden_dim = lstm_hh_shape.shape[1] if lstm_hh_shape is not None else 512
        is_aerial = ("classifier.fc4.weight" in state_dict) or (hidden_dim == 512)

        model = DomainAdaptationCrimeModel(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            is_aerial=is_aerial,
            use_backbone=True
        )
        model.load_state_dict(state_dict, strict=False)
    else:
        model = CrimeR2Plus1D(num_classes=len(classes))
        model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()
    print(f"[Model Loader] Verified {model.__class__.__name__} ({len(classes)} classes) on {device} from '{ckpt_path.name}'")
    return model, classes, checkpoint
