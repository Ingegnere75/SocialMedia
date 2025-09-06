import math
from typing import Optional, Literal, Dict

import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision import models


# ========================= Utils =========================

def freeze_module(module: nn.Module, freeze: bool = True):
    if freeze:
        for p in module.parameters():
            p.requires_grad = False


def build_default_backbone() -> nn.Module:
    """
    ResNet50 pre-addestrata con il layer FC sostituito da Identity.
    Output feature dim = 2048 (compatibile con ImageEncoder(out_dim=2048)).
    """
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    return resnet


def xavier_uniform_(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ========================= Encoders (moderni) =========================

class TextEncoder(nn.Module):
    """
    Encoder testuale basato su Transformer.
    - pooling: 'cls' usa il primo token; 'mean' fa mean-pooling sui token validi (mask).
    - proj_dim: proiezione finale nello spazio comune.
    """
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
        proj_dim: int = 768,
        fine_tune: bool = True,
        pooling: Literal["cls", "mean"] = "cls",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        hid = self.transformer.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hid, proj_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        freeze_module(self.transformer, not fine_tune)
        self.apply(xavier_uniform_)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == "cls":
            tok = out.last_hidden_state[:, 0, :]  # [B, H]
        else:
            # mean-pooling mascherato
            last = out.last_hidden_state  # [B, T, H]
            mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
            last = last * mask
            denom = mask.sum(dim=1).clamp(min=1e-6)
            tok = last.sum(dim=1) / denom  # [B, H]
        z = self.proj(tok)
        return z


class ImageEncoder(nn.Module):
    """
    Encoder immagini:
    - backbone pre-addestrato (default: ResNet50 con fc=Identity)
    - proiezione nello spazio comune
    """
    def __init__(self, backbone: nn.Module, out_dim: int = 2048, proj_dim: int = 512, fine_tune: bool = False, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        freeze_module(self.backbone, not fine_tune)
        self.proj = nn.Sequential(
            nn.Linear(out_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.apply(xavier_uniform_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # [B, out_dim]
        z = self.proj(feats)
        return z


class NumericMLP(nn.Module):
    """
    Encoder per feature numeriche robuste a batch piccoli:
    - usa LayerNorm (no BatchNorm → nessun errore con batch_size=1)
    - residual + dropout per stabilità
    """
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        if in_dim <= 0:
            # placeholder per compat, non verrà chiamato
            self.layer1 = nn.Identity()
            self.layer2 = nn.Identity()
            self.res_proj = nn.Identity()
        else:
            self.layer1 = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden),
                nn.Dropout(dropout),
            )
            self.layer2 = nn.Sequential(
                nn.Linear(hidden, out_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(out_dim),
                nn.Dropout(dropout),
            )
            self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            self.apply(xavier_uniform_)

        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_dim <= 0:
            # ritorna vettore a dimensione zero (verrà gestito dal classifier)
            return torch.zeros((x.size(0), 0), device=x.device, dtype=x.dtype)
        h = self.layer1(x)
        h = self.layer2(h)
        return h + self.res_proj(x)  # residual


# ========================= Fusion & Classifier (moderno) =========================

class FusionClassifier(nn.Module):
    """
    Testa di classificazione:
    - concatena (testo, immagine, numerico) con gating per gestire input mancanti
    - MLP con residual, LayerNorm e dropout
    """
    def __init__(self, text_dim: int, img_dim: int, num_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        in_dim = text_dim + img_dim + num_dim
        self.gate_text = nn.Parameter(torch.tensor(1.0))
        self.gate_img = nn.Parameter(torch.tensor(1.0))
        self.gate_num = nn.Parameter(torch.tensor(1.0))

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.res_proj = nn.Linear(in_dim, num_classes)  # residual diretto verso i logits
        self.apply(xavier_uniform_)

        self.img_dim = img_dim
        self.num_dim = num_dim

    def forward(self, z_text: torch.Tensor, z_img: torch.Tensor, z_num: torch.Tensor) -> torch.Tensor:
        B = z_text.size(0)
        device = z_text.device
        if z_img is None and self.img_dim > 0:
            z_img = torch.zeros((B, self.img_dim), dtype=z_text.dtype, device=device)
        if z_num is None and self.num_dim > 0:
            z_num = torch.zeros((B, self.num_dim), dtype=z_text.dtype, device=device)

        z = torch.cat([
            self.gate_text * z_text,
            self.gate_img * (z_img if z_img is not None else torch.zeros((B, 0), device=device, dtype=z_text.dtype)),
            self.gate_num * (z_num if z_num is not None else torch.zeros((B, 0), device=device, dtype=z_text.dtype)),
        ], dim=1)
        logits = self.mlp(z) + self.res_proj(z)  # residual sui logits
        return logits


# ========================= Modello Multimodale (moderno) =========================

class MultimodalModel(nn.Module):
    """
    Modello multimodale moderno:
    - rami: testo (obbligatorio), immagini (opzionale), numerico (opzionale)
    - fallback a zeri se immagini o numeri sono assenti
    """
    def __init__(
        self,
        text_model_name: str,
        num_numeric: int,
        num_classes: int,
        text_proj: int = 768,
        img_proj: int = 512,
        num_proj: int = 64,
        freeze_text: bool = False,
        freeze_image: bool = True,
        backbone: Optional[nn.Module] = None,
        text_pooling: Literal["cls", "mean"] = "cls",
        dropout: float = 0.1,
    ):
        super().__init__()

        # Text
        self.text = TextEncoder(
            model_name=text_model_name,
            proj_dim=text_proj,
            fine_tune=not freeze_text,
            pooling=text_pooling,
            dropout=dropout,
        )

        # Image (opzionale)
        self.image = None
        self.img_proj_dim = img_proj
        if backbone is not None:
            self.image = ImageEncoder(backbone, out_dim=2048, proj_dim=img_proj, fine_tune=not freeze_image, dropout=dropout)

        # Numeric (può essere 0 → disattivato)
        self.num_proj_dim = num_proj if num_numeric > 0 else 0
        if num_numeric > 0:
            self.numeric = NumericMLP(num_numeric, hidden=max(64, num_proj * 2), out_dim=num_proj, dropout=dropout)
        else:
            self.numeric = None

        # Classifier
        img_dim = img_proj if self.image is not None else 0
        num_dim = self.num_proj_dim
        self.classifier = FusionClassifier(text_dim=text_proj, img_dim=img_dim, num_dim=num_dim, num_classes=num_classes, hidden=max(256, text_proj), dropout=dropout)

    def _zeros(self, bsz: int, dim: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((bsz, dim), dtype=torch.float32, device=device)

    def forward(self, enc_text: Dict[str, torch.Tensor], images: Optional[torch.Tensor], nums: Optional[torch.Tensor]) -> torch.Tensor:
        device = enc_text["input_ids"].device
        bsz = enc_text["input_ids"].size(0)

        # Text
        zt = self.text(enc_text["input_ids"], enc_text["attention_mask"])

        # Image
        if self.image is not None and images is not None and images.numel() > 0:
            zi = self.image(images)
        else:
            zi = self._zeros(bsz, self.img_proj_dim if self.image is not None else 0, device)

        # Numeric
        if self.numeric is not None and nums is not None and nums.numel() > 0:
            zn = self.numeric(nums)
        else:
            zn = self._zeros(bsz, self.num_proj_dim, device) if self.num_proj_dim > 0 else self._zeros(bsz, 0, device)

        logits = self.classifier(zt, zi, zn)
        return logits


# ========================= LEGACY (compatibilità checkpoint vecchi) =========================
# Replica esatta dei nomi modulo usati nel training vecchio + opzione di forzare le dimensioni
# in modo da far combaciare lo state_dict (risolve mismatch tipo 1344 vs 1280).

class LegacyTextEncoder(nn.Module):
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment", proj_dim=768, fine_tune=True):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hid = self.transformer.config.hidden_size
        self.proj = nn.Linear(hid, proj_dim)  # chiave: text.proj.weight
        self.act = nn.Tanh()
        if not fine_tune:
            for p in self.transformer.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.act(self.proj(cls))


class LegacyImageEncoder(nn.Module):
    def __init__(self, backbone, out_dim=2048, proj_dim=512, fine_tune=False):
        super().__init__()
        self.backbone = backbone
        if not fine_tune and self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(out_dim, proj_dim)  # chiave: image.proj.weight
        self.act = nn.ReLU()

    def forward(self, x):
        feats = self.backbone(x) if (self.backbone is not None and x is not None) else None
        if feats is None:
            return None
        return self.act(self.proj(feats))


class LegacyNumericMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class LegacyFusionClassifier(nn.Module):
    """
    Usa le dimensioni *forzate* per garantire che il primo Linear riceva input esattamente della dimensione attesa dal checkpoint.
    """
    def __init__(self, text_dim, img_dim, num_dim, num_classes, hidden=512, dropout=0.2):
        super().__init__()
        self.img_dim = int(img_dim)
        self.num_dim = int(num_dim)
        in_dim = int(text_dim) + self.img_dim + self.num_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, z_text, z_img, z_num):
        B = z_text.size(0)
        device = z_text.device
        if (z_img is None or (hasattr(z_img, "numel") and z_img.numel() == 0)) and self.img_dim > 0:
            z_img = torch.zeros((B, self.img_dim), device=device, dtype=z_text.dtype)
        if (z_num is None or (hasattr(z_num, "numel") and z_num.numel() == 0)) and self.num_dim > 0:
            z_num = torch.zeros((B, self.num_dim), device=device, dtype=z_text.dtype)

        parts = [z_text]
        if self.img_dim > 0:
            parts.append(z_img)
        if self.num_dim > 0:
            parts.append(z_num)
        z = torch.cat(parts, dim=1) if len(parts) > 1 else z_text
        return self.head(z)


class LegacyMultimodalModel(nn.Module):
    """
    Versione legacy con nomi layer identici al vecchio training.
    Parametri force_* permettono di *forzare* le dimensioni di img/num nel classifier per combaciare lo state_dict.
    """
    def __init__(self, text_model_name, num_numeric, num_classes,
                 text_proj=768, img_proj=512, num_proj=64,
                 freeze_text=False, freeze_image=True, backbone=None,
                 force_img_dim: Optional[int] = None,
                 force_num_dim: Optional[int] = None):
        super().__init__()
        self.text = LegacyTextEncoder(text_model_name, proj_dim=text_proj, fine_tune=not freeze_text)
        self.image = LegacyImageEncoder(backbone, out_dim=2048, proj_dim=img_proj, fine_tune=not freeze_image) if backbone is not None else LegacyImageEncoder(None, 2048, img_proj, fine_tune=False)
        self.numeric = LegacyNumericMLP(num_numeric, hidden=64, out_dim=num_proj) if (num_numeric and num_numeric > 0) else None

        # Dimensioni del classifier: forzabili per matchare esattamente il checkpoint
        img_dim = force_img_dim if force_img_dim is not None else (img_proj if backbone is not None else 0)
        num_dim = force_num_dim if force_num_dim is not None else (num_proj if self.numeric is not None else 0)
        self.classifier = LegacyFusionClassifier(text_proj, img_dim, num_dim, num_classes)

    def forward(self, enc_text, images, nums):
        zt = self.text(enc_text["input_ids"], enc_text["attention_mask"])

        zi = None
        if images is not None and hasattr(images, "numel") and images.numel() > 0 and self.image.backbone is not None:
            zi = self.image(images)

        zn = None
        if self.numeric is not None and nums is not None and hasattr(nums, "numel") and nums.numel() > 0:
            zn = self.numeric(nums)

        return self.classifier(zt, zi, zn)


# Alias per compatibilità con import esistenti
SocialClassifier = MultimodalModel
