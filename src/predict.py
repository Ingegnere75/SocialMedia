import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

# Import robusto
try:
    from src.model import SocialClassifier, build_default_backbone
except Exception:
    from src.model import MultimodalModel as SocialClassifier  # type: ignore

    def build_default_backbone():
        raise RuntimeError("build_default_backbone non disponibile: aggiorna src/model.py.")


# ------------------------- Utils -------------------------

def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ascii_bar(p: float, width: int = 24) -> str:
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


def short_summary(text: str, max_words: int = 24) -> str:
    toks = str(text).strip().split()
    if len(toks) <= max_words:
        return str(text).strip()
    return " ".join(toks[:max_words]) + " …"


def explain_label(label_idx: int, label_map: Optional[List[str]]) -> str:
    name = label_map[label_idx] if (label_map and 0 <= label_idx < len(label_map)) else str(label_idx)
    low = str(name).lower()
    if low in {"0", "neg", "negative", "contrario", "contro"}:
        return f"Il testo è classificato come **{name}** (tono critico/negativo)."
    if low in {"1", "pos", "positive", "favorevole", "pro"}:
        return f"Il testo è classificato come **{name}** (tono favorevole/positivo)."
    return f"Assegnato alla **classe {name}** in base alle caratteristiche linguistiche rilevate."


def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"]
    args = ckpt.get("args", {}) or {}
    label_map = ckpt.get("label_map", None)
    return state, args, label_map


def has_image_branch(state_dict_keys: List[str]) -> bool:
    return any(k.startswith("image.") for k in state_dict_keys)


def is_legacy_state(state_dict_keys: List[str]) -> bool:
    return any(k.startswith("classifier.head.") for k in state_dict_keys) or (
        "text.proj.weight" in state_dict_keys and not any(k.startswith("text.proj.0.") for k in state_dict_keys)
    )


def infer_legacy_dims_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    dims = {
        "text_proj": 768,
        "img_proj": 0,
        "num_proj": 0,
        "num_numeric": 0,
        "force_img_dim": 0,
        "force_num_dim": 0,
    }
    if "text.proj.weight" in state:
        dims["text_proj"] = int(state["text.proj.weight"].shape[0])
    if "image.proj.weight" in state:
        dims["img_proj"] = int(state["image.proj.weight"].shape[0])
        dims["force_img_dim"] = dims["img_proj"]
    if "numeric.net.3.weight" in state:
        dims["num_proj"] = int(state["numeric.net.3.weight"].shape[0])
        dims["force_num_dim"] = dims["num_proj"]
    if "numeric.net.0.weight" in state:
        dims["num_numeric"] = int(state["numeric.net.0.weight"].shape[1])
    return dims


def build_model_from_args(args_dict: Dict, state_dict: Dict[str, torch.Tensor], device: torch.device) -> SocialClassifier:
    text_model = args_dict.get("text_model", "cardiffnlp/twitter-roberta-base-sentiment")
    num_classes = int(args_dict.get("num_classes", 2))
    freeze_text = str(args_dict.get("freeze_text", "true")).lower() == "true"
    freeze_image = str(args_dict.get("freeze_image", "true")).lower() == "true"
    text_pooling = args_dict.get("text_pooling", "cls")
    dropout = float(args_dict.get("dropout", 0.1))

    keys = list(state_dict.keys())
    legacy = is_legacy_state(keys)
    use_image = has_image_branch(keys)
    backbone = build_default_backbone() if use_image else None

    if legacy:
        from src.model import LegacyMultimodalModel  # type: ignore
        dims = infer_legacy_dims_from_state(state_dict)
        model = LegacyMultimodalModel(
            text_model_name=text_model,
            num_numeric=dims["num_numeric"],
            num_classes=num_classes,
            text_proj=dims["text_proj"],
            img_proj=dims["img_proj"],
            num_proj=dims["num_proj"],
            freeze_text=freeze_text,
            freeze_image=freeze_image,
            backbone=backbone,
            force_img_dim=dims["force_img_dim"],
            force_num_dim=dims["force_num_dim"],
        ).to(device)
    else:
        text_proj = int(args_dict.get("text_proj", 768))
        img_proj = int(args_dict.get("img_proj", 512))
        num_proj = int(args_dict.get("num_proj", 64))
        num_numeric = int(args_dict.get("num_numeric", 0))
        model = SocialClassifier(
            text_model_name=text_model,
            num_numeric=num_numeric,
            num_classes=num_classes,
            text_proj=text_proj,
            img_proj=img_proj,
            num_proj=num_proj,
            freeze_text=freeze_text,
            freeze_image=freeze_image,
            backbone=backbone,
            text_pooling=text_pooling,
            dropout=dropout,
        ).to(device)
    return model


# ------------------------- Collate "text-only" (niente dataset.collate_fn) -------------------------

def text_only_collate(tokenizer: AutoTokenizer, texts: List[str], device: torch.device, max_length: int = 128):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    imgs = None  # Niente immagini in predict testuale
    nums = None  # Niente feature numeriche in predict testuale
    return enc, imgs, nums


# ------------------------- Inference Core -------------------------

@torch.no_grad()
def forward_batch(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 128,
) -> np.ndarray:
    enc, imgs, nums = text_only_collate(tokenizer, texts, device, max_length=max_length)

    with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16, enabled=True):
        logits = model(enc, imgs, nums)

    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    return probs


def predict_texts(
    texts: List[str],
    ckpt_path: str,
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 16,
) -> Tuple[List[Dict], List[str]]:
    state, args_dict, label_map = load_checkpoint(ckpt_path, device)
    label_map = label_map if (label_map and isinstance(label_map, (list, tuple))) else None

    tokenizer = AutoTokenizer.from_pretrained(args_dict.get("text_model", "cardiffnlp/twitter-roberta-base-sentiment"))
    model = build_model_from_args(args_dict, state, device)
    model.load_state_dict(state, strict=False)
    model.eval()

    results: List[Dict] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="predict", leave=False):
        chunk = texts[start : start + batch_size]
        probs = forward_batch(model, tokenizer, chunk, device, max_length=max_length)

        for t, p in zip(chunk, probs):
            pred_idx = int(p.argmax())
            if label_map and pred_idx < len(label_map):
                pred_label = label_map[pred_idx]
                class_names = label_map
            else:
                pred_label = str(pred_idx)
                class_names = [str(i) for i in range(len(p))]

            results.append(
                {
                    "text": t,
                    "summary": short_summary(t),
                    "pred_idx": pred_idx,
                    "pred_label": pred_label,
                    "probs": p.tolist(),
                    "class_names": class_names,
                    "explanation": explain_label(pred_idx, label_map),
                }
            )
    return results, (label_map or [str(i) for i in range(len(results[0]["probs"]))])


def pretty_print(results: List[Dict], class_names: List[str], topk: int = 2):
    for i, r in enumerate(results, 1):
        print(f"\n=== ESEMPIO {i} ===")
        print(f"Testo (sintesi): {r['summary']}")
        print(f"Predizione: {r['pred_label']} (idx={r['pred_idx']})")

        probs = np.array(r["probs"])
        order = probs.argsort()[::-1]
        print("Probabilità per classe (top-k):")
        for j in order[: topk if topk > 0 else len(order)]:
            cname = class_names[j] if j < len(class_names) else str(j)
            p = float(probs[j])
            bar = ascii_bar(p)
            print(f"  - {cname:>10s}: {p:6.3f}  {bar}")

        print(f"Spiegazione: {r['explanation']}")


# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Inference per modello multimodale (solo testo o CSV con colonna 'text').")
    ap.add_argument("--checkpoint", type=str, required=True, help="Percorso a checkpoints/<run_name>/best.pt")
    ap.add_argument("--text", type=str, default=None, help="Testo singolo da classificare")
    ap.add_argument("--csv_path", type=str, default=None, help="CSV con colonna 'text' da classificare in batch")
    ap.add_argument("--out_csv", type=str, default=None, help="(Opzionale) salva le predizioni in CSV")
    ap.add_argument("--out_json", type=str, default=None, help="(Opzionale) salva le predizioni in JSON")
    ap.add_argument("--max_length", type=int, default=128, help="Lunghezza massima per il tokenizer")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size per inference su CSV")
    ap.add_argument("--topk", type=int, default=2, help="Quante classi mostrare come top-k in stampa")
    args = ap.parse_args()

    if (args.text is None) and (args.csv_path is None):
        raise SystemExit("Fornisci --text \"...\" oppure --csv_path file.csv")

    device = device_auto()

    if args.text is not None:
        texts = [args.text]
    else:
        df = pd.read_csv(args.csv_path)
        if "text" not in df.columns:
            raise ValueError("Il CSV deve contenere la colonna 'text'.")
        texts = df["text"].astype(str).tolist()

    results, class_names = predict_texts(
        texts,
        ckpt_path=args.checkpoint,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    pretty_print(results if args.text is not None else results[:3], class_names, topk=args.topk)

    out_rows = []
    for r in results:
        row = {
            "text": r["text"],
            "summary": r["summary"],
            "pred_idx": r["pred_idx"],
            "pred_label": r["pred_label"],
            "explanation": r["explanation"],
        }
        for i, p in enumerate(r["probs"]):
            cname = class_names[i] if i < len(class_names) else f"class_{i}"
            row[f"prob_{cname}"] = float(p)
        out_rows.append(row)

    if args.out_csv:
        out_df = pd.DataFrame(out_rows)
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out_csv, index=False)
        print(f"\n[INFO] Predizioni salvate in CSV: {args.out_csv}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out_rows, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Predizioni salvate in JSON: {args.out_json}")


if __name__ == "__main__":
    main()
