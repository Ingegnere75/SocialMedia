# Multimodal Social Media Stance/Sentiment

Sistema multimodale per classificare stance (pro/contro) dei post sui social usando testo, immagini (opzionali) e feature numeriche.

## 🚀 Avvio rapido su Google Colab

1. Caricare lo ZIP del progetto in Colab e avviare:
```python
from google.colab import files
uploaded = files.upload()

!rm -rf /content/proj
!mkdir -p /content/proj

import glob, os
zip_name = next(iter(globals().get('uploaded', {'SocialMediaPost.zip':None}).keys()))
!unzip -q "$zip_name" -d /content/proj

roots = [os.path.dirname(p) for p in glob.glob("/content/proj/**/src", recursive=True) if os.path.isdir(p)]
PROJECT = roots[0]
%cd $PROJECT
!ls -la
```

2. Installare dipendenze:
```python
!pip install emoji
!pip install -r /content/proj/SocialMediaPost/requirements.txt
```

3. Training rapido:
```python
!python -m src.train   --csv_path data/processed/train/aborto_train.csv   --num_classes 2   --epochs 1   --batch_size 16   --val_split 0.2   --save_dir checkpoints
```

4. Predizione su un testo singolo:
```python
!python -m src.predict --checkpoint checkpoints/best.pt   --text "Questo è un nuovo tweet di esempio su aborto e opinioni contrastanti." --topk 2
```

---

## 📦 Struttura del progetto

```
project-root/
├─ src/
├─ data/
├─ checkpoints/
├─ predictions/
├─ requirements.txt
└─ README.md
```

## 🧰 Requisiti principali
- Python 3.10+
- PyTorch, Transformers, Torchvision, scikit-learn, pandas, emoji, matplotlib

## 🗂️ Dataset
Origine: Hugging Face → TweetEval (stance_abortion).

## 📝 Licenza
Rilasciato sotto licenza MIT. Vedi file LICENSE.
