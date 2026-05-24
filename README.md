# Azərbaycan Dili üçün ASR Pipeline

Açıq mənbəli Whisper modelləri əsasında Azərbaycan dili üçün avtomatik nitq tanıma (ASR) pipeline-ı. Hissə A zero-shot inferens, Hissə B isə fine-tuning cəhdini əhatə edir.

---

## İstifadə olunan modellər və parametrlər

| | Hissə A | Hissə B |
|---|---|---|
| **Model** | openai/whisper-large-v3 | openai/whisper-small |
| **Dataset** | google/fleurs [az_az] | google/fleurs [az_az] |
| **Nümunə sayı** | 100 (test) | 200 train / 50 val / 100 test |
| **Epochs** | — | 10 |
| **Learning rate** | — | 1e-5 |
| **Batch size** | 8 | 8 |
| **Early stopping** | — | patience=3 |
| **Device** | CUDA (T4) | CUDA (T4) |

---

## WER/CER nəticələri

### Hissə A — whisper-large-v3 (zero-shot)

| Metrika | Nəticə |
|---|---|
| Corpus WER | 19.86% |
| Corpus CER | 4.76% |
| Ortalama WER | 19.25% |
| Ortalama CER | 4.49% |

### Hissə B — Baseline vs Fine-tuned müqayisəsi

| Model | WER | CER |
|---|---|---|
| whisper-large-v3 (zero-shot) | 19.86% | 4.76% |
| whisper-small (fine-tuned, 200 nümunə) | 44.01% | 13.68% |

---

## Setup və işə salma

```bash
# 1. Asılılıqları qur
pip install -r requirements.txt

# Hissə A
# 1. Kaggle.com → "Create New Notebook" → File → Import Notebook
# 2. part_a/part_a.ipynb faylını yüklə
# 3. Settings → Accelerator → GPU T4 x2 seç
# 4. Run All

# Hissə B (eyni addımlar, part_b/part_b.ipynb ilə)
# 1. Kaggle.com → "Create New Notebook" → File → Import Notebook  
# 2. part_b/part_b.ipynb faylını yüklə
# 3. Settings → Accelerator → GPU T4 x2 seç
# 4. Run All 
```


---

## Fine-tuning nəticəsinin müqayisəsi

Fine-tuned whisper-small modeli baseline whisper-large-v3-dən yüksək WER göstərdi. Bunun əsas səbəbləri: model ölçüsündəki böyük fərq (242M vs 1543M parametr) və məhdud training datası (200 nümunə). Validation loss overfitting olmadan ardıcıl azaldı; early stopping step 110-da ən yaxşı checkpointi qorudu. Daha böyük data ilə fine-tuned whisper-large-v3 əhəmiyyətli performans artımı verər.

---

## Layihə strukturu

```
az-stt-intern/
├── README.md
├── part_a/          ← zero-shot inferens kodu
├── part_b/          ← fine-tuning kodu
├── results/         ← WER/CER cədvəlləri, qrafiklər
├── report.pdf       ← Hissə C analitik hesabat
└── requirements.txt
```
