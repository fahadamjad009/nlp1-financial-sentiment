# 📈 NLP1 — Financial Sentiment Analyser

> Fine-tuned **DistilBERT** transformer on 9,543 financial news headlines for 3-class sentiment classification — Bearish, Neutral, Bullish — with a production-grade interactive Streamlit dashboard featuring real-time inference, session analytics, and model performance visualisations.

![Python](https://img.shields.io/badge/python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=flat&logo=plotly&logoColor=white)
![Tests](https://img.shields.io/badge/tests-8%20passed-brightgreen)
![Accuracy](https://img.shields.io/badge/accuracy-87.9%25-blue)
![License](https://img.shields.io/badge/license-MIT-green)

### [▶ View Live App on Streamlit Cloud](https://nlp1-financial-sentiment.streamlit.app)

---

## Why this project

Most NLP portfolio projects call a pre-built sentiment API or run zero-shot inference. This project builds the full transformer fine-tuning pipeline from scratch — data ingestion, tokenisation, DistilBERT fine-tuning via the HuggingFace Trainer API, evaluation with per-class F1, a reusable inference module, and a 3-tab interactive Streamlit dashboard with Plotly visualisations. The model is hosted on HuggingFace Hub and loaded at runtime — production deployment pattern used in real FinTech pipelines.

**Target use case:** Trading signal generation, earnings sentiment monitoring, portfolio risk dashboards, regulatory news classification.

---

## Live Demo

Three-tab interactive dashboard:

| Tab | What it shows |
|---|---|
| **🔍 Analyse** | Single headline inference with confidence bar chart · Batch analysis with results table, distribution pie, confidence line chart |
| **📊 Dashboard** | Session analytics — total analysed, avg confidence, dominant sentiment, distribution donut, confidence over time, avg confidence per class, full history table |
| **🏆 Model Performance** | F1 by class bar chart · Loss curve (train vs eval) · Validation accuracy per epoch · Training summary table |

---

## Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | **87.9%** |
| **F1 Macro** | **84.2%** |
| F1 — Bullish | 91.6% |
| F1 — Neutral | 82.8% |
| F1 — Bearish | 78.1% |
| Train samples | 9,543 |
| Val samples | 2,388 |
| Epochs | 3 |
| Training time | ~2 hrs (CPU) |

---

## All components

| Component | File | What it does |
|---|---|---|
| Data pipeline | `src/data_loader.py` | Downloads Twitter Financial News Sentiment via HuggingFace Datasets, computes label distribution, saves CSV |
| Fine-tuning | `src/train_sentiment.py` | Full DistilBERT fine-tuning — tokenisation, TrainingArguments, Trainer, per-class F1, saves best model |
| Inference module | `src/predict.py` | Production-ready — loads model from HuggingFace Hub, single + batch prediction, confidence scores |
| Dashboard | `app.py` | 3-tab Streamlit app with Plotly charts, session state analytics, dark theme |
| Unit tests | `tests/test_predict.py` | 8 pytest tests — model load, label validity, confidence range, scores sum, batch, directional accuracy |

---

## NLP / Transformer skills demonstrated

| Skill | Where | Interview talking point |
|---|---|---|
| **Transformer fine-tuning** | `train_sentiment.py` | "DistilBERT is 40% smaller than BERT with 97% of performance — ideal for CPU inference" |
| **HuggingFace Trainer API** | `TrainingArguments`, `Trainer` | "Handles gradient accumulation, mixed precision, checkpoint saving — production-equivalent setup" |
| **Tokenisation** | `AutoTokenizer`, max_length=128 | "Truncation at 128 tokens covers 99%+ of financial headlines — no information loss" |
| **Per-class F1 evaluation** | `classification_report` | "Macro F1 penalises class imbalance — more honest than accuracy alone" |
| **HuggingFace Hub deployment** | `predict.py` | "Model hosted on Hub, loaded at runtime — mirrors production MLOps pattern" |
| **Reusable inference pipeline** | `predict.py` | "Decoupled from training — importable module for any downstream application" |
| **Session state analytics** | `app.py` | "Streamlit session_state tracks inference history for live dashboard updates" |
| **Interactive visualisations** | Plotly in `app.py` | "Plotly renders client-side — no server round-trip per chart interaction" |

---

## Dataset

**Twitter Financial News Sentiment** — [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)

| Label | Class | Count | % |
|---|---|---|---|
| 0 | Bearish | 1,442 | 15.1% |
| 1 | Neutral | 1,923 | 20.2% |
| 2 | Bullish | 6,178 | 64.7% |

Class imbalance is the primary challenge — Bearish F1 (78.1%) is lower than Bullish (91.6%) due to fewer training examples.

---

## Model details

| Component | Detail |
|---|---|
| Base model | `distilbert-base-uncased` |
| Task | Sequence classification — 3 classes |
| Max sequence length | 128 tokens |
| Batch size | 16 (train) / 32 (eval) |
| Epochs | 3 |
| Optimiser | AdamW (HuggingFace Trainer default) |
| Scheduler | Linear warmup |
| Best model metric | F1 Macro |
| Model hosting | HuggingFace Hub |

---

## Project structure
nlp1-financial-sentiment/
├── app.py                    3-tab Streamlit dashboard — inference, analytics, model performance
├── src/
│   ├── data_loader.py        Dataset download, EDA, label distribution, CSV export
│   ├── train_sentiment.py    DistilBERT fine-tuning pipeline — full Trainer API setup
│   └── predict.py            Production inference module — HuggingFace Hub model loading
├── tests/
│   └── test_predict.py       8 pytest unit tests — model, labels, confidence, batch, directional
├── data/
│   └── phrasebank_raw.csv    Raw dataset export (9,543 rows)
├── .streamlit/
│   └── config.toml           Dark blue theme configuration
├── requirements.txt
├── packages.txt
└── .gitignore

---

## Local setup

```bash
git clone https://github.com/fahadamjad009/nlp1-financial-sentiment.git
cd nlp1-financial-sentiment
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**Train from scratch (~2 hrs CPU):**
```bash
python src/data_loader.py
python src/train_sentiment.py
```

**Run tests:**
```bash
python -m pytest tests/ -v
```

**Run the app:**
```bash
streamlit run app.py
```

---

## Tech stack

Python · HuggingFace Transformers · PyTorch · HuggingFace Datasets · scikit-learn · Streamlit · Plotly · pytest

---

## License

MIT — see `LICENSE`
