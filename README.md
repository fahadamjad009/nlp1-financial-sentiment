\# 📈 Financial Sentiment Analyser



> Fine-tuned \*\*DistilBERT\*\* transformer for 3-class financial news sentiment classification.  

> Built with HuggingFace Transformers · PyTorch · Streamlit · Plotly



\[!\[Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat\&logo=python\&logoColor=white)](https://python.org)

\[!\[HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat\&logo=huggingface\&logoColor=black)](https://huggingface.co)

\[!\[PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?style=flat\&logo=pytorch\&logoColor=white)](https://pytorch.org)

\[!\[Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=flat\&logo=streamlit\&logoColor=white)](https://streamlit.io)

\[!\[Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=flat\&logo=plotly\&logoColor=white)](https://plotly.com)



\---



\## 🎯 Project Overview



This project fine-tunes `distilbert-base-uncased` on 9,543 financial news headlines to classify sentiment as \*\*Bearish\*\*, \*\*Neutral\*\*, or \*\*Bullish\*\*. It demonstrates a complete NLP engineering pipeline: data ingestion, transformer fine-tuning, evaluation, reusable inference module, and an interactive multi-tab Streamlit dashboard.



\*\*Target use case:\*\* FinTech / trading signal generation, earnings sentiment monitoring, portfolio risk dashboards.



\---



\## 📊 Model Performance



| Metric | Score |

|---|---|

| \*\*Accuracy\*\* | \*\*87.9%\*\* |

| \*\*F1 Macro\*\* | \*\*84.2%\*\* |

| F1 — Bullish | 91.6% |

| F1 — Neutral | 82.8% |

| F1 — Bearish | 78.1% |

| Train Samples | 9,543 |

| Val Samples | 2,388 |

| Epochs | 3 |



\---



\## 🏗️ Architecture

nlp1-financial-sentiment/

├── app.py                  # Streamlit multi-tab dashboard

├── src/

│   ├── data\_loader.py      # Dataset download \& EDA

│   ├── train\_sentiment.py  # DistilBERT fine-tuning pipeline

│   └── predict.py          # Reusable inference module

├── tests/

│   └── test\_predict.py     # Unit tests for inference pipeline

├── notebooks/

│   └── eda.ipynb           # Exploratory data analysis

├── .streamlit/

│   └── config.toml         # Dark theme configuration

├── requirements.txt

└── packages.txt



\---



\## 🚀 Live Demo



\[!\[Streamlit App](https://static.streamlit.io/badges/streamlit\_badge\_black\_white.svg)](https://nlp1-financial-sentiment.streamlit.app)



\*\*Three interactive tabs:\*\*

\- \*\*🔍 Analyse\*\* — Single headline + batch analysis with confidence bar charts

\- \*\*📊 Dashboard\*\* — Session analytics: distribution, confidence over time, class breakdown

\- \*\*🏆 Model Performance\*\* — F1 scores, loss curves, accuracy per epoch



\---



\## ⚙️ Local Setup



```bash

git clone https://github.com/fahadamjad009/nlp1-financial-sentiment.git

cd nlp1-financial-sentiment

python -m venv venv

venv\\Scripts\\activate        # Windows

pip install -r requirements.txt

```



\*\*Train the model (optional — \~20 min CPU):\*\*

```bash

python src/data\_loader.py

python src/train\_sentiment.py

```



\*\*Run the app:\*\*

```bash

streamlit run app.py

```



\---



\## 🔬 Dataset



\*\*Twitter Financial News Sentiment\*\* (\[zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment))



| Label | Class | Train Count |

|---|---|---|

| 0 | Bearish | 1,442 |

| 1 | Neutral | 1,923 |

| 2 | Bullish | 6,178 |



\---



\## 🧠 Model Details



| Component | Detail |

|---|---|

| Base model | `distilbert-base-uncased` |

| Task | Sequence classification (3-class) |

| Max sequence length | 128 tokens |

| Batch size | 16 (train) / 32 (eval) |

| Epochs | 3 |

| Optimiser | AdamW (default HF Trainer) |

| Best model metric | F1 Macro |



\---



\## 🛠️ Tech Stack



| Layer | Technology |

|---|---|

| Model | HuggingFace Transformers, DistilBERT |

| Training | PyTorch, HuggingFace Trainer API |

| Data | HuggingFace Datasets |

| Evaluation | scikit-learn (classification\_report) |

| Dashboard | Streamlit, Plotly |

| Deployment | Streamlit Cloud |



\---



\## 📁 Key Files



| File | Purpose |

|---|---|

| `src/train\_sentiment.py` | Full fine-tuning pipeline with eval metrics |

| `src/predict.py` | Production-ready inference — load model, predict single or batch |

| `app.py` | 3-tab interactive Streamlit dashboard |

| `src/data\_loader.py` | Dataset download, label distribution EDA |



\---



