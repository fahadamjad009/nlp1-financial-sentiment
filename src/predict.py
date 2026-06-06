"""
Inference pipeline for financial sentiment classification.
Loads fine-tuned DistilBERT from HuggingFace Hub at runtime.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

HF_MODEL_ID = "fahadamjad009/nlp1-financial-sentiment"
LABEL_MAP = {0: "bearish", 1: "neutral", 2: "bullish"}
EMOJI_MAP = {0: "🔴", 1: "🟡", 2: "🟢"}


def load_model(model_path=None):
    """Load fine-tuned model from HuggingFace Hub."""
    source = model_path if model_path else HF_MODEL_ID
    print(f"Loading model from {source}...")
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    model.eval()
    return tokenizer, model


def predict(text, tokenizer, model):
    """Run sentiment prediction on a single text string."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1).squeeze()
    pred_id = torch.argmax(probs).item()

    return {
        "label": LABEL_MAP[pred_id],
        "emoji": EMOJI_MAP[pred_id],
        "confidence": round(probs[pred_id].item(), 4),
        "scores": {
            "bearish": round(probs[0].item(), 4),
            "neutral": round(probs[1].item(), 4),
            "bullish": round(probs[2].item(), 4),
        },
    }


def predict_batch(texts, tokenizer, model):
    """Run predictions on a list of texts."""
    return [predict(t, tokenizer, model) for t in texts]


if __name__ == "__main__":
    tokenizer, model = load_model()
    test_headlines = [
        "Apple beats earnings expectations, stock surges 8%",
        "Fed raises interest rates amid inflation concerns",
        "Tesla misses delivery targets for third consecutive quarter",
    ]
    print("\n--- Inference Test ---")
    for headline in test_headlines:
        result = predict(headline, tokenizer, model)
        print(f"\n{result['emoji']} {result['label'].upper()} ({result['confidence']:.1%})")
        print(f"   {headline}")