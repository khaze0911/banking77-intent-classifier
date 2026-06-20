import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "./models/banking77-distilbert")

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

# Load the tokenizer and model ONCE at startup
_tokenizer = None
_model = None

# Load the tokenizer and model from disk into memory. Called once at server startup.
def load_model():
    global _tokenizer, _model
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Running on device: {DEVICE}")

    # Tokenizer
    # raw text → token IDs
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Model
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    # Move model weights to GPU (if available)
    _model.to(DEVICE)

    # switches off dropout and batch normalization
    _model.eval()
    
    print(f"Model loaded. {_model.config.num_labels} classes, device={DEVICE}")

"""
Run inference on a single text string

Args:
    text:  The raw banking query, e.g. "How do I freeze my card?"
    top_k: Number of top predictions to return (default 3)

Returns:
    A dict with:
        - predicted_label: highest-confidence class name
        - confidence:       probability of the top class (0–1)
        - top_k:           list of {label, score} for top_k classes
"""
def predict(text: str, top_k: int = 3) -> dict:
    if _tokenizer is None or _model is None:
        raise RuntimeError("Model not loaded")
    
    # Tokenization
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )

    # Move input tensors to the same device as the model.
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Forward Pass
#   # the model returns a SequenceClassifierOutput object.
    # .logits is a tensor of shape [batch_size, num_classes] = [1, 77]
    # logits are raw, unnormalised scores.
    with torch.no_grad():
    # tells PyTorch not to build the computation graph used for backpropagation
        outputs = _model(**inputs)
        logits = outputs.logits # shape: [1, 77]

    # Convert logits
    # softmax turns the 77 raw scores into a probability distribution
    # dim=-1 means "apply softmax along the last dimension aka the 77 classes"
    probabilities = torch.softmax(logits, dim=-1) # shape: [1, 77]

    # removes the batch dimension and moves the tensor back to CPU so we can convert it to Python types
    probs = probabilities.squeeze(0).cpu() # shape: [77]

    # Top k predictions
    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))

    # Convert to human-readable label strings
    id2label = _model.config.id2label

    top_predictions = [
        {"label": id2label[idx.item()], "score": round(prob.item(), 4)}
        for idx, prob in zip(top_indices, top_probs)
    ]

    return {
        "predicted_label": top_predictions[0]["label"],
        "confidence": top_predictions[0]["score"],
        "top_k": top_predictions,
    }




