"""
Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""


from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.model import load_model, predict, is_ready


# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Server starting, loading model...")
    load_model()
    print("Model loaded, server running")
    yield
    # Shutdown
    print("Server shutting down")

# App instance
app = FastAPI(
    title="Banking77 Intent Classifier",
    description=(
        "Fine-tuned DistilBERT model classifying banking customer queries, 77 categories. 93 accuracy/macro F1"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Pydantic schemas
class TopKPrediction(BaseModel):
    label: str
    score: float

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Banking query to classify",
        examples=["I have been waiting over a week. Is the card still coming?"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of top predictions to return",
    )

class PredictResponse(BaseModel):
    predicted_label: str = Field(description="Highest confidence class")
    confidence: float = Field(description="Prop of top prediction")
    top_k: list[TopKPrediction] = Field(description="Top-k predictions with scores")


@app.get("/healthz", tags=["meta"])
def liveness():
    # Liveness: is the process up and able to respond at all?
    # Deliberately doesn't check the model. A model-aware liveness probe
    # would kill the container during the slow model-load window and cause a
    # permanent crash loop.
    return {"status": "alive"}

@app.get("/readyz", tags=["meta"])
def readiness():
    # Readiness: can this pod actually serve predictions yet?
    # Returns 503 until the model and tokenizer are loaded
    if not is_ready():
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}

# Health check endpoint
# status check for containerised services
# decides whether to route traffic to this instance
@app.get("/health", tags=["meta"])
def health_check():
    return {"status": "ok"}

# Root endpoint
@app.get("/", tags=["Meta"])
def root():
    return {
        "service": "Banking77 Intent Classifier",
        "docs": "/docs",
        "predict": "/predict",
    }

# Predict endpoint
# classifies a banking query into one of 77 intent categories
# returns the top predicted intent and its confidence score, plus the next top_k - 1 runner-up predictions
@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_intent(request: PredictRequest):
    try:
        result = predict(text=request.text, top_k=request.top_k)
    except RuntimeError as e:
        # Model not loaded
        raise HTTPException(status_code=503, detail=str(e))        
    except Exception as e:
        # Catch-all for unexpected inference errors
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    return result