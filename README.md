# Banking77 Intent Classifier

Fine-tuning DistilBERT on 77-class banking query intent classification.

**Status: In Progress**

- [x] EDA and class distribution analysis
- [x] DistilBERT fine-tuning — 93% accuracy, 93% macro F1
- [x] Evaluation — confusion matrix, per-class F1 analysis
- [x] FastAPI REST endpoint
- [ ] Docker + AWS EC2 deployment

## Quick start (local)
uvicorn app.main:app --reload
