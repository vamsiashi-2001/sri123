"""
=============================================================
  AI Medical Assistant — Complete Flask Backend
  RAG + ML Training/Testing Pipeline
=============================================================
  Author  : AI Medical Assistant
  Stack   : Flask · FAISS · SentenceTransformers · Groq LLM
            · scikit-learn (ML train/test/predict)
=============================================================
"""

from __future__ import annotations
import os, pickle, json, time
import requests
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ── Embeddings & FAISS ────────────────────────────────────────────────────────
import faiss
from sentence_transformers import SentenceTransformer

# ── Text Splitting ────────────────────────────────────────────────────────────
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=500, chunk_overlap=50, **kw):
            self.chunk_size    = chunk_size
            self.chunk_overlap = chunk_overlap
        def split_text(self, text: str) -> List[str]:
            chunks, start = [], 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunks.append(text[start:end])
                start += self.chunk_size - self.chunk_overlap
            return [c.strip() for c in chunks if c.strip()]

# ── ML (scikit-learn) ─────────────────────────────────────────────────────────
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix)
from sklearn.preprocessing   import LabelEncoder, StandardScaler
import pandas as pd

# =============================================================================
#  Flask App
# =============================================================================
app = Flask(__name__)
CORS(app)

# =============================================================================
#  Configuration — Load from environment variables ONLY (no hardcoded keys)
# =============================================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

DATA_DIR      = "data"
INDEX_FILE    = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"
ML_MODEL_FILE = "ml_model.pkl"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50
TOP_K            = 4

# =============================================================================
#  Global State
# =============================================================================
embedding_model  = None
faiss_index      = None
chunk_metadata:  List[dict] = []
ml_model         = None
ml_label_encoder = None
ml_scaler        = None
ml_training_log: List[dict] = []


# =============================================================================
#  Document Loading
# =============================================================================
def load_documents(data_dir: str) -> List[dict]:
    docs = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return docs

    for fname in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            if fname.lower().endswith(".txt"):
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                if text:
                    docs.append({"text": text, "source": fname})
                    print(f"  [LOAD] {fname} ({len(text):,} chars)")
            elif fname.lower().endswith(".pdf"):
                try:
                    from pypdf import PdfReader
                    reader    = PdfReader(fpath)
                    full_text = "\n\n".join(
                        p.extract_text() for p in reader.pages if p.extract_text()
                    )
                    if full_text:
                        docs.append({"text": full_text, "source": fname})
                        print(f"  [LOAD] {fname} ({len(reader.pages)} pages)")
                except Exception as pe:
                    print(f"  [WARN] PDF skip {fname}: {pe}")
        except Exception as e:
            print(f"  [ERR ] {fname}: {e}")

    print(f"[INFO] {len(docs)} document(s) loaded.")
    return docs


# =============================================================================
#  Chunking
# =============================================================================
def split_into_chunks(docs: List[dict]) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in docs:
        for part in splitter.split_text(doc["text"]):
            if part.strip():
                chunks.append({"text": part.strip(), "source": doc["source"]})
    print(f"[INFO] {len(chunks)} chunks created.")
    return chunks


# =============================================================================
#  FAISS Index
# =============================================================================
def build_faiss_index(chunks: List[dict], model: SentenceTransformer):
    texts      = [c["text"] for c in chunks]
    print("[INFO] Encoding embeddings …")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[INFO] FAISS index: {index.ntotal} vectors (dim={dim})")
    return index, chunks


def save_index(index, meta):
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(meta, f)


def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        meta = pickle.load(f)
    print(f"[INFO] Index loaded: {index.ntotal} vectors")
    return index, meta


# =============================================================================
#  Retrieval
# =============================================================================
def retrieve_chunks(query: str, top_k: int = TOP_K) -> List[dict]:
    q = np.array(embedding_model.encode([query]), dtype="float32")
    faiss.normalize_L2(q)
    scores, idxs = faiss_index.search(q, top_k)
    return [
        {"text": chunk_metadata[i]["text"],
         "source": chunk_metadata[i]["source"],
         "score": float(s)}
        for s, i in zip(scores[0], idxs[0]) if i != -1
    ]


# =============================================================================
#  LLM (Groq)
# =============================================================================
def call_groq(question: str, chunks: List[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in chunks
    )
    system = (
        "You are an expert medical assistant AI. "
        "Answer ONLY from the provided context. "
        "Use ## headings and - bullet points. "
        "Always advise consulting a licensed doctor. "
        "For emergencies say: call 911 immediately."
    )
    user = f"Context:\n{context}\n\n---\nQuestion: {question}\n\nAnswer:"

    resp = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": GROQ_MODEL,
              "messages": [{"role": "system", "content": system},
                           {"role": "user",   "content": user}],
              "temperature": 0.3, "max_tokens": 1024},
        timeout=30
    )
    if resp.status_code == 401:
        raise Exception("Invalid Groq API key.")
    if resp.status_code == 429:
        raise Exception("Groq rate limit. Wait and retry.")
    if resp.status_code != 200:
        raise Exception(f"Groq error {resp.status_code}: {resp.text[:200]}")
    return resp.json()["choices"][0]["message"]["content"]


# =============================================================================
#  ML — Synthetic Medical Dataset Generator
# =============================================================================
def generate_medical_dataset(n_samples: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    n = n_samples

    age        = np.random.randint(18, 85,  n)
    bmi        = np.round(np.random.normal(26, 5, n).clip(15, 50), 1)
    bp         = np.random.randint(80, 200, n)
    glucose    = np.random.randint(70, 300, n)
    chol       = np.random.randint(120, 350, n)
    heart_rate = np.random.randint(50, 110, n)
    smoking    = np.random.randint(0, 2,    n)
    diabetes_h = np.random.randint(0, 2,    n)

    score = (
        (age        > 60).astype(int) * 2 +
        (bmi        > 30).astype(int) * 2 +
        (bp         > 140).astype(int)* 3 +
        (glucose    > 180).astype(int)* 3 +
        (chol       > 240).astype(int)* 2 +
        (heart_rate > 90).astype(int) * 1 +
        smoking * 2 +
        diabetes_h * 2 +
        np.random.randint(0, 3, n)
    )

    labels = np.where(score <= 3, "Low",
             np.where(score <= 7, "Medium", "High"))

    return pd.DataFrame({
        "age":              age,
        "bmi":              bmi,
        "blood_pressure":   bp,
        "glucose":          glucose,
        "cholesterol":      chol,
        "heart_rate":       heart_rate,
        "smoking":          smoking,
        "diabetes_history": diabetes_h,
        "risk_level":       labels
    })


# =============================================================================
#  ML — Train
# =============================================================================
def train_ml_model(algorithm: str = "random_forest", n_samples: int = 500) -> dict:
    global ml_model, ml_label_encoder, ml_scaler

    t0 = time.time()
    df = generate_medical_dataset(n_samples)

    feature_cols = ["age","bmi","blood_pressure","glucose",
                    "cholesterol","heart_rate","smoking","diabetes_history"]
    X = df[feature_cols].values
    y = df["risk_level"].values

    ml_label_encoder = LabelEncoder()
    y_enc = ml_label_encoder.fit_transform(y)

    ml_scaler = StandardScaler()
    X_scaled  = ml_scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    algo_map = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    if algorithm not in algo_map:
        algorithm = "random_forest"
    clf = algo_map[algorithm]

    cv_scores = cross_val_score(clf, X_tr, y_tr, cv=5, scoring="accuracy")
    clf.fit(X_tr, y_tr)
    ml_model = clf

    with open(ML_MODEL_FILE, "wb") as f:
        pickle.dump({"model": ml_model,
                     "encoder": ml_label_encoder,
                     "scaler": ml_scaler,
                     "features": feature_cols}, f)

    y_pred = clf.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    report = classification_report(
        y_te, y_pred,
        target_names=ml_label_encoder.classes_,
        output_dict=True
    )
    cm = confusion_matrix(y_te, y_pred).tolist()

    feat_imp = {}
    if hasattr(clf, "feature_importances_"):
        feat_imp = dict(zip(feature_cols, clf.feature_importances_.round(4).tolist()))
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_).mean(axis=0)
        feat_imp = dict(zip(feature_cols, imp.round(4).tolist()))

    elapsed = round(time.time() - t0, 2)

    result = {
        "algorithm":             algorithm,
        "n_samples":             n_samples,
        "train_size":            len(X_tr),
        "test_size":             len(X_te),
        "accuracy":              round(acc * 100, 2),
        "cv_mean":               round(cv_scores.mean() * 100, 2),
        "cv_std":                round(cv_scores.std()  * 100, 2),
        "cv_scores":             [round(s * 100, 2) for s in cv_scores.tolist()],
        "classification_report": report,
        "confusion_matrix":      cm,
        "classes":               ml_label_encoder.classes_.tolist(),
        "feature_importance":    feat_imp,
        "training_time_sec":     elapsed,
        "trained_at":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    ml_training_log.append({
        "algorithm": algorithm,
        "accuracy":  result["accuracy"],
        "trained_at": result["trained_at"]
    })

    return result


# =============================================================================
#  ML — Predict
# =============================================================================
def predict_risk(patient: dict) -> dict:
    if ml_model is None:
        raise Exception("Model not trained. Call /ml/train first.")

    feature_cols = ["age","bmi","blood_pressure","glucose",
                    "cholesterol","heart_rate","smoking","diabetes_history"]
    try:
        X = np.array([[float(patient.get(f, 0)) for f in feature_cols]])
    except (ValueError, TypeError) as e:
        raise Exception(f"Invalid input: {e}")

    X_scaled = ml_scaler.transform(X)
    pred_enc = ml_model.predict(X_scaled)[0]
    proba    = ml_model.predict_proba(X_scaled)[0]

    label    = ml_label_encoder.inverse_transform([pred_enc])[0]
    classes  = ml_label_encoder.classes_.tolist()
    prob_map = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}

    advice = {
        "Low":    "Your risk indicators look healthy. Maintain regular check-ups.",
        "Medium": "Moderate risk detected. Please consult a doctor for evaluation.",
        "High":   "High risk indicators present. Seek immediate medical attention."
    }

    return {
        "prediction":    label,
        "confidence":    round(float(proba[pred_enc]) * 100, 1),
        "probabilities": prob_map,
        "advice":        advice.get(label, ""),
        "features_used": feature_cols
    }


# =============================================================================
#  Flask Routes — Frontend
# =============================================================================
@app.route("/")
def index():
    return render_template("index.html")


# =============================================================================
#  Flask Routes — RAG
# =============================================================================
@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "index_loaded": faiss_index is not None,
        "chunks":       len(chunk_metadata),
        "ml_trained":   ml_model is not None,
        "embed_model":  EMBED_MODEL_NAME,
        "groq_model":   GROQ_MODEL,
    })


@app.route("/ask", methods=["POST"])
def ask():
    if not request.is_json:
        return jsonify({"error": "Send JSON."}), 400

    question = (request.get_json().get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400
    if len(question) > 1000:
        return jsonify({"error": "Max 1000 characters."}), 400
    if faiss_index is None:
        return jsonify({"error": "Knowledge base not loaded. Restart the server."}), 503

    try:
        chunks = retrieve_chunks(question)
        if not chunks:
            return jsonify({"answer": "No relevant info found.", "sources": [], "error": None})

        answer = call_groq(question, chunks)

        seen, sources = set(), []
        for c in chunks:
            if c["source"] not in seen:
                seen.add(c["source"])
                sources.append({
                    "file":    c["source"],
                    "excerpt": c["text"][:200] + ("…" if len(c["text"]) > 200 else ""),
                    "score":   round(c["score"], 3)
                })

        return jsonify({"answer": answer, "sources": sources, "error": None})

    except requests.exceptions.Timeout:
        return jsonify({"error": "Groq API timed out."}), 504
    except Exception as e:
        print(f"[ERR /ask] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/rebuild-index", methods=["POST"])
def rebuild_index():
    global faiss_index, chunk_metadata
    try:
        docs   = load_documents(DATA_DIR)
        chunks = split_into_chunks(docs)
        if not chunks:
            return jsonify({"error": "No documents in data/."}), 400
        faiss_index, chunk_metadata = build_faiss_index(chunks, embedding_model)
        save_index(faiss_index, chunk_metadata)
        return jsonify({"message": f"Index rebuilt: {len(chunk_metadata)} chunks."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
#  Flask Routes — ML Training & Testing
# =============================================================================
@app.route("/ml/train", methods=["POST"])
def ml_train():
    body      = request.get_json(silent=True) or {}
    algorithm = body.get("algorithm", "random_forest")
    n_samples = int(body.get("n_samples", 500))
    n_samples = max(100, min(n_samples, 5000))
    try:
        result = train_ml_model(algorithm, n_samples)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        print(f"[ERR /ml/train] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ml/predict", methods=["POST"])
def ml_predict():
    body = request.get_json(silent=True) or {}
    try:
        result = predict_risk(body)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/ml/test", methods=["POST"])
def ml_test():
    if ml_model is None:
        return jsonify({"error": "Train the model first (/ml/train)."}), 400

    body   = request.get_json(silent=True) or {}
    n_test = int(body.get("n_test", 100))
    n_test = max(20, min(n_test, 1000))

    try:
        df = generate_medical_dataset(n_test)
        feature_cols = ["age","bmi","blood_pressure","glucose",
                        "cholesterol","heart_rate","smoking","diabetes_history"]
        X      = df[feature_cols].values
        y_true = df["risk_level"].values

        X_scaled = ml_scaler.transform(X)
        y_enc    = ml_label_encoder.transform(y_true)
        y_pred_e = ml_model.predict(X_scaled)
        y_pred   = ml_label_encoder.inverse_transform(y_pred_e)

        acc    = accuracy_score(y_enc, y_pred_e)
        report = classification_report(
            y_true, y_pred,
            target_names=ml_label_encoder.classes_,
            output_dict=True
        )
        cm = confusion_matrix(y_enc, y_pred_e).tolist()

        samples = []
        for i in range(min(10, n_test)):
            row = df.iloc[i]
            samples.append({
                "actual":    y_true[i],
                "predicted": y_pred[i],
                "correct":   bool(y_true[i] == y_pred[i]),
                "age":       int(row["age"]),
                "bmi":       float(row["bmi"]),
                "glucose":   int(row["glucose"]),
                "risk":      y_pred[i],
            })

        return jsonify({
            "status":                "success",
            "n_tested":              n_test,
            "accuracy":              round(acc * 100, 2),
            "classification_report": report,
            "confusion_matrix":      cm,
            "classes":               ml_label_encoder.classes_.tolist(),
            "sample_predictions":    samples,
        })

    except Exception as e:
        print(f"[ERR /ml/test] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ml/history")
def ml_history():
    return jsonify({"history": ml_training_log})


@app.route("/ml/dataset-preview")
def dataset_preview():
    df = generate_medical_dataset(200)
    return jsonify({
        "columns": df.columns.tolist(),
        "rows":    df.head(20).to_dict(orient="records"),
        "shape":   list(df.shape),
        "label_distribution": df["risk_level"].value_counts().to_dict()
    })


# =============================================================================
#  Startup
# =============================================================================
def initialize():
    global embedding_model, faiss_index, chunk_metadata, ml_model, ml_label_encoder, ml_scaler

    print("\n" + "="*60)
    print("  AI Medical Assistant — Starting Up")
    print("="*60)

    if not GROQ_API_KEY:
        print("[WARN] ⚠️  No Groq API key set! Chat will not work.")
        print("[WARN]    Set GROQ_API_KEY in Render's Environment Variables.")
        print("[WARN]    Get a free key at https://console.groq.com")
    else:
        print(f"[INFO] Groq API key configured ✓ (model: {GROQ_MODEL})")

    print(f"[INFO] Loading embedding model: {EMBED_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
    print("[INFO] Embedding model ready ✓")

    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        faiss_index, chunk_metadata = load_index()
    else:
        print("[INFO] Building FAISS index from documents …")
        docs   = load_documents(DATA_DIR)
        chunks = split_into_chunks(docs)
        if chunks:
            faiss_index, chunk_metadata = build_faiss_index(chunks, embedding_model)
            save_index(faiss_index, chunk_metadata)
        else:
            print("[WARN] No documents found — RAG unavailable until data/ is populated.")

    if os.path.exists(ML_MODEL_FILE):
        with open(ML_MODEL_FILE, "rb") as f:
            saved = pickle.load(f)
        ml_model         = saved["model"]
        ml_label_encoder = saved["encoder"]
        ml_scaler        = saved["scaler"]
        print("[INFO] Saved ML model loaded ✓")
    else:
        print("[INFO] Auto-training ML model …")
        train_ml_model("random_forest", 500)
        print("[INFO] ML model ready ✓")

    print("="*60)
    print("  🏥  App is running")
    print("="*60 + "\n")


# =============================================================================
#  Entry Point — Render compatible
# =============================================================================
# Initialize on module load so gunicorn workers also initialize
initialize()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
