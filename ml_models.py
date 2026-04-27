"""
ML Models — CyberShield AI
Author: Abhay Sharma | github.com/KAZURIKAFU
Trains & evaluates 3 ML models for network intrusion detection
Random Forest | SVM | Neural Network (MLP)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from traffic_simulator import generate_batch, get_feature_columns, ATTACK_TYPES
import warnings
warnings.filterwarnings("ignore")

FEATURE_COLS = get_feature_columns()
ATTACK_LABELS = list(ATTACK_TYPES.keys())

# ── Training Data Generator ───────────────────────────────────────────────────
def _generate_training_data(n_samples: int = 3000) -> tuple:
    """Generate balanced training dataset."""
    df = generate_batch(n_samples)
    X = df[FEATURE_COLS].values
    y = df["attack_type"].values
    return X, y


# ── Model Trainer ─────────────────────────────────────────────────────────────
class CyberShieldModels:
    def __init__(self):
        self.scaler   = StandardScaler()
        self.encoder  = LabelEncoder()
        self.models   = {}
        self.metrics  = {}
        self.trained  = False
        self._train()

    def _train(self):
        print("  🔧 Generating training data...")
        X, y = _generate_training_data(3000)
        y_enc = self.encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

        model_configs = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "SVM": SVC(
                kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
            "Neural Network": MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), activation="relu",
                max_iter=300, random_state=42, early_stopping=True),
        }

        for name, model in model_configs.items():
            print(f"  🏋️  Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            self.models[name] = model
            self.metrics[name] = {
                "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
                "precision": round(precision_score(y_test, y_pred, average="weighted",
                                                   zero_division=0) * 100, 2),
                "recall":    round(recall_score(y_test, y_pred, average="weighted",
                                                zero_division=0) * 100, 2),
                "f1_score":  round(f1_score(y_test, y_pred, average="weighted",
                                            zero_division=0) * 100, 2),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classes":   list(self.encoder.classes_),
            }

        self.trained = True
        print("  ✅ All models trained successfully!")

    def predict(self, packet: dict, model_name: str = "Random Forest") -> dict:
        """Predict attack type for a single packet."""
        if not self.trained:
            return {"attack_type": "Unknown", "confidence": 0.0}
        features = np.array([[packet.get(f, 0) for f in FEATURE_COLS]])
        features_scaled = self.scaler.transform(features)
        model = self.models[model_name]
        pred_enc = model.predict(features_scaled)[0]
        pred_label = self.encoder.inverse_transform([pred_enc])[0]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_scaled)[0]
            confidence = round(float(proba.max()) * 100, 1)
        else:
            confidence = 85.0
        return {"attack_type": pred_label, "confidence": confidence,
                "model": model_name}

    def predict_batch(self, df: pd.DataFrame,
                      model_name: str = "Random Forest") -> pd.DataFrame:
        """Predict attack types for a batch of packets."""
        if not self.trained or df.empty:
            return df
        features = df[FEATURE_COLS].values
        features_scaled = self.scaler.transform(features)
        model = self.models[model_name]
        preds_enc = model.predict(features_scaled)
        preds = self.encoder.inverse_transform(preds_enc)
        df = df.copy()
        df["predicted_attack"] = preds
        df["is_threat"] = df["predicted_attack"] != "Normal"
        return df

    def get_metrics_df(self) -> pd.DataFrame:
        """Return model metrics as a DataFrame for display."""
        rows = []
        for name, m in self.metrics.items():
            rows.append({
                "Model":     name,
                "Accuracy":  f"{m['accuracy']}%",
                "Precision": f"{m['precision']}%",
                "Recall":    f"{m['recall']}%",
                "F1-Score":  f"{m['f1_score']}%",
            })
        return pd.DataFrame(rows)

    def get_best_model(self) -> str:
        """Return name of best performing model by F1-Score."""
        return max(self.metrics, key=lambda k: self.metrics[k]["f1_score"])


# ── Singleton instance ────────────────────────────────────────────────────────
print("🛡️  CyberShield AI — Initializing ML Models...")
MODELS = CyberShieldModels()
print(f"  🏆 Best model: {MODELS.get_best_model()} "
      f"({MODELS.metrics[MODELS.get_best_model()]['accuracy']}% accuracy)\n")
