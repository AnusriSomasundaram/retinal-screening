"""
OLIVES Dataset - Test Set Evaluation (Steps 12-13 of the algorithm)
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from data_pipeline import test_tf, test_dataset

model = tf.keras.models.load_model("best_model.keras")

print("Evaluating on held-out test patients...")
results = model.evaluate(test_tf, return_dict=True)
print("\nTest Results:")
for k, v in results.items():
    print(f"  {k}: {v:.4f}")

y_true = np.array(test_dataset["Disease Label"]).astype(int)
y_prob = model.predict(test_tf).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Normal", "Disease"]))

print("\nROC AUC:", roc_auc_score(y_true, y_prob))
