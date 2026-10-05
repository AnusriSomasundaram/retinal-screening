"""
OLIVES Dataset - EfficientNetB0 Training (2-stage: frozen head, then fine-tune)
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from data_pipeline import train_tf, val_tf, class_weight

print("=" * 60)
print("Building EfficientNetB0")
print("=" * 60)

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)
base_model.trainable = False

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

inputs = tf.keras.Input(shape=(224, 224, 3))
x = augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)
print("\nModel Created Successfully")
model.summary()

metrics = [
    "accuracy",
    tf.keras.metrics.AUC(name="auc"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=metrics,
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras", monitor="val_auc", mode="max", save_best_only=True, verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc", patience=7, mode="max", restore_best_weights=True, verbose=1
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, verbose=1
)

print("\nStarting Training (Stage 1: frozen backbone)...")
history = model.fit(
    train_tf,
    validation_data=val_tf,
    epochs=30,
    class_weight=class_weight,
    callbacks=[checkpoint, early_stop, reduce_lr],
)
print("\nStage 1 Training Completed")

# ==========================================
# Fine-tuning
# ==========================================
base_model.trainable = True

# Keep BatchNorm layers frozen (standard practice for fine-tuning)
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Only unfreeze the last layers of the backbone
for layer in base_model.layers[:200]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=metrics,
)

print("\nStarting Training (Stage 2: fine-tuning)...")
history_fine = model.fit(
    train_tf,
    validation_data=val_tf,
    epochs=15,
    class_weight=class_weight,
    callbacks=[checkpoint, early_stop, reduce_lr],
)
print("\nFine-tuning Completed")

model.save("final_model.keras")
print("\nSaved final_model.keras (last epoch) and best_model.keras (best val_auc)")

# ==========================================
# Step 11: Save history + graphs
# ==========================================
full_history = {}
for k in history.history:
    full_history[k] = history.history[k] + history_fine.history.get(k, [])

with open("training_history.json", "w") as f:
    json.dump(full_history, f, indent=2)


def plot_metric(metric_name, filename):
    plt.figure()
    plt.plot(full_history[metric_name], label=f"train_{metric_name}")
    plt.plot(full_history[f"val_{metric_name}"], label=f"val_{metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name.capitalize()} over Epochs")
    plt.legend()
    plt.savefig(filename)
    plt.close()


plot_metric("accuracy", "accuracy_graph.png")
plot_metric("loss", "loss_graph.png")
plot_metric("auc", "auc_graph.png")

print("Saved training_history.json, accuracy_graph.png, loss_graph.png, auc_graph.png")
