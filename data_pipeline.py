"""
OLIVES Dataset - Data Pipeline (fixed)

The original script split "patients with a normal-labeled image" and
"patients with a disease-labeled image" into train/val/test INDEPENDENTLY.
Because a single patient can have both normal and disease images (different
eyes / visits), the same patient could end up in two different splits
(e.g. train AND test) -> the model saw that patient's retina during
training and again during evaluation. That leakage is almost certainly
why accuracy looked like ~99%.

Fix: give every patient exactly ONE label, then split PATIENTS (not
images) so a patient can only ever appear in a single split. Asserts are
added to make this verifiable rather than assumed.
"""

from datasets import load_dataset
import numpy as np
import random
import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

print("Loading OLIVES dataset...")
dataset = load_dataset(
    "gOLIVES/OLIVES_Dataset",
    "disease_classification"
)["train"]
print("Dataset loaded successfully!")
print("Total Images:", len(dataset))

# ==========================================
# Step 2: Explore Dataset
# ==========================================
sample0 = dataset[0]
img0 = np.array(sample0["Image"])
print("\nSample Image Shape:", img0.shape)
print("Sample Image Dtype :", img0.dtype)
print(
    "Sample Image Mode  :",
    "Grayscale" if img0.ndim == 2 else f"{img0.shape[-1]} channel(s)"
)

# ==========================================
# Step 3: One label per patient, then patient-level split
# ==========================================
patient_id_col = np.array(dataset["Patient_ID"]).astype(int)
label_col = np.array(dataset["Disease Label"]).astype(int)

# A patient counts as "disease" if ANY of their images is labeled disease.
patient_has_disease = {}
for pid, lbl in zip(patient_id_col, label_col):
    patient_has_disease[pid] = patient_has_disease.get(pid, False) or (lbl == 1)

normal_patients = [pid for pid, d in patient_has_disease.items() if not d]
disease_patients = [pid for pid, d in patient_has_disease.items() if d]

print("\nUnique Normal Patients :", len(normal_patients))
print("Unique Disease Patients:", len(disease_patients))
print("Total Unique Patients  :", len(patient_has_disease))

assert set(normal_patients).isdisjoint(disease_patients), \
    "A patient landed in both label buckets - this should be impossible now."

random.seed(SEED)
random.shuffle(normal_patients)
random.shuffle(disease_patients)


def split_patients(patient_list):
    n = len(patient_list)
    train = patient_list[:int(0.70 * n)]
    val = patient_list[int(0.70 * n):int(0.85 * n)]
    test = patient_list[int(0.85 * n):]
    return train, val, test


train_normal, val_normal, test_normal = split_patients(normal_patients)
train_disease, val_disease, test_disease = split_patients(disease_patients)

train_patients = set(train_normal + train_disease)
val_patients = set(val_normal + val_disease)
test_patients = set(test_normal + test_disease)

# The whole point of the fix - verify no patient appears in 2+ splits.
assert train_patients.isdisjoint(val_patients), "Leakage: train/val overlap!"
assert train_patients.isdisjoint(test_patients), "Leakage: train/test overlap!"
assert val_patients.isdisjoint(test_patients), "Leakage: val/test overlap!"

print("\n==============================")
print("Patient Split (verified leak-free)")
print("==============================")
print("Training Patients  :", len(train_patients))
print("Validation Patients:", len(val_patients))
print("Testing Patients   :", len(test_patients))
print("Total Patients     :",
      len(train_patients) + len(val_patients) + len(test_patients))

# ==========================================
# Step 4: Image Split (vectorized - much faster than row-by-row .filter)
# ==========================================
train_mask = np.isin(patient_id_col, list(train_patients))
val_mask = np.isin(patient_id_col, list(val_patients))
test_mask = np.isin(patient_id_col, list(test_patients))

train_dataset = dataset.select(np.where(train_mask)[0])
val_dataset = dataset.select(np.where(val_mask)[0])
test_dataset = dataset.select(np.where(test_mask)[0])

# IMPORTANT: the OLIVES dataset is stored grouped by patient, so each
# patient contributes a long run of consecutive same-label images. A
# tf.data .shuffle(buffer) later on can't fully break up runs longer than
# the buffer, so training batches can end up almost entirely one class
# (you'll see accuracy near 100% but auc/precision/recall stuck at 0 -
# that's a sign the model is only ever seeing one label per batch).
# Shuffling the HF dataset itself first (cheap - it just reorders an
# index list, not decoded images) fixes this at the source.
train_dataset = train_dataset.shuffle(seed=SEED)
val_dataset = val_dataset.shuffle(seed=SEED)
test_dataset = test_dataset.shuffle(seed=SEED)

print("\n==============================")
print("Image Split")
print("==============================")
print("Training Images  :", len(train_dataset))
print("Validation Images:", len(val_dataset))
print("Testing Images   :", len(test_dataset))
print("Total Images     :",
      len(train_dataset) + len(val_dataset) + len(test_dataset))

# Class balance in the training set - if this is very skewed, accuracy
# alone will look inflated no matter how clean the split is.
train_labels = np.array(train_dataset["Disease Label"]).astype(int)
n_pos = int(train_labels.sum())
n_neg = int(len(train_labels) - n_pos)
print(f"\nTrain label balance -> Normal: {n_neg}  Disease: {n_pos}")

# ==========================================
# Step 5: Image Preprocessing
# ==========================================

def preprocess(sample):
    image = np.array(sample["Image"]).astype(np.float32)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)

    # EfficientNet's preprocess_input expects raw 0-255 pixel values and
    # applies the correct scaling internally - don't also divide by 255
    # yourself or you'll double-normalize.
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    label = np.float32(sample["Disease Label"])
    return image, label


def generator(hf_dataset):
    for sample in hf_dataset:
        yield preprocess(sample)


print("\n==============================")
print("Inspect First Training Image")
print("==============================")
sample = train_dataset[0]
image, label = preprocess(sample)
print("Image Shape :", image.shape)
print("Image Type  :", image.dtype)
print("Label       :", label)
print("Min Pixel   :", tf.reduce_min(image).numpy())
print("Max Pixel   :", tf.reduce_max(image).numpy())

# ==========================================
# Step 6: TensorFlow Dataset
# ==========================================
output_signature = (
    tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.float32),
)

train_tf = tf.data.Dataset.from_generator(
    lambda: generator(train_dataset), output_signature=output_signature
)
val_tf = tf.data.Dataset.from_generator(
    lambda: generator(val_dataset), output_signature=output_signature
)
test_tf = tf.data.Dataset.from_generator(
    lambda: generator(test_dataset), output_signature=output_signature
)

train_tf = train_tf.shuffle(1000, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_tf = val_tf.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_tf = test_tf.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Class weights - use these in model.fit(..., class_weight=class_weight)
class_weight = None
if n_pos > 0 and n_neg > 0:
    total = n_pos + n_neg
    class_weight = {
        0: total / (2.0 * n_neg),
        1: total / (2.0 * n_pos),
    }
    print("\nClass weights:", class_weight)

print("\n==============================")
print("TensorFlow Pipeline Ready")
print("==============================")
for images, labels in train_tf.take(1):
    print("Image batch :", images.shape)
    print("Label batch :", labels.shape)
    print("Image dtype :", images.dtype)
    print("Labels      :", labels[:10].numpy())
