from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score,
    # roc_auc_score,
)
from transformers import pipeline
import torch
import pandas as pd

DATASET_PATH = "../data/cleaned_train.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)

# batch_df = df.sample(n=100, random_state=42).copy()
batch_df = df # SInce taking all rows

# Map numeric labels to text labels
# label_map = {
#     0: "NEGATIVE",
#     1: "POSITIVE",
# }

# batch_df["expected"] = batch_df["label"].map(label_map)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model="google/gemma-3-1b-it",
    device="cuda",
    torch_dtype=torch.bfloat16,
    max_length=None
)

# Build prompts for all 100 reviews
messages = []

for review in batch_df["text"]:
    messages.append([
        {
            "role": "system",
            "content": [{"type": "text", "text": 
"""
Classify the input movie review.

Allowed outputs: POSITIVE or NEGATIVE.
POSITIVE: The reviewer likes, enjoys, praises, or recommends the movie.
NEGATIVE: The reviewer dislikes, criticizes, dislikes, or does not recommend the movie.

Judge the overall sentiment, not individual words.

Return exactly one allowed output and nothing else.
No explanation. No punctuation. No additional words.
"""
                         },]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": str(review),
            }],
        },
    ])

# Batch inference
outputs = pipe(
    messages,
    max_new_tokens=4,
    batch_size=16,  # Adjust based on GPU memory
    do_sample=False,
)

# Extract generated response
predictions = []
try:
    for output in outputs:
        response = output[0]['generated_text'][2]['content'].strip().lower()

        # Keep only the allowed labels
        if "positive" == response:
            prediction = 1
        elif "negative" == response:
            prediction = 0
        else:
            prediction = -1

        predictions.append(prediction)
except Exception as e:
    print("Moj nhi hui bete beacuse", e)
batch_df["prediction"] = predictions

# accuracy = (batch_df["prediction"] == batch_df["label"]).mean()
# invalids = (batch_df["prediction"] == -1).sum()

# print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
# print(f"Number of Invalids:", invalids)

# ---------------------------------------------------------
# Remove invalid predictions for metric calculation
# ---------------------------------------------------------
valid_df = batch_df[batch_df["prediction"] != -1].copy()

y_true = valid_df["label"]
y_pred = valid_df["prediction"]

# ---------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

# ---------------------------------------------------------
# Metrics for each class
# ---------------------------------------------------------
precision_per_class = precision_score(
    y_true,
    y_pred,
    average=None,
    labels=[0, 1],
    zero_division=0
)

recall_per_class = recall_score(
    y_true,
    y_pred,
    average=None,
    labels=[0, 1],
    zero_division=0
)

f1_per_class = f1_score(
    y_true,
    y_pred,
    average=None,
    labels=[0, 1],
    zero_division=0
)

# ---------------------------------------------------------
# Macro / weighted metrics
# ---------------------------------------------------------
macro_precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)

macro_recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)

macro_f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

weighted_precision = precision_score(
    y_true, y_pred, average="weighted", zero_division=0
)

weighted_recall = recall_score(
    y_true, y_pred, average="weighted", zero_division=0
)

weighted_f1 = f1_score(
    y_true, y_pred, average="weighted", zero_division=0
)

# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

tn, fp, fn, tp = cm.ravel()

# Specificity = TN / (TN + FP)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# Sensitivity is the same as recall for the positive class
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

# ---------------------------------------------------------
# Other useful metrics
# ---------------------------------------------------------
balanced_acc = balanced_accuracy_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)

# ---------------------------------------------------------
# Invalid predictions
# ---------------------------------------------------------
total = len(batch_df)
valid = len(valid_df)
invalids = total - valid
invalid_rate = invalids / total

# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------
print("=" * 60)
print("EVALUATION METRICS")
print("=" * 60)

print(f"Total samples        : {total}")
print(f"Valid predictions    : {valid}")
print(f"Invalid predictions  : {invalids}")
print(f"Invalid rate         : {invalid_rate:.4f} ({invalid_rate * 100:.2f}%)")

print("\n--- Overall Metrics ---")
print(f"Accuracy             : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision (Positive) : {precision:.4f}")
print(f"Recall (Positive)    : {recall:.4f}")
print(f"F1 Score (Positive)  : {f1:.4f}")
print(f"Specificity          : {specificity:.4f}")
print(f"Sensitivity          : {sensitivity:.4f}")
print(f"Balanced Accuracy    : {balanced_acc:.4f}")
print(f"MCC                  : {mcc:.4f}")
print(f"Cohen's Kappa        : {kappa:.4f}")

print("\n--- Per-Class Metrics ---")
print("Class 0 (NEGATIVE)")
print(f"  Precision          : {precision_per_class[0]:.4f}")
print(f"  Recall             : {recall_per_class[0]:.4f}")
print(f"  F1                 : {f1_per_class[0]:.4f}")

print("\nClass 1 (POSITIVE)")
print(f"  Precision          : {precision_per_class[1]:.4f}")
print(f"  Recall             : {recall_per_class[1]:.4f}")
print(f"  F1                 : {f1_per_class[1]:.4f}")

print("\n--- Averaged Metrics ---")
print(f"Macro Precision      : {macro_precision:.4f}")
print(f"Macro Recall         : {macro_recall:.4f}")
print(f"Macro F1             : {macro_f1:.4f}")
print(f"Weighted Precision   : {weighted_precision:.4f}")
print(f"Weighted Recall      : {weighted_recall:.4f}")
print(f"Weighted F1          : {weighted_f1:.4f}")

print("\n--- Confusion Matrix ---")
print("                    Predicted")
print("                  NEGATIVE  POSITIVE")
print(f"Actual NEGATIVE     {tn:6d}   {fp:6d}")
print(f"Actual POSITIVE     {fn:6d}   {tp:6d}")

# ---------------------------------------------------------
# Full sklearn classification report
# ---------------------------------------------------------
print("\n--- Classification Report ---")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["NEGATIVE", "POSITIVE"],
        digits=4,
        zero_division=0,
    )
)