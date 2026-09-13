from transformers import pipeline
import torch
import pandas as pd

DATASET_PATH = "data/cleaned_train.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Take first 100 rows
batch_df = df.sample(n=1000, random_state=42).copy()

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
    batch_size=25,  # Adjust based on GPU memory
    do_sample=False,
)

# Extract generated response
predictions = []

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

batch_df["prediction"] = predictions

accuracy = (batch_df["prediction"] == batch_df["label"]).mean()
invalids = (batch_df["prediction"] == -1).sum()

print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Number of Invalids:", invalids)

# Show results
    # print(
    #     batch_df[
    #         ["text", "label", "prediction"]
    #     ].to_string(index=False)
    # )