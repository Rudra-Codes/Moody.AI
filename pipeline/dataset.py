import torch
from torch.utils.data import Dataset

class IMDBDataset(Dataset):
    def __init__(self,
                 df,
                 tokenizer,
                 text_col="text",
                 label_col="label",
                 max_len=512):
        self.texts = df[text_col].to_list()
        self.labels = df[label_col].to_list()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.system_prompt = """
        Classify the input movie review.
        
        Allowed outputs: POSITIVE or NEGATIVE.
        POSITIVE: The reviewer likes, enjoys, praises, or recommends the movie.
        NEGATIVE: The reviewer dislikes, criticizes, dislikes, or does not recommend the movie.
        
        Judge the overall sentiment, not individual words.
        
        Return exactly one allowed output and nothing else.
        No explanation. No punctuation. No additional words.
        """

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        text = f"{self.system_prompt}\n{text}\nSentiment:"

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }