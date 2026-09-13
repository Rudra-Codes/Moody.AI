import os
import pandas as pd
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from huggingface_hub import login

import sampler
from dataset import IMDBDataset

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

login(token=hf_token)
print("Logged-in to HuggingFace")

train_data = pd.read_csv("../data/cleaned_train.csv")
test_data = pd.read_csv("../data/cleaned_test.csv")

train_sample = sampler.stratified_sample(train_data, 10000)
test_sample = sampler.stratified_sample(test_data, 1000)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

train_dataset = IMDBDataset(train_sample, tokenizer)
test_dataset = IMDBDataset(test_sample, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

batch = next(iter(train_loader))
print(batch['input_ids'].shape)
print(batch['attention_mask'].shape)
print(batch['label'].shape)