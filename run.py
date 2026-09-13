import torch
import torch.nn as nn
import os

import pandas as pd
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from huggingface_hub import login
from pipeline.sampler import stratified_sample
from pipeline.dataset import IMDBDataset
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

login(token=hf_token)
print("Logged-in to HuggingFace")

train_data = pd.read_csv("data/cleaned_train.csv")
test_data = pd.read_csv("data/cleaned_test.csv")
train_sample = stratified_sample(train_data, 10000)
test_sample = stratified_sample(test_data, 1000)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

train_dataset = IMDBDataset(train_sample, tokenizer)
test_dataset = IMDBDataset(test_sample, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
criterion = nn.CrossEntropyLoss()

print("Hello")
def get_model(model_name):
    if model_name == "gemma":
        from models.gemma_bin import GemmaModel
        return GemmaModel()

    raise ValueError(f"Unknown model name: {model_name}")


def load_model(model_name, model_path=None, load_existing=False):
    model = get_model(model_name)

    if load_existing:
        state_dict = torch.load(
            model_path,
            map_location="cpu"
        )
        model.load_state_dict(state_dict)

    return model


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def get_metrics(pred, target):
    pred_labels = torch.argmax(pred, dim=1)

    tp = ((pred_labels == 1) & (target == 1)).sum().item()
    fp = ((pred_labels == 1) & (target == 0)).sum().item()
    fn = ((pred_labels == 0) & (target == 1)).sum().item()

    accuracy = (pred_labels == target).float().mean().item()

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):

    model.eval()

    all_preds = []
    all_targets = []
    total_loss = 0.0

    for batch in dataloader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids,
            attention_mask
        )

        loss = criterion(outputs, labels)

        total_loss += loss.item()

        all_preds.append(outputs)
        all_targets.append(labels)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics = get_metrics(
        all_preds,
        all_targets
    )

    metrics["loss"] = total_loss / len(dataloader)

    return metrics

def train(load_existing_model=False):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # train_dataset = IMDBDataset(split="train")
    # val_dataset = IMDBDataset(split="val")

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=32,
    #     shuffle=True
    # )

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=32,
    #     shuffle=False
    # )


    model = load_model(
        "gemma",
        "gemma_model.pt",
        load_existing=load_existing_model
    )

    model.to(device)

    lora_params = []
    classifier_params = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if "classifier" in name:
            classifier_params.append(param)
        else:
            lora_params.append(param)

    optimizer = torch.optim.AdamW([
        {
            "params": lora_params,
            "lr": 2e-5
        },
        {
            "params": classifier_params,
            "lr": 5e-4
        }
    ])


    epochs = 10

    best_f1 = -float("inf")

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids,
                attention_mask
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        val_metrics = evaluate(
            model,
            test_loader,
            device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {val_metrics['loss']:.4f} | "
            f"Test Acc: {val_metrics['accuracy']:.4f} | "
            f"Test F1: {val_metrics['f1_score']:.4f}"
        )


        if val_metrics["f1_score"] > best_f1:

            best_f1 = val_metrics["f1_score"]

            save_model(
                model,
                "gemma_model_best.pt"
            )

            print(
                f"New best F1: {best_f1:.4f} "
                f"model saved"
            )
if __name__=="main":
    train(False)
