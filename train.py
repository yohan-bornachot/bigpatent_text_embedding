from argparse import ArgumentParser
from datetime import datetime
import numpy as np
import os
import time
import torch.nn.functional as F
import torch
import torch.nn.functional
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel
import random
import yaml

from patent_dataset import PatentDataset
from utils import add_attr_interface


def train(cfg_path: str, data_path: str, output_dir: str, device: str):
    
    # Load config
    with open(cfg_path, "r") as yml_file:
        cfg = add_attr_interface(yaml.safe_load(yml_file))

    # Outputs configuration and settings
    dt = datetime.now()
    date_dir_name = f"{dt.year}_{dt.month:02d}_{dt.day:02d}-{dt.hour:02d}h{dt.minute:02d}_{cfg.TRAIN.MODEL_NAME.split('/')[-1]}"
    output_path = os.path.join(output_dir, date_dir_name)
    summary_path = os.path.join(output_path, 'summary')
    model_path = os.path.join(output_path, "models")
    device = torch.device(device)
    os.makedirs(model_path, exist_ok=True)
    writer = SummaryWriter(log_dir=summary_path)
    
    # Ensure reproducibility (seed management)
    seed = random.randint(1, 1000000000) if cfg.TRAIN.SEED is None else cfg.TRAIN.SEED
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(cfg.TRAIN.USE_DETERMINISTIC_ALGORITHMS)
    torch_gen = torch.Generator().manual_seed(seed)
    np.random.seed(seed)
       
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.TRAIN.MODEL_NAME)
    model = AutoModel.from_pretrained(cfg.TRAIN.MODEL_NAME).to(device)
    
    # Create the dataset and dataloader
    dataset = PatentDataset(data_path)
    indices = list(range(len(dataset)))
    train_ratio, val_ratio = cfg.TRAIN.TRAIN_VAL_SPLIT
    split_train = int(np.floor(train_ratio * len(dataset)))
    split_val = int(np.floor((train_ratio + val_ratio) * len(dataset)))
    train_indices, val_indices = indices[:split_train], indices[split_train: split_val]
    train_sampler = SubsetRandomSampler(train_indices, torch_gen)
    val_sampler = SubsetRandomSampler(val_indices, torch_gen)
    train_loader = DataLoader(dataset, cfg.TRAIN.BATCH_SIZE, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(dataset, cfg.TRAIN.BATCH_SIZE, sampler=val_sampler, num_workers=16)
    print(f"[train] - nb batches for train: {len(train_loader)}; nb batches for validation: {len(val_loader)}")

    # Fine-tuning setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.INIT_LEARNING_RATE)

    for epoch in range(cfg.TRAIN.N_EPOCHS):  # Number of epochs
        
        # Epoch training
        epoch_train_loss = 0
        for i_batch, (query, positive, negative) in enumerate(train_loader):
            start_batch = time.time()
            
             # Tokenize and get embeddings
            query_inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(device)
            positive_inputs = tokenizer(positive, padding=True, truncation=True, return_tensors="pt").to(device)
            negative_inputs = tokenizer(negative, padding=True, truncation=True, return_tensors="pt").to(device)

            query_emb = model(**query_inputs).last_hidden_state.mean(dim=1)
            positive_emb = model(**positive_inputs).last_hidden_state.mean(dim=1)
            negative_emb = model(**negative_inputs).last_hidden_state.mean(dim=1)

            train_loss = F.triplet_margin_loss(query_emb, positive_emb, negative_emb, **cfg.TRAIN.LOSS)
            epoch_train_loss += train_loss
            duration = time.time() - start_batch
            print(f"batch {i_batch+1}/{len(train_loader)}: training_loss={train_loss} (duration: {duration:.2f}s)")
            writer.add_scalar("Loss/train_batch_triplet_loss", train_loss, epoch * len(train_loader) + i_batch)
            
            # Backpropagation
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
        writer.add_scalar("Loss/train_triplet_loss", epoch_train_loss/len(train_loader), epoch)
        
        # Epoch Validation
        val_losses = 0
        model.eval()  # Switch to model eval mode
        start_batch = time.time()
        for i_batch, (query, positive, negative) in enumerate(val_loader):
            with torch.no_grad():
                
                # Tokenize and get embeddings
                query_inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(device)
                positive_inputs = tokenizer(positive, padding=True, truncation=True, return_tensors="pt").to(device)
                negative_inputs = tokenizer(negative, padding=True, truncation=True, return_tensors="pt").to(device)

                query_emb = model(**query_inputs).last_hidden_state.mean(dim=1)
                positive_emb = model(**positive_inputs).last_hidden_state.mean(dim=1)
                negative_emb = model(**negative_inputs).last_hidden_state.mean(dim=1)
                
                # Compute loss
                val_loss = F.triplet_margin_loss(query_emb, positive_emb, negative_emb, **cfg.TRAIN.LOSS)
                duration = time.time() - start_batch
                print(f"batch {i_batch + 1}/{len(val_loader)}: validation_loss={val_loss} (duration: {duration:.2f}s)")
                writer.add_scalar("Loss/val_batch_triplet_loss", val_loss, epoch * len(val_loader) + i_batch)
                val_losses += val_loss
                
            start_batch = time.time()
        writer.add_scalar("Loss/val_triplet_loss", val_losses/len(val_loader), epoch)
        
        # Save the model each cfg.TRAIN.SAVE_MODEL_PERIOD epochs
        if epoch % cfg.TRAIN.SAVE_MODEL_PERIOD == 0:
            model.save_pretrained(model_path)
        
        model.train()


if __name__ == "__main__":
    
    parser = ArgumentParser(description='Zero-shot test of pretrained model')
    parser.add_argument("--cfg_path", "-c", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_path", "-i", type=str, required=True, help="Path to data (expects a .json file)")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Path to save outputs")
    parser.add_argument("--device", "-d", type=str, default="cuda:0", help="Device to use for computations")
    args = parser.parse_args()
    
    train(args.cfg_path, args.data_path, args.output_dir, args.device)
    