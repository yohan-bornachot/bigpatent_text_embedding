from argparse import ArgumentParser

import torch
import yaml

from patent_dataset import PatentDataset
from metrics import compute_metrics, plot_metrics
from utils import add_attr_interface
from transformers import AutoTokenizer, AutoModel


def get_embeddings(text_list, tokenizer, model, batch_size: int = 8):
    ret = []
    batch_cnt = 0
    n_batches = len(text_list) // batch_size + int(len(text_list) % batch_size != 0)
    for sample_idx in range(0, len(text_list), batch_size):
        print(f"[get_embeddings] - processing batch n°{batch_cnt + 1}/{n_batches}...")
        inputs = tokenizer(text_list[sample_idx: sample_idx + batch_size], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        res = outputs.last_hidden_state.mean(dim=1)  # Average pooling
        ret.append(res)
        batch_cnt += 1
    ret = torch.concat(ret)
    return ret


def do_zero_shot(tokenizer, model, df, batch_size):
    print("\n[do_zero_shot] - Processing query embeddings...")
    query_embeddings = get_embeddings(df['query'].tolist(), tokenizer, model, batch_size)
    print("\n[do_zero_shot] - Processing positives embeddings...")
    positive_embeddings = get_embeddings(df['pos'].tolist(), tokenizer, model,batch_size)
    print("\n[do_zero_shot] - Processing negatives embeddings...")
    negative_embeddings = get_embeddings(df['negative'].tolist(), tokenizer, model, batch_size)
    return query_embeddings, positive_embeddings, negative_embeddings


if __name__ == "__main__":
    
    parser = ArgumentParser(description='Zero-shot test of pretrained model')
    parser.add_argument("--cfg_path", "-c", type=str, required=True, help="Path to config file for registration")
    parser.add_argument("--data_path", "-m", type=str, required=True, help="Path to data file (expects a .json file)")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory to store results")
    parser.add_argument("--device", "-d", type=str, default="cuda:0", help="Device to use for computations")
    args = parser.parse_args()
    
    
    # Load config
    with open(args.cfg_path, 'r') as yml_file:
        cfg = add_attr_interface(yaml.safe_load(yml_file))

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.MODEL_NAME)
    model = AutoModel.from_pretrained(cfg.MODEL.MODEL_NAME)
    
    dataset = PatentDataset(args.data_path)
    query_embeddings, positive_embeddings, negative_embeddings = do_zero_shot(tokenizer, model, dataset.df, cfg.DATA.BATCH_SIZE)
    metrics = compute_metrics(query_embeddings, positive_embeddings, negative_embeddings)
    print("metrics = ", metrics )
    plot_metrics(metrics)
    