import pickle
from argparse import ArgumentParser

import torch
import yaml
from metrics import compute_metrics, visualize_similarity_distrib
from patent_dataset import PatentDataset
from transformers import AutoTokenizer, AutoModel
from utils import add_attr_interface


def get_embeddings(text_list, tokenizer, model, batch_size: int = 8, device: str = "cpu"):
    ret = []
    batch_cnt = 0
    n_batches = len(text_list) // batch_size + int(len(text_list) % batch_size != 0)
    for sample_idx in range(0, len(text_list), batch_size):
        print(f"[get_embeddings] - processing batch n°{batch_cnt + 1}/{n_batches}...")
        inputs = tokenizer(text_list[sample_idx: sample_idx + batch_size], padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        res = outputs.last_hidden_state.mean(dim=1)  # Average pooling
        ret.append(res)
        batch_cnt += 1
    ret = torch.concat(ret)
    return ret


def encode_all(tokenizer, model, df, batch_size, device: str = 'cpu'):
    with torch.no_grad():
        print("\n[encode_all] - Processing query embeddings...")
        query_embeddings = get_embeddings(df['query'].tolist(), tokenizer, model, batch_size, device)
        print("\n[encode_all] - Processing positives embeddings...")
        positive_embeddings = get_embeddings(df['pos'].tolist(), tokenizer, model, batch_size, device)
        print("\n[encode_all] - Processing negatives embeddings...")
        negative_embeddings = get_embeddings(df['negative'].tolist(), tokenizer, model, batch_size, device)
    return query_embeddings, positive_embeddings, negative_embeddings


if __name__ == "__main__":
    
    parser = ArgumentParser(description='Zero-shot test of pretrained model')
    parser.add_argument("--cfg_path", "-c", type=str, required=True, help="Path to config file")
    parser.add_argument("--pretrained_model", "-c", type=str, required=True, help="Path or name of model file")
    parser.add_argument("--data_path", "-m", type=str, required=True, help="Path to dataset file (expects a .json file)")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory to store results")
    parser.add_argument("--device", "-d", type=str, default="cuda:0", help="Device to use for computations")
    args = parser.parse_args()

    # Load config
    with open(args.cfg_path, 'r') as yml_file:
        cfg = add_attr_interface(yaml.safe_load(yml_file))

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.TRAIN.TOKENIZER_NAME)
    model = AutoModel.from_pretrained(args.pretrained_model).to(args.device)

    # Load dataset
    dataset = PatentDataset(args.data_path)

    # Load test indices in split file if it exists, otherwise creates it
    split_path = os.path.join(args.data_path, "indices_split.pkl")
    if os.path.exists(split_path):
        with open(split_path, 'rb') as pkl_file:
            test_indices = pickle.load(pkl_file)['test_indices']
    else:
        indices = list(range(len(dataset)))
        train_ratio, val_ratio = cfg.TRAIN.TRAIN_VAL_SPLIT
        split_train = int(np.floor(train_ratio * len(dataset)))
        split_val = int(np.floor((train_ratio + val_ratio) * len(dataset)))
        train_indices, val_indices = indices[:split_train], indices[split_train: split_val]
        test_indices = indices[split_val:]
        with open(os.path.join(os.path.dirname(data_path), "indices_split.pkl"), 'wb') as pkl_file:
            indices_dict = {"train_indices": train_indices, "val_indices": val_indices,
                            "test_indices": test_indices}
            pickle.dump(indices_dict, pkl_file)

    query_embeddings, positive_embeddings, negative_embeddings = encode_all(tokenizer, model,
                                                                            dataset.df.iloc[test_indices],
                                                                            cfg.TRAIN.BATCH_SIZE, args.device)
    # Put back tensors on cpu before computing metrics
    query_embeddings = query_embeddings.cpu()
    positive_embeddings = positive_embeddings.cpu()
    negative_embeddings = negative_embeddings.cpu()

    # Compute metrics
    metrics = compute_metrics(query_embeddings, positive_embeddings, negative_embeddings)
    print("metrics = ", metrics)

    # Plot results
    visualize_similarity_distrib(metrics)
    