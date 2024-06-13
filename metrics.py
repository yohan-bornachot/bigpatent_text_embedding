import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


def mean_reciprocal_rank(pos_sim, neg_sim):
    ranks = []
    for pos, neg in zip(pos_sim, neg_sim):
        all_sims = np.concatenate([pos, neg])
        sorted_indices = np.argsort(-all_sims)  # Descending order
        rank = np.where(sorted_indices < len(pos))[0][0] + 1  # 1-based index
        ranks.append(1.0 / rank)
    return np.mean(ranks)


def compute_metrics(query_embeddings, positive_embeddings, negative_embeddings):
        
    # Compute similarities
    pos_similarities = cosine_similarity(query_embeddings, positive_embeddings)
    neg_similarities = cosine_similarity(query_embeddings, negative_embeddings)
    
    # Compute Mean Reciprocal Rank (MRR)
    mrr_score = mean_reciprocal_rank(pos_similarities, neg_similarities)
    print(f"MRR Score: {mrr_score}")
    
    # Get similarities for corresponding pairs
    pos_similarities = pos_similarities.diagonal()
    neg_similarities = neg_similarities.diagonal()

    # Compute accuracy, precision, recall, and f1-score
    is_correct = pos_similarities > neg_similarities
    accuracy = is_correct.mean()
    print(f'Accuracy: {accuracy}')
    
    metrics = {
        "pos_similarities": pos_similarities,
        "neg_similarities": neg_similarities,
        "is_correct": is_correct,
        "mrr_score": mrr_score,
        "accuracy": accuracy,
    }
    return metrics


def visualize_similarity_distrib(metrics: dict):
    
    # Plot the distribution of positive and negative similarities
    plt.figure(figsize=(12, 6))
    sns.histplot(metrics['pos_similarities'], bins=50, color='blue', label='Positive Similarity', kde=True)
    sns.histplot(metrics['neg_similarities'], bins=50, color='green', label='Negative Similarity', kde=True)
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
