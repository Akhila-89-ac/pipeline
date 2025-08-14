from typing import List, Tuple
import torch

def calculate_topk_accuracy(similarity_matrix: torch.Tensor,
                            true_labels: List[str],
                            category_list: List[str],
                            k: int = 10) -> Tuple[float, float, List[List[str]]]:
    # top-k indices per row
    topk = torch.topk(similarity_matrix, k=min(k, similarity_matrix.size(1)), dim=1)
    top_k_indices = topk.indices.cpu().tolist()
    top_k_categories = [[category_list[i] for i in row] for row in top_k_indices]

    correct_top1 = 0
    correct_topk = 0
    n = len(true_labels)

    for i, true_cat in enumerate(true_labels):
        preds = top_k_categories[i]
        if preds and preds[0] == true_cat:
            correct_top1 += 1
        if true_cat in preds:
            correct_topk += 1

    top1_acc = correct_top1 / max(n, 1)
    topk_acc = correct_topk / max(n, 1)
    return top1_acc, topk_acc, top_k_categories

def mean_average_precision(true_labels: List[str], preds_list: List[List[str]]) -> float:
    ap = []
    for true, preds in zip(true_labels, preds_list):
        if true in preds:
            rank = preds.index(true) + 1
            ap.append(1.0 / rank)
        else:
            ap.append(0.0)
    return sum(ap) / max(len(ap), 1)

def mean_reciprocal_rank(true_labels: List[str], preds_list: List[List[str]]) -> float:
    rr = 0.0
    for true, preds in zip(true_labels, preds_list):
        if true in preds:
            rank = preds.index(true) + 1
            rr += 1.0 / rank
    return rr / max(len(true_labels), 1)
