from joblib import Parallel, delayed
import torch
from torch import Tensor
from tqdm import tqdm


def _precision_recall_single(relevant_tensor, pred_tensor, k) -> tuple[float, float]:
    if (num_relevant_items := len(relevant_tensor)) == 0:
        return torch.nan
    correctly_pred = torch.isin(pred_tensor[:k], relevant_tensor).sum().float()
    return correctly_pred / k, correctly_pred / num_relevant_items

def precision_recall_at_k(y_true, y_pred, k, n_jobs=-1) -> tuple[float, float]:
    precisions_recalls = Parallel(n_jobs=n_jobs)(
        delayed(_precision_recall_single)(ts, pred, k)
        for ts, pred in tqdm(zip(y_true, y_pred),
                             desc=f'Calculating Precision@{k} and Recall@{k}',
                             total=len(y_true),
                             leave=False)
    )
    return tuple(torch.tensor(precisions_recalls).nanmean(dim=0).tolist())

def _average_precision_single_full(relevant_ids, predicted_ids, k = None):
    """
    Calculate mean Average Precision (mAP) for recommendation systems.

    Args:
        relevant_ids: Tensor of relevant item IDs (ground truth)
        predicted_ids: Tensor of predicted item IDs sorted by relevance
        k: Optional cutoff for top-k items to consider. If None, use all predicted items.

    Returns:
        mAP score
    """
    if (num_relevant := len(relevant_ids)) == 0:
        return torch.nan

    if k is not None:
        predicted_ids = predicted_ids[:k]

    # Create binary relevance tensor (1 for relevant items, 0 otherwise)
    relevance = torch.isin(predicted_ids, relevant_ids).float()

    # Calculate precision at each position
    cum_relevance = torch.cumsum(relevance, dim=0)
    positions = torch.arange(1, len(predicted_ids) + 1)
    precision_at_k = cum_relevance / positions

    # Only consider precision at positions where item is relevant
    relevant_precision = precision_at_k * relevance

    # Avoid division by zero if there are no relevant items

    # Sum relevant precisions and divide by total relevant items
    ap = torch.sum(relevant_precision) / num_relevant

    return ap.item()

def mean_average_precision(y_true: list[Tensor],
                           y_pred: list[Tensor],
                           k: int | None = None,
                           n_jobs: int = -1) -> float:
    """
    Compute MAP over full prediction list.
    """
    ap_scores = Parallel(n_jobs=n_jobs)(
        delayed(_average_precision_single_full)(true, pred, k)
        for true, pred in tqdm(zip(y_true, y_pred),
                               desc=f'Calculating mAP{f"@{k}" if k is not None else ""}',
                               total=len(y_true),
                               leave=False)
    )

    return torch.nanmean(torch.tensor(ap_scores, dtype=torch.float)).item()