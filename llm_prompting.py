import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, TypedDict
from vlm_localizer import encode_texts

class QuerySelectionMeta(TypedDict, total=False):
    reason: str
    best_score: float

def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union


def select_proposal(inputs, gamma=0.6):
    weights = inputs[:, -1].clip(min=0)
    proposals = inputs[:, :-1]
    scores = np.zeros_like(weights)

    for j in range(scores.shape[0]):
        iou = calc_iou(proposals, proposals[j])
        scores[j] += (iou ** gamma * weights).sum()

    idx = np.argsort(-scores)
    return inputs[idx]

searched_proposals = []
def search_combination(cands, idx, cur=[], relation='sequentially'):
    if idx >= len(cands):
        cur = np.array(cur)

        if relation == 'simultaneously' and cur.max(axis=0, keepdims=True)[:, 0] > cur.min(axis=0, keepdims=True)[:, 1]:
            return
        st = cur.min(axis=0, keepdims=True)[:, 0]
        end = cur.max(axis=0, keepdims=True)[:, 1]
        score = cur[:, -1].clip(min=0).prod()
        global searched_proposals
        searched_proposals.append([float(st), float(end), float(score)])
        return
    
    for cur_idx in range(len(cands[idx])):
        if len(cur) > 0 and relation == 'sequentially' and cands[idx][cur_idx][0] < cur[-1][1]:
            continue

        search_combination(cands, idx+1, cur + [cands[idx][cur_idx]], relation)


def filter_and_integrate(sub_query_proposals, relation):
    if len(sub_query_proposals) == 0:
        return []
    global searched_proposals
    searched_proposals = []
    search_combination(sub_query_proposals, 0, cur=[], relation=relation)
    if len(searched_proposals) == 0:
        return []
    proposals = select_proposal(np.array(searched_proposals))

    return proposals.tolist()[:2]


def select_debiased_query(
    candidates: List[str],
    visual_descriptions: Optional[List[str]] = None,
    min_similarity: float = 0.2,
    device: str = 'cuda',
) -> Tuple[Optional[str], Dict[str, object]]:
) -> Tuple[Optional[str], QuerySelectionMeta]:
    """Select the query candidate most similar to visual descriptions.

    Args:
        candidates: Candidate query rewrites (first item should be the original query).
        visual_descriptions: Keyframe/segment descriptions for cross-modal verification.
        min_similarity: Minimum similarity threshold to accept a rewritten query.
        device: Torch device for text encoding.

    Returns:
        (selected_query, metadata) where metadata includes the selection reason and score.
    """
    if not candidates:
        return None, {'reason': 'no_candidates'}
    if not visual_descriptions:
        return candidates[0], {'reason': 'no_visual_descriptions'}

    with torch.no_grad():
        cand_emb = encode_texts(candidates, device=device)
        desc_emb = encode_texts(visual_descriptions, device=device)
        cand_emb = F.normalize(cand_emb, dim=-1)
        desc_emb = F.normalize(desc_emb, dim=-1)
        scores = cand_emb @ desc_emb.t()
        scores = scores.mean(dim=1)

    best_idx = torch.argmax(scores).item()
    best_score = scores[best_idx].item()
    if best_score < min_similarity:
        return candidates[0], {'reason': 'low_similarity', 'best_score': best_score}

    return candidates[best_idx], {'reason': 'selected', 'best_score': best_score}
