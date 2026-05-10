import numpy as np
import re

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


_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "with",
    "for", "from", "by", "is", "are", "was", "were", "be", "been", "being",
    "into", "onto", "over", "under", "up", "down", "off", "out", "about",
    "near", "around", "as",
}

_SYNONYM_MAP = {
    "sofa": ["couch"],
    "tv": ["television"],
    "cellphone": ["cell phone", "mobile"],
    "kitchen": ["cooking area"],
    "cup": ["mug"],
    "fridge": ["refrigerator"],
    "bike": ["bicycle"],
    "kleenex": ["tissue"],
    "cabinet": ["cupboard"],
    "sneakers": ["shoes"],
}


def _normalize_query(text):
    text = text.strip()
    if not text:
        return text
    text = text.replace("\n", " ").replace("\t", " ")
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def expand_queries(text, max_variants=3):
    """Expand a query with normalization, synonym replacement, and stopword filtering."""
    base = text.strip() if isinstance(text, str) else str(text).strip()
    if not base:
        return []

    normalized = _normalize_query(base)
    variants = [base]
    if normalized and normalized != base:
        variants.append(normalized)

    tokens = normalized.split() if normalized else []
    for idx, token in enumerate(tokens):
        if token in _SYNONYM_MAP:
            for synonym in _SYNONYM_MAP[token]:
                new_tokens = tokens.copy()
                new_tokens[idx] = synonym
                variants.append(" ".join(new_tokens))

    if tokens:
        content_tokens = [token for token in tokens if token not in _STOPWORDS]
        if content_tokens:
            variants.append(" ".join(content_tokens))

    deduped = []
    for variant in variants:
        variant = variant.strip()
        if variant and variant not in deduped:
            deduped.append(variant)
        if len(deduped) >= max_variants:
            break

    return deduped
