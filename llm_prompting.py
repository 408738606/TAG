import numpy as np
import re
from typing import Dict, List, Optional

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


_DEFAULT_SYNONYMS: Dict[str, str] = {
    "commence": "start",
    "purchase": "buy",
    "observe": "watch",
    "relocate": "move",
    "assists": "helps",
    "assisting": "helping",
    "utilize": "use",
    "speaking": "talking",
    "child": "kid",
    "sofa": "couch",
    "automobile": "car",
}

_SHORTEN_PREFIX_TOKENS = 3
_SHORTEN_SUFFIX_TOKENS = 2


def normalize_query(query: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", query)
    return " ".join(cleaned.lower().split())


def heuristic_debias_variants(
    query: str,
    max_variants: int = 3,
    synonym_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    variants: List[str] = []
    normalized = normalize_query(query)
    if normalized and normalized != query:
        variants.append(normalized)

    tokens = normalized.split()
    synonym_map = synonym_map or _DEFAULT_SYNONYMS
    if tokens:
        replaced = [synonym_map.get(token, token) for token in tokens]
        replaced_query = " ".join(replaced)
        if replaced_query and replaced_query not in variants and replaced_query != query:
            variants.append(replaced_query)

    if len(tokens) > _SHORTEN_PREFIX_TOKENS + _SHORTEN_SUFFIX_TOKENS:
        shortened = " ".join(tokens[:_SHORTEN_PREFIX_TOKENS] + tokens[-_SHORTEN_SUFFIX_TOKENS:])
        if shortened and shortened not in variants and shortened != query:
            variants.append(shortened)

    return variants[:max_variants]


def filter_queries_by_frame_similarity(
    queries: List[str],
    frame_descriptions: List[str],
    min_similarity: float = 0.15,
) -> List[str]:
    if not frame_descriptions:
        return queries

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = queries + frame_descriptions
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(corpus)
    query_vecs = vectors[: len(queries)]
    frame_vecs = vectors[len(queries) :]

    sims = cosine_similarity(query_vecs, frame_vecs).max(axis=1)
    filtered = [query for query, score in zip(queries, sims) if score >= min_similarity]
    return filtered if filtered else queries[:1]


def get_debiased_queries(
    query: str,
    response_entry: Optional[Dict[str, object]] = None,
    frame_descriptions: Optional[List[str]] = None,
    max_variants: int = 3,
    min_similarity: float = 0.15,
    synonym_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    # min_similarity controls how strict the cross-modal consistency filter is.
    variants: List[str] = [query]

    if response_entry:
        llm_queries = response_entry.get("query_json")
        if isinstance(llm_queries, list) and llm_queries:
            descriptions = llm_queries[0].get("descriptions", [])
            variants.extend([str(item) for item in descriptions if item])
        extra_queries = response_entry.get("debiased_queries")
        if isinstance(extra_queries, list):
            variants.extend([str(item) for item in extra_queries if item])

    variants.extend(
        heuristic_debias_variants(query, max_variants=max_variants, synonym_map=synonym_map)
    )

    unique_variants = []
    seen = set()
    for variant in variants:
        cleaned = " ".join(str(variant).split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique_variants.append(cleaned)

    if frame_descriptions:
        unique_variants = filter_queries_by_frame_similarity(unique_variants, frame_descriptions, min_similarity)

    return unique_variants[:max_variants]
