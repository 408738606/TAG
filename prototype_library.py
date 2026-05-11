from dataclasses import dataclass
import json
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from sklearn.feature_extraction.text import TfidfVectorizer

SegmentDescription = Dict[str, Union[float, str]]
SegmentMap = Dict[str, List[SegmentDescription]]


@dataclass
class PrototypeLibrary:
    vectorizer: "TfidfVectorizer"
    prototype_descriptions: List[str]
    prototype_vectors: np.ndarray


def load_segment_descriptions(path: str) -> SegmentMap:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if all(isinstance(value, list) for value in data.values()):
            normalized_map: SegmentMap = {}
            for vid, segments in data.items():
                normalized_segments = []
                for segment in segments:
                    if isinstance(segment, dict):
                        normalized = _normalize_segment_entry(segment)
                        if normalized:
                            normalized_segments.append(normalized)
                if normalized_segments:
                    normalized_map[str(vid)] = normalized_segments
            return normalized_map
        if "segments" in data and isinstance(data["segments"], list):
            data = data["segments"]

    segment_map: SegmentMap = {}
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            vid = entry.get("video_id") or entry.get("vid") or entry.get("video")
            if vid is None:
                continue
            segment = _normalize_segment_entry(entry)
            if segment is None:
                continue
            segment_map.setdefault(str(vid), []).append(segment)

    return segment_map


def _normalize_segment_entry(entry: Dict[str, object]) -> Optional[SegmentDescription]:
    description = entry.get("description") or entry.get("caption") or entry.get("text")
    timestamp = entry.get("timestamp") or entry.get("timestamps")
    start = entry.get("start")
    end = entry.get("end")
    if timestamp and isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
        start = start if start is not None else timestamp[0]
        end = end if end is not None else timestamp[1]
    if description is None or start is None or end is None:
        return None
    return {"start": float(start), "end": float(end), "description": str(description)}


def flatten_descriptions(segment_map: SegmentMap) -> List[str]:
    descriptions: List[str] = []
    for segments in segment_map.values():
        for segment in segments:
            description = segment.get("description")
            if description:
                descriptions.append(str(description))
    return descriptions


def build_prototype_library(
    segment_map: SegmentMap,
    num_prototypes: int = 120,
    random_state: int = 60,
    max_features: int = 5000,
) -> Optional[PrototypeLibrary]:
    descriptions = flatten_descriptions(segment_map)
    if not descriptions:
        return None

    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=max_features)
    tfidf = vectorizer.fit_transform(descriptions)

    if len(descriptions) <= num_prototypes:
        prototype_indices = list(range(len(descriptions)))
    else:
        kmeans = KMeans(n_clusters=num_prototypes, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(tfidf)
        prototype_indices = []
        for cluster_id in range(num_prototypes):
            cluster_indices = np.where(labels == cluster_id)[0]
            if cluster_indices.size == 0:
                continue
            centroid = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
            sims = cosine_similarity(tfidf[cluster_indices], centroid).reshape(-1)
            best_idx = cluster_indices[sims.argmax()]
            prototype_indices.append(best_idx)

    prototype_descriptions = [descriptions[idx] for idx in prototype_indices]
    prototype_vectors = vectorizer.transform(prototype_descriptions).toarray()

    return PrototypeLibrary(
        vectorizer=vectorizer,
        prototype_descriptions=prototype_descriptions,
        prototype_vectors=prototype_vectors,
    )


def save_prototype_library(path: str, library: PrototypeLibrary) -> None:
    payload = {"prototypes": library.prototype_descriptions}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_prototype_library(path: str, max_features: int = 5000) -> Optional[PrototypeLibrary]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    prototypes = payload.get("prototypes")
    if not prototypes:
        return None

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=max_features)
    prototype_vectors = vectorizer.fit_transform(prototypes).toarray()
    return PrototypeLibrary(
        vectorizer=vectorizer,
        prototype_descriptions=list(prototypes),
        prototype_vectors=prototype_vectors,
    )


def score_query_to_prototypes(query: str, library: PrototypeLibrary) -> Tuple[Optional[str], float]:
    if library is None or not library.prototype_descriptions:
        return None, 0.0
    from sklearn.metrics.pairwise import cosine_similarity

    query_vec = library.vectorizer.transform([query]).toarray()
    sims = cosine_similarity(query_vec, library.prototype_vectors).reshape(-1)
    best_idx = sims.argmax()
    return library.prototype_descriptions[best_idx], float(sims[best_idx])


def score_query_to_texts(query: str, texts: List[str], library: PrototypeLibrary) -> np.ndarray:
    if library is None or not texts:
        return np.zeros(len(texts))
    from sklearn.metrics.pairwise import cosine_similarity

    query_vec = library.vectorizer.transform([query])
    text_vecs = library.vectorizer.transform(texts)
    return cosine_similarity(query_vec, text_vecs).reshape(-1)


def rerank_proposals_with_prototypes(
    proposals: List[List[float]],
    query: str,
    segment_descriptions: List[SegmentDescription],
    library: PrototypeLibrary,
    weight: float = 0.2,
) -> Tuple[List[List[float]], Optional[Dict[str, object]]]:
    if not proposals or not segment_descriptions or library is None:
        return proposals, None

    matched_descriptions: List[str] = []
    for proposal in proposals:
        start, end = proposal[0], proposal[1]
        best_desc = None
        best_iou = 0.0
        for segment in segment_descriptions:
            iou = _temporal_iou(start, end, float(segment["start"]), float(segment["end"]))
            if iou > best_iou:
                best_iou = iou
                best_desc = segment.get("description")
        matched_descriptions.append(best_desc if best_desc else "")

    sims = score_query_to_texts(query, matched_descriptions, library)
    reranked = []
    for proposal, sim in zip(proposals, sims):
        adjusted = proposal[2] + weight * float(sim)
        reranked.append([proposal[0], proposal[1], adjusted])

    prototype_desc, prototype_score = score_query_to_prototypes(query, library)
    evidence = None
    if prototype_desc:
        evidence = {"description": prototype_desc, "score": float(prototype_score)}

    return reranked, evidence


def _temporal_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    inter = min(end_a, end_b) - max(start_a, start_b)
    if inter <= 0:
        return 0.0
    union = max(end_a, end_b) - min(start_a, start_b)
    return float(inter / union) if union > 0 else 0.0
