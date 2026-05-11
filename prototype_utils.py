import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
import torch
import torch.nn.functional as F

from vlm_localizer import encode_texts


@dataclass
class PrototypeLibrary:
    embeddings: torch.Tensor
    texts: List[str]


class SegmentEntry(TypedDict, total=False):
    start: float
    end: float
    description: str
    prototype_id: int


class PrototypeExplanation(TypedDict, total=False):
    prototype_id: int
    prototype_text: str
    prototype_score: float
    segment_start: float
    segment_end: float
    segment_text: str


def load_prototype_library(path: str, device: str = 'cuda') -> PrototypeLibrary:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f'Prototype library not found: {path}')
    if path.endswith('.npz'):
        data = np.load(path, allow_pickle=True)
        embeddings = torch.tensor(data['embeddings'], device=device)
        texts = data['texts'].tolist()
        return PrototypeLibrary(embeddings=embeddings, texts=texts)
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        embeddings = torch.tensor(payload['embeddings'], device=device)
        texts = payload['texts']
        return PrototypeLibrary(embeddings=embeddings, texts=texts)
    raise ValueError(f'Unsupported prototype library format: {path}')


def load_segment_prototypes(path: str) -> Dict[str, List[SegmentEntry]]:
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def segment_iou(seg_a: Tuple[float, float], seg_b: Tuple[float, float]) -> float:
    start = max(seg_a[0], seg_b[0])
    end = min(seg_a[1], seg_b[1])
    inter = max(end - start, 0.0)
    len_a = max(seg_a[1] - seg_a[0], 0.0)
    len_b = max(seg_b[1] - seg_b[0], 0.0)
    union = len_a + len_b - inter
    return inter / union if union > 0 else 0.0


def _assign_segment(proposal: Tuple[float, float], segments: List[SegmentEntry]) -> Optional[SegmentEntry]:
    if not segments:
        return None
    best_seg = None
    best_iou = -1.0
    for seg in segments:
        seg_start = float(seg['start'])
        seg_end = float(seg['end'])
        iou = segment_iou((seg_start, seg_end), proposal)
        if iou > best_iou:
            best_iou = iou
            best_seg = seg
    return best_seg


def apply_prototype_guidance(
    proposals: List[List[float]],
    query: str,
    prototype_lib: PrototypeLibrary,
    segment_map: Dict[str, List[SegmentEntry]],
    video_id: str,
    weight: float = 0.2,
    device: str = 'cuda',
) -> Tuple[List[List[float]], Optional[PrototypeExplanation]]:
    if not proposals or not prototype_lib or video_id not in segment_map:
        return proposals, None

    proto_embeddings = prototype_lib.embeddings.to(device)
    proto_texts = prototype_lib.texts
    if proto_embeddings.numel() == 0:
        return proposals, None

    with torch.no_grad():
        query_emb = encode_texts([query], device=device)
        query_emb = F.normalize(query_emb, dim=-1)
        proto_emb = F.normalize(proto_embeddings, dim=-1)
        proto_scores = (query_emb @ proto_emb.t()).squeeze(0)

    updated = []
    best_explanation = None
    best_score = float('-inf')
    segments = segment_map.get(video_id, [])

    for proposal in proposals:
        start, end, score = proposal
        seg = _assign_segment((start, end), segments)
        if seg is None:
            updated.append([start, end, score])
            continue
        proto_id = int(seg.get('prototype_id', -1))
        if proto_id < 0 or proto_id >= proto_scores.numel():
            updated.append([start, end, score])
            continue
        proto_score = proto_scores[proto_id].item()
        adjusted_score = float(score) * (1 + weight * proto_score)
        updated.append([start, end, adjusted_score])

        if adjusted_score > best_score:
            best_score = adjusted_score
            best_explanation = {
                'prototype_id': proto_id,
                'prototype_text': proto_texts[proto_id],
                'prototype_score': proto_score,
                'segment_start': seg.get('start'),
                'segment_end': seg.get('end'),
                'segment_text': seg.get('description'),
            }

    return updated, best_explanation
