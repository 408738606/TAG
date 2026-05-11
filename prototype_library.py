import argparse
import json
import os
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
import torch

from vlm_localizer import encode_texts

DEFAULT_RANDOM_SEED = 60  # Match KMeans random_state used in TAG localizer.

def load_segments(path: str) -> Tuple[List[str], List[dict], List[str]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    video_ids = []
    segments = []
    descriptions = []
    for vid, items in data.items():
        for seg in items:
            desc = seg.get('description', '').strip()
            if not desc:
                continue
            video_ids.append(vid)
            segments.append({
                'start': float(seg['start']),
                'end': float(seg['end']),
                'description': desc,
            })
            descriptions.append(desc)
    return video_ids, segments, descriptions


def encode_descriptions(descriptions: List[str], batch_size: int = 64, device: str = 'cuda') -> np.ndarray:
    all_embeds = []
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i + batch_size]
        with torch.no_grad():
            embeds = encode_texts(batch, device=device).float().cpu().numpy()
        all_embeds.append(embeds)
    if not all_embeds:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(all_embeds, axis=0)


def build_prototypes(
    embeddings: np.ndarray,
    descriptions: List[str],
    num_clusters: int,
    seed: int = DEFAULT_RANDOM_SEED,
):
    """Cluster segment descriptions with K-Means and return prototype centroids and texts."""
    cluster_count = min(num_clusters, len(descriptions))
    if cluster_count <= 0:
        return np.empty((0, 0), dtype=np.float32), [], np.array([], dtype=np.int64)
    kmeans = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    prototype_texts = []
    for cid in range(cluster_count):
        cluster_indices = np.where(labels == cid)[0]
        cluster_embeds = embeddings[cluster_indices]
        centroid = centroids[cid]
        distances = np.linalg.norm(cluster_embeds - centroid, axis=1)
        best_idx = cluster_indices[np.argmin(distances)]
        prototype_texts.append(descriptions[best_idx])
    return centroids.astype(np.float32), prototype_texts, labels


def save_outputs(
    output_lib: str,
    output_map: str,
    video_ids: List[str],
    segments: List[dict],
    labels: np.ndarray,
    prototype_embeddings: np.ndarray,
    prototype_texts: List[str],
):
    np.savez(output_lib, embeddings=prototype_embeddings, texts=np.array(prototype_texts, dtype=object))

    segment_map = {}
    for vid, seg, label in zip(video_ids, segments, labels):
        entry = {
            'start': seg['start'],
            'end': seg['end'],
            'description': seg['description'],
            'prototype_id': int(label),
        }
        segment_map.setdefault(vid, []).append(entry)

    with open(output_map, 'w', encoding='utf-8') as f:
        json.dump(segment_map, f, ensure_ascii=False, indent=2)


def get_args():
    parser = argparse.ArgumentParser(description='Build an unsupervised event prototype library.')
    parser.add_argument('--segment_desc', required=True, type=str, help='Path to segment description JSON.')
    parser.add_argument('--output_lib', required=True, type=str, help='Output .npz file for prototype library.')
    parser.add_argument('--output_map', required=True, type=str, help='Output JSON mapping segments to prototypes.')
    parser.add_argument('--num_clusters', default=50, type=int, help='Number of prototype clusters.')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for text encoding.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.segment_desc):
        raise FileNotFoundError(f'Segment description file not found: {args.segment_desc}')

    video_ids, segments, descriptions = load_segments(args.segment_desc)
    if not descriptions:
        raise ValueError('No valid descriptions found in segment_desc.')

    embeddings = encode_descriptions(descriptions, batch_size=args.batch_size)
    prototype_embeddings, prototype_texts, labels = build_prototypes(
        embeddings, descriptions, args.num_clusters
    )

    save_outputs(
        args.output_lib,
        args.output_map,
        video_ids,
        segments,
        labels,
        prototype_embeddings,
        prototype_texts,
    )
