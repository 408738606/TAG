from data_configs import DATASETS
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from vlm_localizer import localize
from llm_prompting import get_debiased_queries, select_proposal
from prototype_library import (
    build_prototype_library,
    load_prototype_library,
    load_segment_descriptions,
    save_prototype_library,
)

DEFAULT_DEBIAS_CONFIG = {
    "enabled": False,
    "max_variants": 3,
    "min_similarity": 0.15,
}

DEFAULT_PROTOTYPE_CONFIG = {
    "enabled": False,
    "num_prototypes": 120,
    "weight": 0.2,
    "max_features": 5000,
}

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split. See supported splits in data_configs.py.')
    parser.add_argument('--use_llm', action='store_true', help='Enable use llm')
    parser.add_argument('--use_debias', action='store_true', help='Enable query debiasing and cross-modal verification.')
    parser.add_argument('--tckmeans', action='store_true', help='Enable use GPU KMeans')
    parser.add_argument('--llm_output', default=None, type=str, help='LLM prompt output. If not specified, use nonly VLM for evaluation.')
    parser.add_argument('--frame_desc', default=None, type=str, help='Keyframe description JSON for cross-modal verification.')
    parser.add_argument('--segment_desc', default=None, type=str, help='Segment description JSON for prototype guidance.')
    parser.add_argument('--prototype_cache', default=None, type=str, help='Cache path for prototype library descriptions.')

    return parser.parse_args()


def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union

def eval(
    data,
    feature_path,
    stride,
    hyperparams,
    use_llm,
    tckmeans,
    pad_sec=0.0,
    use_debias=False,
    frame_desc_map=None,
    prototype_context=None,
    segment_desc_map=None,
    debias_config=None,
    prototype_config=None,
):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])

    debias_cfg = merge_config(DEFAULT_DEBIAS_CONFIG, debias_config)
    debias_cfg["enabled"] = debias_cfg["enabled"] or use_debias
    proto_cfg = merge_config(DEFAULT_PROTOTYPE_CONFIG, prototype_config)
    
    pbar = tqdm(data.items())
    for vid, ann in pbar:
        duration = ann['duration'] if 'duration' in ann else ann['video_duration']
        video_feature_path = os.path.join(feature_path, vid+'.npy')
        video_feature = np.load(video_feature_path)
        if pad_sec > 0:
            pad_noise = np.random.randn(round(video_feature.shape[0] / duration * pad_sec), video_feature.shape[1], video_feature.shape[2])
            video_feature = np.concatenate([pad_noise, video_feature], axis=0)
            duration += pad_sec

        for i in range(len(ann['sentences'])):
            gt = ann['timestamps'][i]
            response_entry = None
            if use_llm and 'response' in ann and i < len(ann['response']):
                response_entry = ann['response'][i]

            frame_descriptions = extract_frame_descriptions(ann, response_entry, frame_desc_map, vid)
            if use_llm or debias_cfg["enabled"]:
                queries = get_debiased_queries(
                    ann['sentences'][i],
                    response_entry=response_entry if use_llm else None,
                    frame_descriptions=frame_descriptions,
                    max_variants=debias_cfg["max_variants"],
                    min_similarity=debias_cfg["min_similarity"],
                )
            else:
                queries = [ann['sentences'][i]]

            proposals = []
            for query_text in queries:
                query_json = [{'descriptions': query_text}]
                proposals += localize(
                    video_feature,
                    duration,
                    query_json,
                    stride,
                    hyperparams,
                    tckmeans,
                    prototype_context=prototype_context if proto_cfg["enabled"] else None,
                    segment_descriptions=segment_desc_map.get(vid) if segment_desc_map else None,
                )

            proposals = select_proposal(np.array(proposals))

            s, e = ann['timestamps'][i]
            s, e = s + pad_sec, e + pad_sec

            sp, ep = proposals[0][0],  proposals[0][1]

            iou_ = (min(e, ep) - max(s, sp)) / (max(e, ep) - min(s, sp))
            ious.append(max(iou_, 0))
            recall += thresh <= iou_
        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))


def merge_config(defaults, overrides):
    config = dict(defaults)
    if overrides:
        config.update(overrides)
    return config


def load_frame_descriptions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return {key: normalize_description_list(value) for key, value in data.items()}

    descriptions = {}
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            vid = entry.get("video_id") or entry.get("vid") or entry.get("video")
            if vid is None:
                continue
            description_list = entry.get("descriptions") or entry.get("frames") or entry.get("frame_descriptions")
            descriptions[str(vid)] = normalize_description_list(description_list)
    return descriptions


def normalize_description_list(value):
    if value is None:
        return []
    if isinstance(value, dict):
        value = value.get("descriptions") or value.get("frames")
        if value is None:
            return []
    if not isinstance(value, list):
        return []
    descriptions = []
    for item in value:
        if isinstance(item, dict):
            desc = item.get("description") or item.get("caption") or item.get("text")
            if desc:
                descriptions.append(str(desc))
        elif item:
            descriptions.append(str(item))
    return descriptions


def extract_frame_descriptions(ann, response_entry, frame_desc_map, vid):
    if response_entry and isinstance(response_entry, dict):
        response_desc = response_entry.get("frame_descriptions")
        if response_desc:
            return normalize_description_list(response_desc)
    ann_desc = ann.get("frame_descriptions")
    if ann_desc:
        return normalize_description_list(ann_desc)
    if frame_desc_map and vid in frame_desc_map:
        return normalize_description_list(frame_desc_map[vid])
    return []



if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)


    if args.llm_output and os.path.exists(args.llm_output):
        with open(args.llm_output) as f:
            data = json.load(f)
    else:
        with open(dataset['splits'][args.split]['annotation_file']) as f:
            data = json.load(f)

    frame_desc_map = None
    if args.frame_desc and os.path.exists(args.frame_desc):
        frame_desc_map = load_frame_descriptions(args.frame_desc)

    segment_desc_map = None
    if args.segment_desc and os.path.exists(args.segment_desc):
        segment_desc_map = load_segment_descriptions(args.segment_desc)

    hyperparams = dataset["hyper_parameters"]
    debias_cfg = merge_config(DEFAULT_DEBIAS_CONFIG, hyperparams.get("debias"))
    proto_override = hyperparams.get("prototype") or {}
    proto_cfg = merge_config(DEFAULT_PROTOTYPE_CONFIG, proto_override)
    if segment_desc_map and "enabled" not in proto_override:
        proto_cfg["enabled"] = True

    prototype_context = None
    if segment_desc_map and proto_cfg.get("enabled", False):
        if args.prototype_cache and os.path.exists(args.prototype_cache):
            prototype_context = load_prototype_library(args.prototype_cache, max_features=proto_cfg["max_features"])
        else:
            prototype_context = build_prototype_library(
                segment_desc_map,
                num_prototypes=proto_cfg["num_prototypes"],
                max_features=proto_cfg["max_features"],
            )
            if args.prototype_cache and prototype_context:
                save_prototype_library(args.prototype_cache, prototype_context)

    eval(
        data,
        dataset['feature_path'],
        dataset['stride'],
        hyperparams,
        args.use_llm,
        args.tckmeans,
        dataset['splits'][args.split]['pad_sec'],
        use_debias=args.use_debias,
        frame_desc_map=frame_desc_map,
        prototype_context=prototype_context,
        segment_desc_map=segment_desc_map,
        debias_config=debias_cfg,
        prototype_config=proto_cfg,
    )
