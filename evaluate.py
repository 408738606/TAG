from data_configs import DATASETS
import argparse
import numpy as np
import json
import torch
from tqdm import tqdm
from vlm_localizer import localize
import os
from llm_prompting import select_proposal, select_debiased_query
from prototype_utils import apply_prototype_guidance, load_prototype_library, load_segment_prototypes

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split. See supported splits in data_configs.py.')
    parser.add_argument('--use_llm', action='store_true', help='Enable use llm')
    parser.add_argument('--tckmeans', action='store_true', help='Enable use GPU KMeans')
    parser.add_argument('--llm_output', default=None, type=str, help='LLM prompt output. If not specified, use nonly VLM for evaluation.')
    parser.add_argument('--debias_queries', action='store_true', help='Enable LLM query debias with cross-modal verification.')
    parser.add_argument('--visual_desc', default=None, type=str, help='Path to keyframe description JSON for query verification.')
    parser.add_argument('--debias_min_sim', default=0.2, type=float, help='Minimum similarity to accept a debiased query.')
    parser.add_argument('--prototype_lib', default=None, type=str, help='Path to prototype library (.npz or .json).')
    parser.add_argument('--segment_proto_map', default=None, type=str, help='Path to segment-to-prototype mapping JSON.')
    parser.add_argument('--prototype_weight', default=0.2, type=float, help='Prototype guidance strength.')
    parser.add_argument('--save_predictions', default=None, type=str, help='Optional path to save predictions with explanations.')

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
    debias_queries=False,
    visual_descs=None,
    debias_min_sim=0.2,
    prototype_lib=None,
    segment_map=None,
    prototype_weight=0.2,
    save_predictions=None,
):
    """Run evaluation with optional debiasing, prototype guidance, and prediction export."""
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    predictions = {} if save_predictions else None

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
            base_query = ann['sentences'][i]
            candidates = [base_query]
            if 'response' in ann and 'query_json' in ann['response'][i]:
                candidates += ann['response'][i]['query_json'][0]['descriptions']

            selected_query = base_query
            selected_meta = None
            if debias_queries and len(candidates) > 1:
                selected_query, selected_meta = select_debiased_query(
                    candidates,
                    visual_descs.get(vid) if visual_descs else None,
                    min_similarity=debias_min_sim,
                )
                query_variants = [selected_query]
            elif use_llm and len(candidates) > 1:
                query_variants = candidates
            else:
                query_variants = [base_query]

            proposals = []
            for query_text in query_variants:
                query_json = [{'descriptions': query_text}]
                proposals += localize(video_feature, duration, query_json, stride, hyperparams, tckmeans)

            prototype_explanation = None
            if prototype_lib and segment_map:
                proto_query = selected_query if debias_queries else base_query
                proposals, prototype_explanation = apply_prototype_guidance(
                    proposals,
                    proto_query,
                    prototype_lib,
                    segment_map,
                    vid,
                    weight=prototype_weight,
                )

            proposals = select_proposal(np.array(proposals))

            s, e = ann['timestamps'][i]
            s, e = s + pad_sec, e + pad_sec

            sp, ep = proposals[0][0],  proposals[0][1]

            iou_ = (min(e, ep) - max(s, sp)) / (max(e, ep) - min(s, sp))
            ious.append(max(iou_, 0))
            recall += thresh <= iou_

            if predictions is not None:
                predictions.setdefault(vid, []).append({
                    'query': base_query,
                    'selected_query': selected_query,
                    'selection_meta': selected_meta,
                    'prediction': {
                        'start': float(sp),
                        'end': float(ep),
                        'score': float(proposals[0][2]),
                    },
                    'prototype': prototype_explanation,
                })
        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))

    if predictions is not None:
        with open(save_predictions, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)



if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)


    visual_descs = None
    if args.visual_desc and os.path.exists(args.visual_desc):
        with open(args.visual_desc, 'r', encoding='utf-8') as f:
            visual_descs = json.load(f)

    prototype_lib = None
    segment_map = None
    if args.prototype_lib and args.segment_proto_map:
        prototype_lib = load_prototype_library(args.prototype_lib)
        segment_map = load_segment_prototypes(args.segment_proto_map)

    eval_kwargs = dict(
        debias_queries=args.debias_queries,
        visual_descs=visual_descs,
        debias_min_sim=args.debias_min_sim,
        prototype_lib=prototype_lib,
        segment_map=segment_map,
        prototype_weight=args.prototype_weight,
        save_predictions=args.save_predictions,
    )

    if args.llm_output and os.path.exists(args.llm_output):
        with open(args.llm_output) as f:
            data = json.load(f)
        eval(
            data,
            dataset['feature_path'],
            dataset['stride'],
            dataset['hyper_parameters'],
            args.use_llm,
            args.tckmeans,
            pad_sec=0.0,
            **eval_kwargs,
        )
    else:
        with open(dataset['splits'][args.split]['annotation_file']) as f:
            data = json.load(f)
        eval(
            data,
            dataset['feature_path'],
            dataset['stride'],
            dataset['hyper_parameters'],
            args.use_llm,
            args.tckmeans,
            dataset['splits'][args.split]['pad_sec'],
            **eval_kwargs,
        )
