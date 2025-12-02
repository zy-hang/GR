import json
import os
import struct
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset
from model import HSTUModel
from tools import *
from dataset import feat2tensor

def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    print('*' * 50)
    print('模型所在路径')
    print(ckpt_path)
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    raise FileNotFoundError(f"No .pt file found in {ckpt_path}")


def save_item_emb(item_ids, feat_dict_all, feature_types,model,batch_size=1024):
    """Save item embeddings (same as baseline)"""
    all_embs = []

    for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
        end_idx = min(start_idx + batch_size, len(item_ids))

        item_seq = torch.tensor(item_ids[start_idx:end_idx], device=model.dev).unsqueeze(0)
        batch_feat = []
        for i in range(start_idx, end_idx):
            batch_feat.append(feat_dict_all[i])

        batch_feat = np.array(batch_feat, dtype=object)
        batch_feat = [batch_feat]

        all_feat_keys = set()
        for feat_list in [batch_feat]:
            for feat_seq in feat_list:
                for feat_dict in feat_seq:
                    all_feat_keys.update(feat_dict.keys())

        # Process each feature type
        seq_feat_tensors = {}

        for k in all_feat_keys:
            # Convert seq_feat
            seq_feat_tensors[k] = feat2tensor(list(batch_feat), k, feature_types)
        batch_emb = model.feat2emb(item_seq, seq_feat_tensors, include_user=False).squeeze(0)
        batch_emb = batch_emb / batch_emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        all_embs.append(batch_emb)
    # Save embeddings
    final_embs = torch.cat(all_embs, dim=0)

    return final_embs


def process_cold_start_feat(feat: dict):
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if isinstance(feat_value, list):
            processed_feat[feat_id] = [0 if isinstance(v, str) else v for v in feat_value]
        elif isinstance(feat_value, str):
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


def build_candidates(indexer, feat_types, feat_default_value, mm_emb_dict, model):
    """
    直接构造候选库，并让模型返回：
      - candidate_vectors: np.ndarray [N, D]
      - candidate_ids: np.ndarray[uint64] (即 creative_id)
    全程不再使用 retrieve_id，也不落盘。
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, features = [], [], []

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            feature = line['features']
            creative_id = line['creative_id']
            # retrieval_id = line.get('retrieval_id')  # 已弃用，不再使用

            item_id = indexer[creative_id] if creative_id in indexer else 0
            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature)
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]
            for feat_id in feat_types['item_emb']:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            features.append(feature)

    # 调用模型：直接返回候选库向量与 creative_id（无需落盘）
    candidate_vectors = save_item_emb(
        item_ids=item_ids,
        feat_dict_all=features,
        feature_types=feat_types,
        model=model,
        batch_size=2048,
    )
    # 统一 dtype

    return candidate_vectors, creative_ids


def ann_topk_batched(dataset_vectors,
                     dataset_ids,
                     query_vectors,
                     top_k: int,
                     metric_type: int = 1,
                     device: str | torch.device | None = None,
                     batch_size: int | None = None) -> np.ndarray:
    """
    内存内 ANN（batched TopK，支持 CUDA/CPU 自动回退）
    metric_type: 1=内积；0=L2
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    dv = dataset_vectors
    N = dv.shape[0]

    Q = query_vectors.shape[0]
    if batch_size is None:
        batch_size = min(4096, Q) if Q > 0 else 0

    out_ids = np.empty((Q, min(top_k, N)), dtype=np.uint64)
    ptr = 0

    for s in tqdm(range(0, Q, batch_size), desc="ANN search", total=(Q + batch_size - 1) // batch_size):
        e = min(s + batch_size, Q)
        qb = query_vectors[s:e]

        if metric_type == 1:
            scores = qb @ dv.t()               # [B, N]
            _, idx = torch.topk(scores, k=min(top_k, N), dim=1)
        else:
            dist = torch.cdist(qb, dv)         # [B, N]
            _, idx = torch.topk(-dist, k=min(top_k, N), dim=1)

        out_ids[ptr:ptr + (e - s)] = np.asarray(dataset_ids, dtype=np.uint64)[idx.detach().cpu().numpy()]
        ptr += (e - s)

        # 释放临时张量
        del qb, idx
        if metric_type == 1:
            del scores
        else:
            del dist
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return out_ids


def infer():
    args = get_args()
    # 环境准备（与 infer.py 一致，但不再构造外部 ANN 命令）
    if 'EVAL_DATA_PATH' not in os.environ:
        cwd = os.getcwd()
        print(cwd)
        os.chdir(cwd + '/JoyRec')
        print(os.getcwd())

        os.environ['EVAL_DATA_PATH'] = './eval_data'
        os.environ['EVAL_RESULT_PATH'] = './ANNresult'  # 不再使用，但保留变量避免其他地方依赖
        if not os.path.exists(os.environ['EVAL_RESULT_PATH']):
            os.makedirs(os.environ['EVAL_RESULT_PATH'])
        model_dirs = [d for d in os.listdir('./checkpoints') if os.path.isdir(os.path.join('./checkpoints', d))]
        if not model_dirs:
            raise FileNotFoundError("No model directories found in ./checkpoints")
        model_dirs.sort(key=lambda x: int(x.split('global_step')[1].split('.')[0]))
        latest_model_dir = model_dirs[-1]
        os.environ['MODEL_OUTPUT_PATH'] = os.path.join('./checkpoints', latest_model_dir)
        print(f"Using model from: {os.environ['MODEL_OUTPUT_PATH']}")
        args.local_eval = True
    else:
        args.local_eval = False

    # Dataset 准备
    data_path = os.environ.get('EVAL_DATA_PATH')
    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=8, collate_fn=test_dataset.collate_fn,
        pin_memory=True, persistent_workers=True, prefetch_factor=1
    )

    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types

    if args.use_hstu:
        model = HSTUModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()

    ckpt_path = get_ckpt_path()
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))

    # 1) 生成用户查询向量（全程内存）
    user_embs = []
    user_list = []

    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            time_batch_s = time.time()
            for step, batch in enumerate(test_loader):
                if args.load_time_intervals:
                    seq, token_type, action_type, seq_feat, user_id, cur_timestamp = batch
                    seq = seq.to(args.device)
                    logits = model.predict(seq, seq_feat, token_type, action_type, cur_timestamp)
                else:
                    seq, token_type, action_type, seq_feat, user_id = batch
                    seq = seq.to(args.device)
                    logits = model.predict(seq, seq_feat, token_type, action_type)
                user_embs.append(logits)
                user_list += user_id
                # 每个step打印进度
                if step % 100 == 0:
                    time_batch_e = time.time()
                    print(f"Step {step}, Time taken for batch: {time_batch_e - time_batch_s:.2f} seconds")
                    time_batch_s = time_batch_e

    user_embs = torch.cat(user_embs, dim=0)

    # 2) 构造候选库（creative_id 为主键，直接返回向量与 id）
    candidate_vectors, candidate_ids = build_candidates(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
    )

    # 3) 内存内 ANN 检索（结果直接是 creative_id，不再需要任何映射）
    top_k = 1000 if args.local_eval else 100
    result_ids = ann_topk_batched(
        dataset_vectors=candidate_vectors,
        dataset_ids=candidate_ids,
        query_vectors=user_embs,
        top_k=top_k,
        metric_type=1,     # 与原配置一致：内积
        device=None,       # 自动选择 cuda/cpu
        batch_size=1024    # 可按显存调小
    )

    # 4) 取前 10 返回（已是 creative_id）
    top10s = result_ids[:, :10].tolist()
    return top10s, user_list

if __name__ == "__main__":
    top10s, user_list = infer()
    print(f"共为{len(user_list)}个用户召回了top10候选集")