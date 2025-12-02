import gc
import json
import os
import struct
from pathlib import Path
import sys
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


def save_item_emb(item_ids, feat_dict_all, feature_types,model,batch_size=1024, knns_list=None):
    """Save item embeddings (same as baseline)"""
    all_embs = []
    knn_start_idx = 0
    knn_end_idx = 0
    all_feat_keys = feature_types['item_sparse'] + feature_types['item_array'] + feature_types['item_dynamic_sparse'] + feature_types['item_emb'] + feature_types.get('item_dynamic_continual', [])
    for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
        end_idx = min(start_idx + batch_size, len(item_ids))

        item_seq = torch.tensor(item_ids[start_idx:end_idx], device=model.dev).unsqueeze(0)
        # 找出其中为-1的冷启动item
        knn_end_idx += (item_seq == -1).sum().item()
        batch_feat = feat_dict_all[start_idx:end_idx]
        batch_knns = knns_list[knn_start_idx:knn_end_idx]
        cold_start_knn_tensor = torch.from_numpy(np.array(batch_knns))
        knn_start_idx = knn_end_idx
        batch_feat = np.array(batch_feat, dtype=object)
        batch_feat = [batch_feat]

        # Process each feature type
        seq_feat_tensors = {}

        for k in all_feat_keys:
            # Convert seq_feat
            seq_feat_tensors[k] = feat2tensor(list(batch_feat), k, feature_types)
        batch_emb = model.feat2emb(item_seq, seq_feat_tensors, include_user=False,cold_start_list_tensor=cold_start_knn_tensor).squeeze(0)
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


def build_candidates(testdataset, model):
    """
    直接构造候选库，并让模型返回：
      - candidate_vectors: np.ndarray [N, D]
      - candidate_ids: np.ndarray[uint64] (即 creative_id)
    全程不再使用 retrieve_id，也不落盘。
    """
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, features = [], [], []
    indexer = testdataset.indexer['i']
    feat_types = testdataset.feature_types
    cold_start_knn_dict = testdataset.cold_start_dict
    knn_num = testdataset.knn_num
    knns_list = []
    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            feature = line['features']
            creative_id = line['creative_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0
            feature = process_cold_start_feat(feature)
            feature = testdataset.fill_missing_feat(feature, item_id,is_user=False)
            # item_id = testdataset.item_expose.get(str(item_id), 0)  # 使用重映射表进行ID转换
            # 查找knn邻居
            if str(creative_id) in cold_start_knn_dict:
                # knns_list.append(cold_start_knn_dict[str(creative_id)][:knn_num])
                knns = cold_start_knn_dict[str(creative_id)][:knn_num]
                # 重映射id
                remapped_knns = [testdataset.item_rid_remap.get(str(knn), 0) for knn in knns]
                knns_list.append(remapped_knns)
                item_id = -1  # cold start 
            else:
                item_id = testdataset.item_rid_remap.get(str(item_id), 0)  # 使用重映射表进行ID转换
            item_ids.append(item_id)
            creative_ids.append(creative_id)
            features.append(feature)

    # 调用模型：直接返回候选库向量与 creative_id（无需落盘）
    candidate_vectors = save_item_emb(
        item_ids=item_ids,
        feat_dict_all=features,
        feature_types=feat_types,
        model=model,
        batch_size=1024,
        knns_list=knns_list
    )
    # 统一 dtype

    return candidate_vectors, creative_ids


def ann_topk_batched(candidate_emb,
                     candidate_ids,
                     user_emb,
                     top_k: int,
                     metric_type: int = 1,
                     device: str | torch.device | None = None,
                     batch_size: int | None = None
                     ) -> np.ndarray:
    """
    内存内 ANN（batched TopK，支持 CUDA/CPU 自动回退）
    metric_type: 1=内积；0=L2
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    dv = candidate_emb
    N = dv.shape[0]

    Q = user_emb.shape[0]
    if batch_size is None:
        batch_size = min(4096, Q) if Q > 0 else 0

    out_ids = np.empty((Q, min(top_k, N)), dtype=np.uint64)
    ptr = 0

    for s in tqdm(range(0, Q, batch_size), desc="ANN search", total=(Q + batch_size - 1) // batch_size):
        e = min(s + batch_size, Q)
        qb = user_emb[s:e]

        if metric_type == 1:
            scores = qb @ dv.t()               # [B, N]
            _, idx = torch.topk(scores, k=min(top_k, N), dim=1)
        else:
            dist = torch.cdist(qb, dv)         # [B, N]
            _, idx = torch.topk(-dist, k=min(top_k, N), dim=1)

        out_ids[ptr:ptr + (e - s)] = np.asarray(candidate_ids, dtype=np.uint64)[idx.detach().cpu().numpy()]
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
        model_dirs.sort(key=lambda x: int(x.split('global_step')[1].split('_')[0]))
        latest_model_dir = model_dirs[-1]
        os.environ['MODEL_OUTPUT_PATH'] = os.path.join('./checkpoints', latest_model_dir)
        print(f"Using model from: {os.environ['MODEL_OUTPUT_PATH']}")
        args.local_eval = True
    else:
        args.local_eval = False

    # Dataset 准备
    data_path = os.environ.get('EVAL_DATA_PATH')
    data_path = Path(data_path)

    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn,
        pin_memory=True, persistent_workers=True, prefetch_factor=1
    )

    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    remap_item_num = test_dataset.item_rid_remap_dict['new_itemnum']
    if args.use_hstu:
        model = HSTUModel(usernum, itemnum,remap_item_num, feat_statistics, feat_types, args).to(args.device)
        model = torch.compile(model)
    model.eval()

    ckpt_path = get_ckpt_path()
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))


    # 2) 构造候选库（creative_id 为主键，直接返回向量与 id）
    with torch.inference_mode():
            candidate_vectors, candidate_ids = build_candidates(
                test_dataset,model
            )

    # 1) 生成用户查询向量（全程内存）
    user_embs = []
    user_list = []
    all_user_histories = [] # 记录所有用户的历史序列,一个list，元素是set
    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            time_batch_s = time.time()
            for step, batch in enumerate(test_loader):
                if args.load_time_intervals:
                    seq, token_type, action_type, seq_feat_user, seq_feat_item, user_id, cur_timestamp, target_actions, cold_start_knn_seq = batch
                    def to_dev(d):
                        return {k: v.to(args.device, non_blocking=True) for k, v in d.items()}
                    seq_feat_user = to_dev(seq_feat_user)
                    seq_feat_item = to_dev(seq_feat_item)
                    seq = seq.to(args.device, non_blocking=True)
                    token_type = token_type.to(args.device, non_blocking=True)
                    action_type = action_type.to(args.device, non_blocking=True)
                    target_actions = target_actions.to(args.device, non_blocking=True)
                    input_interval = cur_timestamp.to(args.device, non_blocking=True) if args.load_time_intervals else None
                    logits = model.predict(seq, seq_feat_user, seq_feat_item,token_type, action_type, input_interval,target_actions, cold_start_knn_seq)
                    # 记录历史序列
                    for s in seq:
                        history_items = set(s.tolist())
                        history_items.discard(0)  # 去除padding
                        all_user_histories.append(history_items)
                else:
                    seq, token_type, action_type, seq_feat, user_id = batch
                    seq = seq.to(args.device)
                    logits = model.predict(seq, seq_feat_user, seq_feat_item, token_type, action_type)
                user_embs.append(logits)
                user_list += user_id
                # 每个step打印进度
                if step % 100 == 0:
                    time_batch_e = time.time()
                    print(f"Step {step}, Time taken for batch: {time_batch_e - time_batch_s:.2f} seconds")
                    time_batch_s = time_batch_e

    user_embs = torch.cat(user_embs, dim=0)

    # 清理内存，释放无用变量
    del model
    del test_loader
    indexer = test_dataset.indexer['i']
    del test_dataset
    batch = None
    logits = None
    seq = None
    seq_feat_user = None
    seq_feat_item = None
    token_type = None
    action_type = None
    cur_timestamp = None
    target_actions = None
    user_id = None
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)  # 等待系统回收内存

    # 3) 内存内 ANN 检索（结果直接是 creative_id，不再需要任何映射）
    top_k = 200
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result_ids = ann_topk_batched(
            candidate_emb=candidate_vectors,
            candidate_ids=candidate_ids,
            user_emb=user_embs,
            top_k=top_k,
            metric_type=1,     # 与原配置一致：内积
            device=None,       # 自动选择 cuda/cpu
            batch_size=1024    # 可按显存调小
        )

    # 4) 过滤掉用户历史序列中已出现的 item，然后取前10
    top10s = []
    num = 0
    time_s = time.time()
    for idx in range(len(user_list)):
        filtered = []
        num += 1
        for candidate_id in result_ids[idx]:
            # candidate_id 转化为 re_id
            re_id = indexer.get(candidate_id, None)
            if re_id not in all_user_histories[idx]:
                filtered.append(int(candidate_id))
            if len(filtered) == 10:
                break
        top10s.append(filtered)
        if num % 50000 == 0:
            time_e = time.time()
            print(f"Filtering user {num}/{len(user_list)}, Time taken for filtering: {time_e - time_s:.2f} seconds")

    return top10s, user_list

if __name__ == "__main__":
    top10s, user_list = infer()
    print(f"共为{len(user_list)}个用户召回了top10候选集")