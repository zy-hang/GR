# JoyRec/tools.py
import argparse
import json
import os
from pathlib import Path
import pickle
import random
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torch.amp import autocast

def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.0015, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--clip_norm', default=2.5, type=float)

    # 梯度检查点
    parser.add_argument('--grad_checkpoint',  default=True,
                        help='Enable activation/gradient checkpointing on attention blocks')
    parser.add_argument('--ckpt_every', type=int, default=1,
                        help='Checkpoint every N attention layers (1 = every layer)')
    
    # 梯度累积步速
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='梯度累积的累计步数')

    # 负样本缓存（跨 step 的 memory bank）
    parser.add_argument('--neg_cache_steps', type=int, default=1,
                        help='保留前 K 个 step 的负样本向量作为额外负样本（0 表示不启用）')
    parser.add_argument('--neg_cache_on_cpu', default=False,
                        help='将负样本缓存放到 CPU 上（节省显存，计算前再搬回 GPU）')
    # 添加负样本数
    parser.add_argument('--num_neg', default=4, type=int, help='每个正样本采样的负样本数')

    # margin设置
    parser.add_argument('--margin_click', type=float, default=0.02)
    parser.add_argument('--margin_conv',  type=float, default=0.02)
    

    # Model 
    parser.add_argument('--hidden_units', default=512, type=int)
    parser.add_argument('--num_blocks', default=48, type=int)
    parser.add_argument('--num_epochs', default=8, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0001, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')

    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--label_smothing', default=False)

    # rope
    parser.add_argument('--use_rope', default=True,
                        help='Enable Rotary Positional Embeddings for Q/K')
    parser.add_argument('--rope_theta', default=10000.0, type=float,
                        help='RoPE base theta (a.k.a. rotary freqs base)')
    parser.add_argument('--rope_partial_dim', default=32, type=int,
                        help='Apply RoPE on first N dims of Q/K (0 = use all)')

    # HSTU Model Options
    parser.add_argument('--use_hstu', action='store_true', default=True, help='Use HSTU model instead of baseline')
    parser.add_argument('--attention_types', action='append', default=['Dot_prd_Time_bias_Position_bias'], type=str,
                        choices=['Dot_prd', 'Dot_prd_Time_bias', 'Dot_prd_Position_bias',
                                 'Dot_prd_Time_bias_Position_bias', 'Time_bias', 'Position_bias'], )
    parser.add_argument('--load_time_intervals', action='store_true', default=True,
                        help='Load and use time interval data')
                        
    parser.add_argument('--linear_dim', default=64, type=int, help='Linear dimension for u and v')
    parser.add_argument('--attention_dim', default=64, type=int, help='Dimension for q and k in attention')
    
    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str,
                        choices=[str(s) for s in range(81, 87)])
    args = parser.parse_args()
    return args

def environment_check():
    if 'TRAIN_LOG_PATH' not in os.environ:
        cwd = os.getcwd()
        print(cwd)  # 输出：/home/user/project（示例）
        os.chdir(cwd+'/JoyRec')
        # 确保当前工作目录是脚本所在目录
        print(os.getcwd()) 
        os.environ['TRAIN_LOG_PATH'] = './logs'
        os.environ['TRAIN_TF_EVENTS_PATH'] = './tf_events'
        os.environ['TRAIN_DATA_PATH'] = './TencentGR_1k_new'
        os.environ['TRAIN_CKPT_PATH'] = './checkpoints'
        os.environ['EVAL_RESULT_PATH'] = './checkpoints'
        os.environ['MODEL_OUTPUT_PATH'] = './'

    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')
    return log_file, writer, data_path


def clear_cache(cache_dir):
        # 自动清理USER_CACHE_PATH下的任务目录，最多保留2个最新任务
    try:
        cache_root = Path(cache_dir)
        if cache_root.exists() and cache_root.is_dir():
            # 只考虑以task开头的目录
            task_dirs = [d for d in cache_root.iterdir() if d.is_dir() and d.name.startswith('task')]
            if len(task_dirs) > 2:
                # 按修改时间排序，最早的在前
                task_dirs.sort(key=lambda d: d.stat().st_mtime)
                for old_dir in task_dirs[:-2]:
                    # 递归删除目录及其内容
                    for file in old_dir.iterdir():
                        if file.is_file():
                            file.unlink()
                        elif file.is_dir():
                            for subfile in file.iterdir():
                                subfile.unlink()
                            file.rmdir()
                    old_dir.rmdir()
                print(f"Cleared old cache directories, kept the latest 2.")
    except Exception as e:
        print(f"Warning: failed to clean USER_CACHE_PATH: {e}")

def load_pretrained_model(model,args):
    epoch_start_idx = 1
    global_step = 0
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            # 找出global_step和epoch
            filename = args.state_dict_path
            if 'epoch' in filename:
                epoch_start_idx = int(filename.split('epoch')[1].split('_')[0]) + 1
            if 'global_step' in filename:
                global_step = int(filename.split('global_step')[1].split('_')[0]) 
                print(f'loaded model from {args.state_dict_path}, epoch_start_idx: {epoch_start_idx}, global_step: {global_step}')
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')
    return model, epoch_start_idx, global_step

@torch.autocast('cuda', enabled=True, dtype=torch.bfloat16)
def get_loss_streaming(args, pos_logits, neg_bank_tuple,
                       next_token_type, next_action_type,
                       action_only=False, temperature=0.022):
    
    mask_all   = (next_token_type == 1)
    mask_click = (next_token_type == 1) & (next_action_type == 1)
    mask_conv = (next_token_type == 1) & (next_action_type == 2)
    click_mask_valid = mask_click[mask_all]         
    conv_mask_valid = mask_conv[mask_all]  
   
    q_valid, neg_cache, model, neg_bank_cur = neg_bank_tuple
    scale = pos_logits.new_tensor(1.0 / float(temperature), dtype=torch.float32)  

    pos_all_valid = pos_logits[mask_all].to(torch.float32)       

    m_click = float(getattr(args, 'margin_click', 0.0))
    m_conv  = float(getattr(args, 'margin_conv',  0.0))
    if (m_click > 0.0) or (m_conv > 0.0):
        nav = next_action_type[mask_all]  # 与 pos_all_valid 对齐
        m_valid = torch.where(nav == 1, pos_all_valid.new_full((), m_click), torch.where(nav == 2, pos_all_valid.new_full((), m_conv), 0.0))
        pos_all_valid = pos_all_valid - m_valid

    s_pos = pos_all_valid * scale                                          

    with torch.no_grad():
        a = float(getattr(args, 'pos_weight_a', 0.8))   

        B, L = next_token_type.shape
        mask_all_f = mask_all.to(torch.float32)              
        rank = (mask_all_f.cumsum(dim=1) - mask_all_f)        
        count = mask_all_f.sum(dim=1, keepdim=True)           
        denom = torch.clamp(count - 1.0, min=1.0)             
        ratio = rank / denom                                

        weights_full = (1.0 - a) + (2.0 * a) * ratio         

        one_pos_mask = (count <= 1).expand(-1, L) & mask_all  # [B,L]
        weights_full = torch.where(one_pos_mask, weights_full.new_tensor(1.0), weights_full)

        weights_valid = weights_full[mask_all].to(torch.float32)  # [M]   

    # ====== 单遍流式 log-sum-exp======
    lse = s_pos.clone().to(torch.float32)                          
    chunk = 4096

    if neg_cache is not None and len(neg_cache) > 0:
        for neg_cached, neg_feat_cached, next_token_type_cached in neg_cache:
            if args.neg_cache_on_cpu:
                neg_cached = neg_cached.to(args.device, non_blocking=True)
                next_token_type_cached = next_token_type_cached.to(args.device, non_blocking=True)
                neg_feat_cached = {k: v.to(args.device, non_blocking=True) 
                                   for k, v in neg_feat_cached.items()}
           
            with autocast(dtype=torch.bfloat16, device_type='cuda', enabled=True):
                if neg_cached.dim() == 3:
                    B, L, N = neg_cached.shape
                    neg_ids_flat = neg_cached.permute(0, 2, 1).reshape(B * N, L)      # [B*N, L]
                    mask_flat_2d = (
                        (next_token_type_cached == 1)
                        .unsqueeze(1).expand(-1, N, -1)                                # [B, N, L]
                        .reshape(B * N, L)                                             # [B*N, L]
                    )
                else:
                    B, L = neg_cached.shape
                    neg_ids_flat = neg_cached                                         # [B, L]
                    mask_flat_2d = (next_token_type_cached == 1)                       # [B, L]

                neg_embs_hist = model.feat2emb(neg_ids_flat, neg_feat_cached, include_user=False)  # [B*N, L, D] or [B, L, D]
                neg_embs_hist = F.normalize(neg_embs_hist.float(), dim=-1, eps=1e-8).to(neg_embs_hist.dtype)

            # 统一：只留可预测位并拍平到 2D
            D = neg_embs_hist.size(-1)
            neg_embs_hist = neg_embs_hist[mask_flat_2d].view(-1, D)                    # [*, D]                 
            bank = neg_embs_hist.to(q_valid.dtype)
            for st in range(0, bank.size(0), chunk):
                bk = bank[st:st + chunk, :]
                s = (q_valid @ bk.T).to(torch.float32) * scale
                lse = torch.logaddexp(lse, torch.logsumexp(s, dim=1))
                del s, bk
            del bank
            
            del neg_cached, neg_feat_cached, next_token_type_cached, mask_flat_2d, neg_embs_hist
            torch.cuda.empty_cache()

    # 处理当前步的负样本
    del neg_cache
    bank = neg_bank_cur.to(q_valid.dtype)
    for st in range(0, bank.size(0), chunk):
        bk = bank[st:st + chunk, :]
        s = (q_valid @ bk.T).to(torch.float32) * scale
        lse = torch.logaddexp(lse, torch.logsumexp(s, dim=1))
        del s, bk
    del bank, q_valid
    torch.cuda.empty_cache()

    loss_vec = lse - s_pos
    # 加权均值
    loss_all = (loss_vec * weights_valid).sum() / weights_valid.sum()  # loss_all = loss_vec.mean()
    loss_click = loss_vec[click_mask_valid].mean() 
    loss_conv = loss_vec[conv_mask_valid].mean() 

    loss = loss_click if action_only else (0.1*loss_click + 0.1*loss_conv + loss_all)
   
    return loss, None, None, None, None, None, None

def set_all_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_job_ids_from_path(p: Path):
    """
    从 TRAIN_CKPT_PATH 反解任务 ID 与实例 ID或者从 state_dict_path 反解，这两种路径的格式不同
    TRAIN_CKPT_PATH路径模式：
    /.../ams_2025_1029731852466342281/angel_training_ams_2025_1029731852466342281_20250916171223_e853fd5c/8b0fbdd9994cc92e01995262d23f00bb/ckpt
    state_dict_path路径模式：'/apdcephfs_fsgm/share_303710656/angel/ams_2025_1029731852466342281/angel_training_ams_2025_1029731852466342281_20250817151921_7783fa70/self/8b0fb81b98a6c39e0198bc5ea3d21506/ckpt/global_step56360.valid_loss=3.1972/model.pt'
    可以依据末尾是否是包含model.pt来区分
    返回：(任务ID, 实例ID)
    """
    parts = p.parts
    if parts[-1] == 'model.pt':
        # state_dict_path模式
        task_id = parts[-6]
        inst_id = parts[-4]
        return task_id, inst_id
    task_id, inst_id = None, None
    for i, part in enumerate(parts):
        if part.startswith("angel_training_ams_"):
            task_id = part
            inst_id = parts[i + 1]
            break
    return task_id or "001", inst_id or "001"

def fill_neg_cache(args, train_loader, neg_cache):
    if neg_cache is not None and len(neg_cache) < args.neg_cache_steps:
        # 最简单的做法，取出dataloader的neg_cache_steps个batch，放入缓存
        # 由于每次iterate dataloader都会打乱顺序，所以每次epoch开始时缓存的样本也不一样
        print(f"Filling negative sample cache with {args.neg_cache_steps} batches...")
        for step, batch in enumerate(train_loader):
            if args.load_time_intervals:
                # seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, cur_time, next_time = batch
                seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat_user,seq_feat_item, pos_feat, neg_feat, cur_time, next_time = batch
            else:
                # seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat_user,seq_feat_item, pos_feat, neg_feat = batch
            def to_dev(d):
                return {k: v.to(args.device, non_blocking=True) for k, v in d.items()}
            neg = neg.to(args.device, non_blocking=True)
            next_token_type = next_token_type.to(args.device, non_blocking=True)
            neg_feat = to_dev(neg_feat)
            if args.neg_cache_on_cpu:
                neg_store = neg.detach().cpu()
                next_token_type_store = next_token_type.detach().cpu()
                neg_feat_store = {k: v.detach().cpu() for k, v in neg_feat.items()}
            else:
                neg_store = neg.detach()
                next_token_type_store = next_token_type.detach()
                neg_feat_store = {k: v.detach() for k, v in neg_feat.items()}
            neg_cache.append((neg_store, neg_feat_store, next_token_type_store))
            if len(neg_cache) >= args.neg_cache_steps:
                break
        print("Negative sample cache filled.")



def save_epoch_checkpoint(ckpt_dir: Path, optimizer, scheduler
                         , rng_states):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
    meta = dict(rng_states=rng_states)
    with open(ckpt_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

def load_epoch_checkpoint(ckpt_dir: Path, optimizer, scheduler, device):
    optimizer.load_state_dict(torch.load(ckpt_dir / f"optimizer.pt", map_location=device))
    scheduler.load_state_dict(torch.load(ckpt_dir / f"scheduler.pt", map_location=device))
    with open(ckpt_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    print("Loaded optimizer and scheduler states from checkpoint.")
    return meta

def output_info(global_step, epoch, log_file, writer,outpot_dice,is_train):
    """
    output_dict: {key: value} where key is the name of the metric and value is the metric value
    """
    if 'loss' in outpot_dice:
        loss = outpot_dice['loss']
    log_json = json.dumps(
        {'global_step': global_step, 'loss'+ 'train' if is_train else 'valid'
        : loss, 'epoch': epoch, 'time': time.time()}
    )
    log_file.write(log_json + '\n')
    log_file.flush()
    # print(log_json)
    for key, value in outpot_dice.items():
        writer.add_scalar(f'{key}/{"train" if is_train else "valid"}', value, global_step)

from typing import Dict, List, Tuple, Union

# -------------------------------------------------
# 1. 2024-2025 法定假日（key: "YYYY-MM-DD"）
#    value=True  => 法定放假
#    value=False => 调休补班
LEGAL_2024_2025: Dict[str, bool] = {
    # ===== 2024 =====
    "2024-01-01": True,     # 元旦
    "2024-02-10": True,     # 春节
    "2024-02-11": True,
    "2024-02-12": True,
    "2024-02-13": True,
    "2024-02-14": True,
    "2024-02-15": True,
    "2024-02-16": True,
    "2024-02-17": True,
    "2024-04-04": True,     # 清明
    "2024-04-05": True,
    "2024-04-06": True,
    "2024-05-01": True,     # 劳动节
    "2024-05-02": True,
    "2024-05-03": True,
    "2024-05-04": True,
    "2024-05-05": True,
    "2024-06-8": True,     # 端午
    "2024-06-9": True,     # 端午
    "2024-06-10": True,     # 端午
    "2024-09-15": True,     # 中秋
    "2024-09-16": True,
    "2024-09-17": True,
    "2024-10-01": True,     # 国庆
    "2024-10-02": True,
    "2024-10-03": True,
    "2024-10-04": True,
    "2024-10-05": True,
    "2024-10-06": True,
    "2024-10-07": True,


    # ===== 2025 =====
    "2025-01-01": True,     # 元旦
    "2025-01-28": True,     # 春节
    "2025-01-29": True,
    "2025-01-30": True,
    "2025-01-31": True,
    "2025-02-01": True,
    "2025-02-02": True,
    "2025-02-03": True,
    "2025-02-04": True,
    "2025-04-04": True,     # 清明
    "2025-04-05": True,
    "2025-04-06": True,
    "2025-05-01": True,     # 劳动节
    "2025-05-02": True,
    "2025-05-03": True,
    "2025-05-04": True,
    "2025-05-05": True,
    "2025-05-31": True,     # 端午
    "2025-06-01": True,
    "2025-06-02": True,
    "2025-10-01": True,     # 国庆&中秋连放
    "2025-10-02": True,
    "2025-10-03": True,
    "2025-10-04": True,
    "2025-10-05": True,
    "2025-10-06": True,
    "2025-10-07": True,
    "2025-10-08": True,
}

# -------------------------------------------------
# 2. 电商人造节日（list 内为月-日）
E_COMMERCE: Dict[Tuple[int, int], str] = {
    '2024-06-18': True,     # 618
    '2024-08-18': True,     # 818
    '2024-11-11': True,     # 双十一
    '2024-12-12': True,     # 双十二
    '2025-06-18': True,     # 618
    '2025-08-18': True,     # 818
}


if __name__ == "__main__":
    # 测试函数parse_job_ids_from_path,输入不同格式的路径
    p1 = Path("/apdcephfs_fsgm/share_303710656/angel/ams_2025_1029731852466342281/angel_training_ams_2025_1029731852466342281_20250817151921_7783fa70/self/8b0fb81b98a6c39e0198bc5ea3d21506/ckpt/global_step56360.valid_loss=3.1972/model.pt")
    p2 = Path("/ams_2025_1029731852466342281/angel_training_ams_2025_1029731852466342281_20250916171223_e853fd5c/8b0fbdd9994cc92e01995262d23f00bb/ckpt")
    task_id1, inst_id1 = parse_job_ids_from_path(p1)
    task_id2, inst_id2 = parse_job_ids_from_path(p2)
    print(f"Path 1: {p1}\n  -> Task ID: {task_id1}, Instance ID: {inst_id1}")
    print(f"Path 2: {p2}\n  -> Task ID: {task_id2}, Instance ID: {inst_id2}")
