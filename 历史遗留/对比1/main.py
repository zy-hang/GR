import os
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import MyDataset
from model import HSTUModel
from tools import *
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_  
from transformers import get_cosine_schedule_with_warmup
import bitsandbytes 

from collections import deque

def train_model(args, log_file, writer, data_path):
    import math
    set_all_seed(20240912)  # 设置随机种子，确保结果可复现
    # 数据集加载
    dataset = MyDataset(data_path, args)
    g = torch.Generator()
    g.manual_seed(20240912)  # 每次生成的随机数都一样，确保每次划分的batch都一样
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=5, collate_fn=dataset.collate_fn,
        pin_memory=True,
        pin_memory_device="cuda",
        persistent_workers=False,
        prefetch_factor=1,
        generator=g
    )
    
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    # 模型加载
    model = HSTUModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model = torch.compile(model)
    model, epoch_start_idx, global_step = load_pretrained_model(model, args)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 给 LN/bias 关闭 weight decay
    decay_others, nodecay_others = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias"):
            nodecay_others.append(p)
        else:
            decay_others.append(p)

    optimizer = bitsandbytes.optim.AdamW8bit(
        [
            {"params": decay_others,   "weight_decay": args.weight_decay},  
            {"params": nodecay_others, "weight_decay": 0.0},                
        ],
        lr=args.lr,
        betas=(0.9, 0.98),
    )
   
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Embedding) and m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()


    # === 梯度累积相关：总优化步数/预热步数 ===
    grad_accum_steps = max(1, getattr(args, "grad_accum_steps", 1))
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5
    )
    neg_cache = deque(maxlen=args.neg_cache_steps) if args.neg_cache_steps > 0 else None
    cache_dir = os.environ.get('USER_CACHE_PATH', '.')

    print("Start training")


    model.train()

    # === 计时/日志 ===
    T_since_log = 0.0  # 自上次日志以来的累计秒数（包含所有 micro steps）
    T_total_since_log_min = 0.0
    out_put_step = 100   # 每多少个“优化器步”打印一次
    optimizer.zero_grad(set_to_none=True)  # 累积前先清空梯度
    # 每个 epoch 开始清空一次（也可以挪到循环外，不清空）
    fill_neg_cache(args, train_loader, neg_cache)
    old_global_step = deque(maxlen=2)
    old_global_step.append(global_step)
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        t0_epoch = time.time()
        print(f"Epoch {epoch} training:")
        if args.inference_only:
            break
        t0 = time.time()
        micro_in_this_epoch = 0  

        for step, batch in enumerate(train_loader):
            if args.load_time_intervals:
                seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat_user,seq_feat_item, pos_feat, neg_feat, cur_time, next_time = batch
            else:
                seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat_user,seq_feat_item, pos_feat, neg_feat = batch
            def to_dev(d):
                return {k: v.to(args.device, non_blocking=True) for k, v in d.items()}
            seq_feat_user = to_dev(seq_feat_user)
            seq_feat_item = to_dev(seq_feat_item)
            pos_feat = to_dev(pos_feat)
            neg_feat = to_dev(neg_feat)
            seq = seq.to(args.device, non_blocking=True)
            pos = pos.to(args.device, non_blocking=True)
            neg = neg.to(args.device, non_blocking=True)
            token_type = token_type.to(args.device, non_blocking=True)
            action_type = action_type.to(args.device, non_blocking=True)
            next_token_type = next_token_type.to(args.device, non_blocking=True)
            next_action_type = next_action_type.to(args.device, non_blocking=True)
            input_interval = cur_time.to(args.device, non_blocking=True) if args.load_time_intervals else None

            with autocast(dtype=torch.bfloat16, device_type='cuda', enabled=True):
                pos_logits, _, neg_bank_tuple, pos_ids_valid, neg_ids_cur = model(
                    seq, pos, neg, token_type, next_token_type, action_type,
                    seq_feat_user,seq_feat_item, pos_feat, neg_feat, input_interval, next_action_type
                )
            q_valid, neg_bank_cur = neg_bank_tuple  # [M, D], [M, D]      
            
            action_only = (epoch == args.num_epochs) # 微调
            loss, loss_click, lose_expose, pos_mean_click, neg_mean_click, pos_mean_all, neg_mean_all = get_loss_streaming(
                args, pos_logits, (q_valid, neg_cache, model, neg_bank_cur, neg_ids_cur), next_token_type, next_action_type, action_only=False, pos_ids_valid=pos_ids_valid,
            )
            if args.l2_emb > 0:
                l2_item = (model.item_emb.weight.float().pow(2).sum() + model.item_emb_.weight.float().pow(2).sum()).sqrt()
                loss = loss + args.l2_emb * l2_item

            # ---------- 梯度累积：缩放 loss 并反传 ----------
            (loss / grad_accum_steps).backward()
            micro_in_this_epoch += 1

            if neg_cache is not None:
                with torch.no_grad():
                    if args.neg_cache_on_cpu:
                        neg_store = neg.detach().cpu()
                        next_token_type_store = next_token_type.detach().cpu()
                        neg_feat_store = {k: v.detach().cpu() for k, v in neg_feat.items()}
                    else:
                        neg_store = neg.detach()
                        next_token_type_store = next_token_type.detach()
                        neg_feat_store = {k: v.detach() for k, v in neg_feat.items()}
                    neg_cache.append((neg_store, neg_feat_store, next_token_type_store))

            # ---------- 到达累积步或最后一个 batch：执行一次优化器更新 ----------
            do_step = (micro_in_this_epoch % grad_accum_steps == 0) or (step == len(train_loader) - 1)
            if do_step:
                # 梯度裁剪
                if global_step >= 3000:
                    grad_norm_before_clip = clip_grad_norm_(model.parameters(), max_norm=args.clip_norm, foreach=False)
                else:
                    grad_norm_before_clip = clip_grad_norm_(model.parameters(), max_norm=20.0, foreach=False)

                optimizer.step()
                scheduler.step()  # 学习率调度（以优化器步为单位）
                optimizer.zero_grad(set_to_none=True)

                global_step += 1  # 注意：现在是“优化器步”意义上的 global_step

            # ---------- 计时/日志 ----------
            t8 = time.time()
            T_since_log += (t8 - t0)
            T_total_since_log_min += (t8 - t0) / 60.0
            # 仅在做了 optimizer.step() 时按 out_put_step 打印
            if do_step and (global_step % out_put_step == 0):
                avg_step_sec = T_since_log / out_put_step
                print(f'到达step{global_step}（每步=优化器更新，梯度累积={grad_accum_steps}），'
                      f'累计时间{T_total_since_log_min:.2f}分，平均每步时间{avg_step_sec:.2f}秒')
                T_since_log = 0.0
                T_total_since_log_min = 0.0
            t0 = time.time()
        t1_epoch = time.time()
        print(f"Epoch {epoch} finished in {(t1_epoch - t0_epoch)/60:.2f} min, loss={loss.item():.8f}")
        # 保存模型检查点
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}_epoch{epoch}_loss{loss.item():.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
if __name__ == '__main__':
    # 获取参数和环境设置
    args = get_args()
    log_file, writer, data_path = environment_check()
    train_model(args, log_file, writer, data_path)
