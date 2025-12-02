import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import MyDataset
from model import HSTUModel
from tools import *
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_  
from transformers import get_cosine_schedule_with_warmup
from eval_metrics import evaluate_recall_metrics

def train_model(args, log_file, writer, data_path):
    # 数据集加载
    dataset = MyDataset(data_path, args)
    # 划分训练集和验证集
    #train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
  
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    # 模型加载
    if args.use_hstu:
        print("Using HSTU Model")
        model = HSTUModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    else:
        raise NotImplementedError("Only HSTU model is implemented currently.")
    with torch.no_grad():
        model.pos_emb.weight.data[0, :] = 0
        model.pos_emb_right.weight.data[0, :] = 0
        model.item_emb.weight.data[0, :] = 0
        model.user_emb.weight.data[0, :] = 0
        model.action_emb.weight.data[0, :] = 0  
        for k in model.sparse_emb:
            model.sparse_emb[k].weight.data[0, :] = 0

    # 如果给了预训练模型路径或者继续训练，则加载模型
    model, epoch_start_idx, global_step = load_pretrained_model(model, args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98),weight_decay=args.weight_decay)

    # 总训练步数
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5  # 半个cosine周期，也可以尝试1.0(完整周期)
    )
    if global_step > 0:
        for _ in range(global_step):
            scheduler.step() # 恢复学习率
    print("Start training")
    model.train()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):

        print(f"Epoch {epoch} training:")

        for step, batch in enumerate(train_loader):
            if args.load_time_intervals:
                seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, cur_time, next_time = batch
            else:
                seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)

            token_type = token_type.to(args.device)
            action_type = action_type.to(args.device)
            next_token_type = next_token_type.to(args.device)
            input_interval = cur_time.to(args.device) if args.load_time_intervals else None

            with autocast(dtype=torch.bfloat16, device_type='cuda', enabled=True):
                loss, diagnostics = model.forward(seq, token_type, action_type, seq_feat, input_interval)
            
            optimizer.zero_grad(set_to_none=True)
            if args.l2_emb > 0:
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
            loss.backward()
           
            if global_step >= 3000:
                grad_norm_before_clip = clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
            else:
                grad_norm_before_clip = clip_grad_norm_(model.parameters(), max_norm=6.0)
                
            optimizer.step()
            scheduler.step()  

            global_step += 1
        
            grad_norm = 0.0
            for param in model.parameters():
                  if param.grad is not None:
                       grad_norm += torch.norm(param.grad).item() ** 2
            grad_norm = np.sqrt(grad_norm)
            
            output_dict = {
                    'loss': loss.item(),
                    'grad_norm_before_clip': grad_norm_before_clip.item(),
                    'grad_norm': grad_norm.item(),
                    'learning_rate': scheduler.get_last_lr()[0],
                    'AUC': diagnostics['auc'].item(),
                }
            output_info(global_step, epoch, log_file, writer, output_dict, is_train=True)
        '''
        model.eval()
        print(f"Epoch {epoch} evaluating:")
        with torch.no_grad():
            for step, batch in enumerate(valid_loader):
                if args.load_time_intervals:
                    seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, cur_time, next_time = batch
                else:
                    seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                seq = seq.to(args.device)

                token_type = token_type.to(args.device)
                action_type = action_type.to(args.device)
                next_token_type = next_token_type.to(args.device)
                input_interval = cur_time.to(args.device) if args.load_time_intervals else None

                with autocast(dtype=torch.bfloat16, device_type='cuda', enabled=True):
                    loss, diagnostics = model.forward(seq, token_type, action_type, seq_feat, input_interval)
                
                output_dict = {
                        'loss': loss.item(),
                        'AUC': diagnostics['auc'].item(),
                    }
                output_info(global_step, epoch, log_file, writer, output_dict, is_train=False)
        '''

        # 保存模型检查点
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}_epoch{epoch}_Validloss{loss.item():.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()

if __name__ == '__main__':
    # 获取参数和环境设置
    args = get_args()
    log_file, writer, data_path = environment_check()
    # 开始训练
    try:
        train_model(args, log_file, writer, data_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred during training: {e}")
        log_file.write(f"An error occurred during training: {e}\n")
        # 打印占用的显存
        if torch.cuda.is_available(): 
            print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        # 清空缓存和释放内存
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        # 导入模型进行继续训练
        # 首先找到最新的模型，模型保存在 TRAIN_CKPT_PATH 目录下，并且都以global_step 开头
        ckpt_path = Path(os.environ.get('TRAIN_CKPT_PATH'))
        model_files = list(ckpt_path.glob("global_step*/model.pt"))
        if model_files:
            latest_model_file = max(model_files, key=os.path.getctime)
            print(f"Found latest model file: {latest_model_file}. Attempting to resume training.")
            log_file.write(f"Found latest model file: {latest_model_file}. Attempting to resume training.\n")
            # 继续训练
            try:
                args.state_dict_path = str(latest_model_file)
                train_model(args, log_file, writer, data_path)
            except Exception as e:
                traceback.print_exc()
                print(f"An error occurred during resumed training: {e}")
        else:
            print("No model files found to resume training.")
