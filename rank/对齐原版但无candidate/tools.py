# JoyRec/tools.py
import argparse
import json
import os
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser()
    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--weight_decay', default=0.005, type=float)
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--clip_norm', default=0.8, type=float)
    

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=512, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_epochs', default=8, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--latent_dim', default=64, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--l2_emb', default=0.00001, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--label_smothing', default=False)

    # rope
    parser.add_argument('--use_rope', default=True,
                        help='Enable Rotary Positional Embeddings for Q/K')
    parser.add_argument('--rope_theta', default=10000.0, type=float,
                        help='RoPE base theta (a.k.a. rotary freqs base)')
    parser.add_argument('--rope_partial_dim', default=0, type=int,
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
        os.environ['TRAIN_DATA_PATH'] = '../JoyRec/TencentGR_1k'
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


