import abc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from dataset import save_emb
from torchmetrics import AUROC


class FM(nn.Module):
    def __init__(self, input_dim=256, latent_dim=64):  # 32做消融
        """
        FM模型实现
        Args:
            input_dim (int): 输入特征维度 [B, D]
            latent_dim (int): 隐向量维度
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # 线性部分（一阶项）
        self.linear = nn.Linear(input_dim, 1, bias=True)

        # 隐向量矩阵（二阶交叉项参数）
        self.embedding = nn.Parameter(torch.Tensor(input_dim, latent_dim))

        # 参数初始化
        self._initialize_parameters()

    def _initialize_parameters(self):
        """参数初始化"""
        # 线性层使用Xavier初始化
        nn.init.xavier_normal_(self.linear.weight)
        # 隐向量矩阵使用截断正态分布初始化
        nn.init.trunc_normal_(self.embedding, std=0.02)
        # 偏置项初始化为0
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        B, K, D = x.shape
        x = x.reshape(B * K, D)

        linear_term = self.linear(x)
        interaction_term = torch.mm(x, self.embedding)
        square_of_sum = torch.pow(interaction_term, 2).sum(dim=1, keepdim=True)

        square = torch.pow(self.embedding, 2)
        x_square = torch.pow(x, 2)
        sum_of_square = torch.mm(x_square, square).sum(dim=1, keepdim=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)  # [batch_size, 1]
        output = linear_term + cross_term

        output = output.reshape(B, K, 1)
        return output


class HSTUMultiHeadAttention(nn.Module):

    def __init__(
            self,
            hidden_units,
            num_heads,
            max_len,
            dropout_rate,
            attention_types=['dot_product'],
            liner_dim=3,
            attention_dim=5,
            use_rope: bool = False,
            rope_theta: float = 10000.0,
            rope_partial_dim: int = 0,
    ):
        super().__init__()

        self.hidden_units = hidden_units
        self._num_heads = num_heads
        self.max_len = max_len
        self.dropout_rate = dropout_rate

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_units, eps=1e-6)
        self._linear_dim = liner_dim
        self._attention_dim = attention_dim

        self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(
                (
                    hidden_units,
                    liner_dim * 2 * num_heads
                    + attention_dim * num_heads * 2,
                )
            ).normal_(mean=0, std=0.02),
        )
        self._o = torch.nn.Linear(
            in_features=liner_dim * num_heads,
            out_features=hidden_units,
        )
        torch.nn.init.xavier_uniform_(self._o.weight)

        # Dropout
        self.dropout_uv = nn.Dropout(dropout_rate)
        self.dropout_att = nn.Dropout(dropout_rate)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._linear_dim], eps=1e-6)

    def forward(self, input, attn_mask):
        """
        input: [B, S, hidden_units]
        input_interval: [B, S] 或 None
        attn_mask: [B, S, S] 的 bool，True 表示可见
        """
        batch_size, seq_len, hidden_unit = input.shape

        norm_input = self.layer_norm(input)
        batched_mm_output = torch.mm(norm_input.view(batch_size * seq_len, hidden_unit), self._uvqk)
        batched_mm_output = F.silu(batched_mm_output)
        U, V, Q, K = torch.split(
            batched_mm_output,
            [
                self._linear_dim * self._num_heads,
                self._linear_dim * self._num_heads,
                self._attention_dim * self._num_heads,
                self._attention_dim * self._num_heads,
            ],
            dim=1,
        )
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self._num_heads, self._attention_dim).transpose(1, 2)  # [B,H,S,Dq]
        K = K.view(batch_size, seq_len, self._num_heads, self._attention_dim).transpose(1, 2)  # [B,H,S,Dk]
        V = V.view(batch_size, seq_len, self._num_heads, self._linear_dim).transpose(1, 2)  # [B,H,S,Dv]
        U = U.view(batch_size, seq_len, self._num_heads, self._linear_dim).transpose(1, 2)  # [B,H,S,Du]

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [B,H,S,S]

        # silu activation 与缩放
        attention_scores = F.silu(attention_scores) / seq_len

        # 因果掩码
        attention_scores = attention_scores.masked_fill(~attn_mask.unsqueeze(1), float(0.))

        output = torch.matmul(attention_scores, V)  # [B,H,S,Dv]

        # 加无学习参数的layernorm, 稳定output的输出
        output = self._norm_attn_output(output)

        u_dot = U * output
        u_dot = u_dot.transpose(1, 2).contiguous().view(batch_size, seq_len, self._linear_dim * self._num_heads)
        new_outputs = input + self._o(u_dot)

        return new_outputs, None


class HSTUModel(nn.Module):

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.maxlen = args.maxlen

        # HSTU specific arguments with defaults
        self.attention_types = getattr(args, 'attention_types', ['dot_product'])
        self.attention_activation = getattr(args, 'attention_activation', 'none')

        # Embeddings
        self.item_emb = nn.Embedding(self.item_num + 1, 128, padding_idx=0)
        self.user_emb = nn.Embedding(self.user_num + 1, 128, padding_idx=0)
        self.pos_emb = nn.Embedding(2 * (args.maxlen + 1) + 1, args.hidden_units, padding_idx=0)
        self.pos_emb_right = nn.Embedding(2 * (args.maxlen + 1) + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.action_emb = torch.nn.Embedding(2 + 1, args.hidden_units, padding_idx=0)

        # Feature embeddings (same as baseline)
        self.sparse_emb = nn.ModuleDict()
        self.emb_transform = nn.ModuleDict()
        self._init_feat_info(feat_statistics, feat_types)

        # Feature processing layers
        userdim = 128 + (12 + 4 + 6 + 4 + 7 + 8 + 5 + 4) * 2 + len(self.USER_CONTINUAL_FEAT)  # 228
        itemdim = 128 + 173 + len(self.ITEM_CONTINUAL_FEAT) + 32 * len(self.ITEM_EMB_FEAT)  # 333

        self.auroc = AUROC(task='binary').to(args.device)

        self.userdnn = nn.Linear(userdim, args.hidden_units)
        self.itemdnn = nn.Linear(itemdim, args.hidden_units)
        self.itemdnn_forward = nn.Linear(itemdim, args.hidden_units)

        self.itemdnn2 = nn.Linear(args.hidden_units, args.hidden_units)

        # HSTU Transformer blocks
        self.attention_layers = nn.ModuleList()
        for _ in range(args.num_blocks):
            attention_layer = HSTUMultiHeadAttention(
                hidden_units=args.hidden_units,
                num_heads=args.num_heads,
                max_len=args.maxlen,
                dropout_rate=args.dropout_rate,
                attention_types=self.attention_types,
                liner_dim=args.linear_dim,
                attention_dim=args.attention_dim,
                use_rope=getattr(args, 'use_rope', True),  # 默认开启 RoPE（若 tools 未加参数也能跑）
                rope_theta=getattr(args, 'rope_theta', 10000.0),
                rope_partial_dim=getattr(args, 'rope_partial_dim', 0),
            )
            self.attention_layers.append(attention_layer)

        # Final layer norm
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.click_fm = FM(input_dim=args.hidden_units, latent_dim=args.latent_dim)

        # Initialize feature embeddings
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, self.each_embedding[k], padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, self.each_embedding[k], padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, self.each_embedding[k], padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, self.each_embedding[k], padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = nn.Linear(self.ITEM_EMB_FEAT[k], 32)

        # 新增特征加权模块
        self.user_SE = nn.Sequential(
            nn.Linear(9, 3),
            nn.ReLU(),
            nn.Linear(3, 9),
            nn.Sigmoid()
        )
        self.item_SE = nn.Sequential(
            nn.Linear(13 + len(self.ITEM_EMB_FEAT), 4),
            nn.ReLU(),
            nn.Linear(4, 13 + len(self.ITEM_EMB_FEAT)),
            nn.Sigmoid()
        )

        # Initialize embeddings with better initialization
        self._init_embeddings()

    def _init_feat_info(self, feat_statistics, feat_types):
        """Initialize feature information (same as baseline)"""
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}
        self.each_embedding = {
            '103': 12 * 2, '104': 4 * 2, '105': 6 * 2, '109': 4 * 2, '106': 7 * 2, '107': 8 * 2, '108': 5 * 2,
            '110': 4 * 2,
            # 以下是item特征
            '100': 5, '117': 16, '111': 64, '118': 18, '101': 10, '102': 22, '119': 20, '120': 19,
            '114': 8, '112': 9, '121': 42, '115': 16, '122': 22, '116': 8
        }

    def _init_embeddings(self):
        """Better embedding initialization following HSTU practices"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def apply_se_weighting(self, id_emb, other_feats, se_net):
        feats = [id_emb] + other_feats  # 将 ID 特征纳入 SE
        squeeze_tensor = torch.stack([feat.mean(dim=2) for feat in feats], dim=2)  # [B, S, num_features]
        B, S, num_features = squeeze_tensor.shape
        squeeze_flat = squeeze_tensor.view(B * S, num_features)
        weights_flat = se_net(squeeze_flat)
        weights = weights_flat.view(B, S, num_features)  # [B, S, num_features]

        reweighted_feats = []
        for i, feat in enumerate(feats):
            weight = weights[:, :, i].unsqueeze(2)  # [B, S, 1]
            reweighted_feats.append(feat * weight * 2)  # 对包括 ID 在内的所有特征加权

        return torch.cat(reweighted_feats, dim=2)

    def feat2emb(self, seq, feature_tensors, mask=None, include_user=False):
        """Convert features to embeddings (updated to work with pre-converted tensors)"""
        seq = seq.to(self.dev)
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]

        # Process all feature types
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]

        if include_user:
            all_feat_types.extend([
                (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
            ])

        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                if k in feature_tensors:
                    tensor_feature = feature_tensors[k].to(self.dev)

                    if feat_type.endswith('sparse'):
                        tensor_feature = tensor_feature.long()
                        feat_list.append(self.sparse_emb[k](tensor_feature))
                    elif feat_type.endswith('array'):
                        tensor_feature = tensor_feature.long()
                        emb = self.sparse_emb[k](tensor_feature)  # (B,S,L,D)
                        mask_arr = (tensor_feature != 0).unsqueeze(-1)  # (B,S,L,1)
                        emb_sum = (emb * mask_arr).sum(dim=2)  # (B,S,D)
                        counts = mask_arr.sum(dim=2).clamp_min(1)  # (B,S,1)
                        counts = counts.to(emb_sum.dtype)
                        feat_list.append(emb_sum / counts)
                    elif feat_type.endswith('continual'):
                        feat_list.append(tensor_feature.float().unsqueeze(2))

        for k in self.ITEM_EMB_FEAT:
            if k in feature_tensors:
                tensor_feature = feature_tensors[k].to(self.dev)  # (B,S,dim)
                item_feat_list.append(self.emb_transform[k](tensor_feature))

        # Combine features
        origin_item_emb = self.apply_se_weighting(item_feat_list[0], item_feat_list[1:], self.item_SE)
        all_item_emb = F.silu(self.itemdnn(origin_item_emb))

        origin_user_emb = None
        if include_user:
            origin_user_emb = self.apply_se_weighting(user_feat_list[0], user_feat_list[1:], self.user_SE)
            all_user_emb = F.silu(self.userdnn(origin_user_emb))
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
        return seqs_emb, origin_item_emb, origin_user_emb

    def forward(self, user_item, mask, action_type, seq_feature, input_interval=None):
        mask = mask.to(self.dev)
        item_action_mask = (mask == 1).to(self.dev)  # mask==0是padding,mask==2是user
        user_mask = (mask == 2).to(self.dev)
        user_item = user_item.to(self.dev)
        action_type = action_type.to(self.dev)

        seq_emb, origin_item_emb, origin_user_emb = self.feat2emb(user_item, seq_feature, mask=mask, include_user=True)

        B, L, D = seq_emb.shape
        device = seq_emb.device
        dtype = seq_emb.dtype

        action_ids = (action_type.to(self.dev) + 1) * item_action_mask
        action_emb = self.action_emb(action_ids)

        action_emb = F.silu(self.itemdnn2(action_emb)) * item_action_mask.unsqueeze(-1)

        new_seqs = torch.zeros(B, 2 * L, D, device=device, dtype=dtype)
        new_seqs[:, 0::2, :] = seq_emb
        new_seqs[:, 1::2, :] = action_emb

        new_seqs *= self.pos_emb.embedding_dim ** 0.5

        # Add positional embeddings
        new_log_seqs = user_item.repeat_interleave(2, dim=1)  # [B, 2L]
        positions = torch.arange(1, 2 * L + 1, device=device).unsqueeze(0).expand(B, -1)
        padding_mask_for_pos = (new_log_seqs != 0).long()
        positions = positions * padding_mask_for_pos
        new_seqs += self.pos_emb(positions)  # user 的padding的action embedding也会加上位置embedding

        ltr_pos = (new_log_seqs != 0).long().cumsum(1)  # [B, 2L], pad 仍为 0，非pad 为 1..L
        new_seqs += self.pos_emb_right(ltr_pos)

        # 加dropout
        new_seqs = self.emb_dropout(new_seqs)

        # new_seqs: [B, 2L, D]
        maxlen = new_seqs.shape[1]

        # 加mask
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask_pad = attention_mask_pad.repeat_interleave(2, dim=1)  # [B, 2L]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)  # 它会取一个矩阵的下三角部分（包括对角线），将上三角部分全部置为 False（禁止关注）。
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1) & attention_mask_pad.unsqueeze(2)

        for i in range(len(self.attention_layers)):
            new_seqs, _ = self.attention_layers[i](new_seqs, attention_mask)

        user_forward = self.last_layernorm(new_seqs[:, ::2, :])  # Shape: [B, L, D]，Item位置是输出下一个action
        item_forward = self.itemdnn_forward(origin_item_emb)

        pred_input = user_forward * item_forward

        # 点击 logits（FM 输出为 [B, L, 1]）
        pred_click = self.click_fm(pred_input).squeeze(-1)  # [B, L]

        # pred_click: [B, L]，越大越可能点击
        item_pos_mask = (mask == 1)  # [B, L]
        logits_all = pred_click  # [B, L]
        labels_all = (action_type == 1)  # [B, L]  True=点击, False=曝光

        pos_mask = item_pos_mask & labels_all  # 点击位 [B, L]
        neg_mask = item_pos_mask & (~labels_all)  # 曝光位 [B, L]

        # 设置温度
        t = float(getattr(self, "t", 0.07))
        logits_all = logits_all / t

        pos_counts = pos_mask.sum(dim=1)  # [B]
        neg_counts = neg_mask.sum(dim=1)  # [B]
        valid_user = (pos_counts > 0) & (neg_counts > 0)  # 同时有点击和曝光的用户

        # 取该 batch 的最大点击数/曝光数，用 topk + -inf padding 来“打包”到等长
        NEG_INF = -1e9
        B, L = logits_all.shape

        # 取出每个用户的所有点击 logits（不足 P_max 用 -inf 填充）
        pos_logits_masked = logits_all.masked_fill(~pos_mask, NEG_INF)  # [B, L]
        pos_topk_vals, _ = torch.topk(pos_logits_masked, k=L, dim=1)  # [B, L]

        # 取出每个用户的所有曝光 logits（不足 N_max 用 -inf 填充）
        neg_logits_masked = logits_all.masked_fill(~neg_mask, NEG_INF)  # [B, L]
        neg_topk_vals, _ = torch.topk(neg_logits_masked, k=L, dim=1)  # [B, L]

        # 合并曝光与点击
        pos_col = pos_topk_vals.unsqueeze(2)  # [B, L, 1]
        neg_block = neg_topk_vals.unsqueeze(1).expand(-1, L, -1)  # [B, L, L]
        logits_mat = torch.cat([pos_col, neg_block], dim=2)  # [B, L, 1+L]

        # 有效行/列 mask（逐用户的“有效点击行”和“有效曝光列”）
        device = logits_mat.device
        row_idx = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        col_idx = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        row_valid = (row_idx < pos_counts.unsqueeze(1))  # [B, L]
        col_valid = (col_idx < neg_counts.unsqueeze(1))  # [B, L]

        # 类别维度（第 0 列为该点击自身）的有效性：行有效且用户有效；负列还需 col 有效
        class_valid_pos = row_valid.unsqueeze(2)  # [B, L, 1]
        class_valid_neg = col_valid.unsqueeze(1).expand(-1, L, -1)  # [B, L, L]
        class_valid = torch.cat([class_valid_pos, class_valid_neg], dim=2)  # [B, L, 1+L]
        class_valid = class_valid & valid_user.view(-1, 1, 1)  # 过滤无效用户

        # 对无效类别置极小值；注意 padding 用 NEG_INF（避免 NaN，logsumexp 也稳定）
        masked_logits = logits_mat.masked_fill(~class_valid, NEG_INF)  # [B, L, 1+L]

        # 计算每行 CE：-logit_pos + logsumexp(all_classes)
        pos_logit = masked_logits[:, :, 0]  # [B, L]
        logsum = torch.logsumexp(masked_logits, dim=2)  # [B, L]
        per_row_loss = -pos_logit + logsum  # [B, L]

        # 仅统计“有效点击行 & 有效用户”的行
        row_valid_all = row_valid & valid_user.view(-1, 1)   # [B, L]
        per_row_loss = per_row_loss * row_valid_all.float()  # [B, L] 每个用户每个点击对应的loss

        # 先对每个用户的点击行做均值，再跨用户均值
        loss_sum_per_user = per_row_loss.sum(dim=1)  # [B]， 点击loss求和
        num_rows_per_user = row_valid_all.sum(dim=1).clamp_min(1)  # [B]
        mean_loss_per_user = loss_sum_per_user / num_rows_per_user  # [B]

        rank_loss = (mean_loss_per_user * valid_user.float()).sum() / valid_user.float().sum().clamp_min(1.0)

        # ------- AUC：仅在“有效用户”的 item 位上评估 -------
        with torch.no_grad():
            eval_mask = item_pos_mask & valid_user.view(-1, 1)  # [B, L]
            v_logits = logits_all[eval_mask]  # [N]
            v_labels = labels_all.float()[eval_mask]  # [N]
            # 这里一定含双类（因为有效用户 >=1），但仍加保险
            if v_labels.numel() > 0 and torch.unique(v_labels.round()).numel() == 2:
                p = torch.sigmoid(v_logits)
                self.auroc.update(p.detach(), v_labels.round().long())
                auc = self.auroc.compute()
                self.auroc.reset()
            else:
                auc = torch.tensor(0.0, device=self.dev)

        diagnostics = {
            'rank_loss': rank_loss,
            'auc': auc,
        }
        return rank_loss, diagnostics

    @torch.no_grad()
    def score_candidates(self, user_item, mask, action_type, seq_feature, input_interval=None, candidate_seq=None,
                         candiadate_feature=None):
        """
            为一批候选物品进行打分(精排)。
            该函数实现了 M-FALCON 算法的核心思想。
        """
        mask = mask.to(self.dev)
        item_action_mask = (mask == 1).to(self.dev)  # mask==0是padding,mask==2是user
        user_mask = (mask == 2).to(self.dev)
        user_item = user_item.to(self.dev)
        action_type = action_type.to(self.dev)

        # 1、得到用户序列和候选物品的embedding
        seq_emb, origin_item_emb, origin_user_emb = self.feat2emb(user_item, seq_feature, mask=mask, include_user=True)
        candidate_emb, cand_origin_item_emb, _ = self.feat2emb(candidate_seq, candiadate_feature, include_user=False)
        B, L, D = seq_emb.shape
        device = seq_emb.device
        dtype = seq_emb.dtype
        K = candidate_emb.shape[1]  # 候选物品数量

        # 2、得到action的embedding
        action_ids = (action_type.to(self.dev) + 1) * item_action_mask
        action_emb = self.action_emb(action_ids)

        # 3、得到user_embedding
        user_sum = (origin_user_emb * user_mask.unsqueeze(-1)).sum(dim=1, keepdim=True)  # [B,1,d]
        user_cnt = user_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)  # [B,1,1]
        origin_user_emb = (user_sum / user_cnt).expand(-1, L, -1)  # [B,L,d]

        # 4、保留原版的user和item的embedding当作wide侧特征
        wide_item = cand_origin_item_emb
        wide_user = (user_sum / user_cnt).expand(-1, K, -1)  # [B,K,d]

        action_emb = F.silu(self.itemdnn2(torch.cat(
            [action_emb, origin_item_emb, origin_user_emb], dim=-1))) * item_action_mask.unsqueeze(-1)

        history_seqs = torch.zeros(B, 2 * L, D, device=device, dtype=dtype)
        history_seqs[:, 0::2, :] = seq_emb  # 此处包含user embedding
        history_seqs[:, 1::2, :] = action_emb

        new_seqs = torch.cat([history_seqs, candidate_emb], dim=1)  # [B, 2L+K, D]
        new_seqs *= self.pos_emb.embedding_dim ** 0.5

        # 3、位置嵌入
        positions_hist = torch.arange(1, 2 * L + 1, device=device).unsqueeze(0).expand(B, -1)
        positions_cand = torch.full((B, K), fill_value=2 * L - 1, device=device,
                                    dtype=torch.long)  # 所有候选物复用最后一个item的position embedding
        positions = torch.cat([positions_hist, positions_cand], dim=1)

        hist_pos_mask = (user_item != 0).repeat_interleave(2, dim=1)  # [B, 2L]
        cand_pos_mask = torch.ones(B, K, dtype=torch.bool, device=device)  # shape为[B, K]的全ture
        pos_mask = torch.cat([hist_pos_mask, cand_pos_mask], dim=1)  # [B, 2L+K]
        positions = positions * pos_mask
        new_seqs += self.pos_emb(positions)

        hist_log_seqs = user_item.repeat_interleave(2, dim=1)  # [B, 2L]
        ltr_pos = (hist_log_seqs != 0).long().cumsum(1)  # [B, 2L]
        cand_ltr_pos = ltr_pos[:, -2].unsqueeze(1).expand(-1, K)  # [B, K], 所有候选物复用最后一个item的position embedding
        ltr_pos = torch.cat([ltr_pos, cand_ltr_pos], dim=1)  # [B, 2L+K]
        new_seqs += self.pos_emb_right(ltr_pos)

        # 4. 构建 M-FALCON注意⻅力掩码
        total_len = 2 * L + K

        # 因果 + M-FALCON 掩码
        m_falcon_mask = torch.ones((total_len, total_len), dtype=torch.bool, device=device)
        m_falcon_mask = torch.tril(m_falcon_mask)  # 历史部分是因果的
        # 候选物之间不能互相看到，只能看到自己和历史，将右下角 KxK 区域清零，然后对角线设为1
        m_falcon_mask[2 * L:, 2 * L:] = torch.eye(K, dtype=torch.bool, device=device)
        final_attention_mask = m_falcon_mask.unsqueeze(0) & pos_mask.unsqueeze(1) & pos_mask.unsqueeze(2)

        # 5. 处理时间间隔, 候选位置直接time+1
        if input_interval is None:
            input_interval = torch.zeros(B, L, dtype=torch.long, device=device)
        else:
            input_interval = input_interval.long().to(device)
        new_input_interval = torch.zeros(B, total_len, device=device, dtype=input_interval.dtype)
        new_input_interval[:, 0:2 * L:2] = input_interval
        new_input_interval[:, 1:2 * L:2] = input_interval
        last_timestamps = input_interval[:, -1].unsqueeze(1)
        new_input_interval[:, 2 * L:] = last_timestamps + 1

        # 6. 通过精排 Transformer 层
        for layer in self.attention_layers:
            new_seqs, _ = layer(new_seqs, new_input_interval, final_attention_mask)  # / seq_len有长度问题，先测试后解决

        # 7. 提取候选物位置的输出，得到预测点击的概率作为分数
        new_seqs = self.last_layernorm(new_seqs)  # [B, 2L+K, D]
        cand_outputs = new_seqs[:, 2 * L:, :]  # [B, K, D]
        pred_action = new_seqs[:, :2 * L:2, :]  # [B, L, D]
        user_sum = (pred_action * user_mask.unsqueeze(-1)).sum(dim=1, keepdim=True)  # [B,1,D]
        user_cnt = user_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)  # [B,1,1]
        user_mean = (user_sum / user_cnt).expand(-1, cand_outputs.size(1), -1)  # [B,K,D]
        pred_input = torch.cat([cand_outputs, user_mean, wide_item, wide_user], dim=-1)  # [B,K,2D]

        pred_click = self.click_fm(pred_input)  # [B, K, 1]
        pred_expose = self.expose_fm(pred_input)  # [B, K, 1]
        pred_click_expose = torch.cat([pred_expose, pred_click], dim=-1)  # [B, K, 2]

        # 计算点击概率，直接用作排序分数
        click_prob = F.softmax(pred_click_expose, dim=-1)[:, :, 1]  # [B, K]
        return click_prob
