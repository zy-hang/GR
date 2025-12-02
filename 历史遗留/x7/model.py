import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast      


class RelativePositionalBias_hstu(nn.Module):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self._max_seq_len: int = max_seq_len
        self._w = torch.nn.Parameter(
            torch.empty(2 * (max_seq_len + 1) - 1).normal_(mean=0, std=0.02),
        )

    def forward(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        # 与输入无关的可学习相对位置偏置（维度匹配 HSTU 设计）
        del all_timestamps
        n: int = self._max_seq_len + 1
        t = F.pad(self._w[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        t = t[..., r:-r].unsqueeze(0)  # [1,1,n,n]
        return t


class RelativeBucketedTime_hstu(nn.Module):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """

    def __init__(self, max_seq_len: int, num_buckets: int = 128) -> None:
        super().__init__()
        self._max_seq_len: int = max_seq_len
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._num_buckets: int = num_buckets
        self._bucketization_fn = lambda x: (torch.log(torch.abs(x).clamp(min=1)) / 0.301).long()

    def forward(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, 1, N, N).
        """
        B = all_timestamps.size(0)
        N = self._max_seq_len + 1

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat([all_timestamps, all_timestamps[:, N - 1: N]], dim=1)
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self._num_buckets,
        ).detach()
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N).unsqueeze(1)  # [B,1,N,N]
        return rel_ts_bias


class HSTUMultiHeadAttention(nn.Module):

    def __init__(
        self,
        hidden_units,
        num_heads,
        max_len,
        dropout_rate,
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
        self.layer_norm = nn.LayerNorm(hidden_units, eps=1e-5)
        self.pinrec_layernorm = nn.LayerNorm(hidden_units, eps=1e-5)
        self._linear_dim = liner_dim
        self._attention_dim = attention_dim

        # PinRec的实现
        self.action_emb_PinRec = torch.nn.Embedding(3 + 1, 32, padding_idx=0)
        self.PinRec_FiLM = nn.Sequential(
                            nn.LayerNorm(32, eps=1e-5),
                            nn.Linear(32, 2 * hidden_units),
                           
        )
        self.r_scale = nn.Parameter(torch.tensor(0.1))
        self.b_scale = nn.Parameter(torch.tensor(0.1))

        # RoPE 配置
        self.use_rope = use_rope
        if self.use_rope:
            assert self._attention_dim % 2 == 0, "attention_dim 必须为偶数以支持 RoPE"
            self._rope_dim = (
                self._attention_dim if rope_partial_dim <= 0
                else min(rope_partial_dim, self._attention_dim)
            )
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self._rope_dim, 2).float() / self._rope_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        # qkvu和o配置
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

        # Bias modules
        self.relative_position_bias = RelativePositionalBias_hstu(max_len)
        self.time_interval_bias = RelativeBucketedTime_hstu(max_len)

        # Dropout
        self.dropout_uv = nn.Dropout(dropout_rate)
        self.dropout_att = nn.Dropout(dropout_rate)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._linear_dim], eps=1e-5)

    # ---------- RoPE 工具函数 ----------
    def _rope_apply(self, x, cos, sin, rope_dim: int):
        """
        x: [B, H, S, D]
        cos/sin: [1, 1, S, rope_dim//2]
        仅对前 rope_dim 维应用 RoPE，其余维度保持不变
        """
        x_rope, x_pass = torch.split(x, [rope_dim, x.size(-1) - rope_dim], dim=-1)  # [B,H,S,rope_dim], [B,H,S,D-rope_dim]
        x_even = x_rope[..., ::2]   # [B,H,S,rope_dim//2]
        x_odd  = x_rope[..., 1::2]  # [B,H,S,rope_dim//2]
        x_rope_out_even = x_even * cos - x_odd * sin
        x_rope_out_odd  = x_odd * cos + x_even * sin
        x_rope_out = torch.stack((x_rope_out_even, x_rope_out_odd), dim=-1).flatten(start_dim=-2)
        return torch.cat((x_rope_out, x_pass), dim=-1)

    def _maybe_apply_rope(self, Q, K):
        """
        Q/K: [B, H, S, D]
        返回：应用（或不应用）RoPE 后的 Q/K
        """
        if not self.use_rope:
            return Q, K
        B, H, S, D = Q.shape
        pos = torch.arange(S, device=Q.device, dtype=self.inv_freq.dtype)  # [S]
        freqs = torch.einsum('s,d->sd', pos, self.inv_freq)                # [S, rope_dim//2]
        cos = freqs.cos()[None, None, :, :]                                # [1,1,S,rope_dim//2]
        sin = freqs.sin()[None, None, :, :]                                # [1,1,S,rope_dim//2]
        cos = cos.to(Q.dtype)
        sin = sin.to(Q.dtype)
        rope_dim = self._rope_dim
        Q = self._rope_apply(Q, cos, sin, rope_dim)
        K = self._rope_apply(K, cos, sin, rope_dim)
        return Q, K
    
    def PinRec_func(self, log_feats, action_embs):
        r, b = self.PinRec_FiLM(action_embs).chunk(2, dim=-1)
        log_feats = log_feats + self.pinrec_layernorm(log_feats) * torch.tanh(r) * self.r_scale + b * self.b_scale
        return log_feats

    def forward(self, input, input_interval, attn_mask, next_action_type, next_mask):
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
        V = V.view(batch_size, seq_len, self._num_heads, self._linear_dim).transpose(1, 2)     # [B,H,S,Dv]
        U = U.view(batch_size, seq_len, self._num_heads, self._linear_dim).transpose(1, 2)     # [B,H,S,Du]

        # 应用 RoPE 到 Q/K, 并计算注意力分数
        Q, K = self._maybe_apply_rope(Q, K)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [B,H,S,S]

        # 加上time bias
        attention_scores += self.time_interval_bias(input_interval)  # [B,1,S,S] 广播到 [B,H,S,S]
        attention_scores += self.relative_position_bias(seq_len)     # [1,1,S,S] 广播

        # silu activation 与缩放
        attention_scores = F.silu(attention_scores) / seq_len

        # 因果掩码
        attention_scores = attention_scores.masked_fill(~attn_mask.unsqueeze(1), float(0.))

        output = torch.matmul(attention_scores, V)  # [B,H,S,Dv]

        # 加无学习参数的layernorm, 稳定output的输出
        output = self._norm_attn_output(output)

        u_dot = U * output
        u_dot = u_dot.transpose(1, 2).contiguous().view(batch_size, seq_len, self._linear_dim * self._num_heads)
        outputs = input + self._o(self.dropout_uv(u_dot))

        action_ids = (next_action_type + 1) * (next_mask == 1)
        action_embs = self.action_emb_PinRec(action_ids)
        outputs = self.PinRec_func(outputs, action_embs)

        return outputs, None


class HSTUModel(nn.Module):

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super().__init__()

        self.use_checkpoint = getattr(args, 'grad_checkpoint', False)
        self.ckpt_every = max(1, getattr(args, 'ckpt_every', 1))

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.maxlen = args.maxlen

        # item Embedding
        self.item_emb = nn.Embedding(self.item_num + 1, 64, padding_idx=0)
        self.item_emb_ = nn.Embedding(self.item_num + 1, 64, padding_idx=0)
        
        # position embedding
        self.pos_emb = nn.Embedding(2 * args.maxlen + 1, int(args.hidden_units//4), padding_idx=0)
        self.pos_emb_right = nn.Embedding(2 * args.maxlen + 1, int(args.hidden_units//4), padding_idx=0)
        self.action_emb = torch.nn.Embedding(3 + 1, int(args.hidden_units//4), padding_idx=0)
        self.action_position_embedding = nn.Linear(int(args.hidden_units//4)*3, args.hidden_units)

        # action embedding (sparse featrue)
        self.action_emb_append = torch.nn.Embedding(3 + 1, 16, padding_idx=0)

        # Feature embeddings (same as baseline)
        self.sparse_emb = nn.ModuleDict()
        self.emb_transform = nn.ModuleDict()
        self._init_feat_info(feat_statistics, feat_types)

        # 动态计算各分支输入维度
        self.item_input_dim_plain = self._calc_item_input_dim(include_time=False)
        self.item_input_dim_hist = self._calc_item_input_dim(include_time=True)
        self.user_input_dim = self._calc_user_input_dim()

        # 两套 DNN + 一套 user DNN
        self.itemdnn_plain = nn.Linear(self.item_input_dim_plain, args.hidden_units)
        self.itemdnn_hist = nn.Linear(self.item_input_dim_hist, args.hidden_units)
        self.userdnn = nn.Linear(self.user_input_dim, args.hidden_units)

        # HSTU Transformer blocks
        self.attention_layers = nn.ModuleList()
        for _ in range(args.num_blocks):
            # HSTU Multi-head attention
            attention_layer = HSTUMultiHeadAttention(
                hidden_units=args.hidden_units,
                num_heads=args.num_heads,
                max_len=args.maxlen,
                dropout_rate=args.dropout_rate,
                liner_dim=args.linear_dim,
                attention_dim=args.attention_dim,
                use_rope=getattr(args, 'use_rope', True),             # 默认开启 RoPE（若 tools 未加参数也能跑）
                rope_theta=getattr(args, 'rope_theta', 10000.0),
                rope_partial_dim=getattr(args, 'rope_partial_dim', 0),
            )
            self.attention_layers.append(attention_layer)

        # Final layer norm
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-5)

        # dropout
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

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

        # 新增：时间离散特征（仅历史分支会使用）
        for k in self.ITEM_TIME_SPARSE_FEAT:
            self.sparse_emb[k] = nn.Embedding(self.ITEM_TIME_SPARSE_FEAT[k] + 1, self.each_embedding[k], padding_idx=0)

        # 新增：物品动态特征
        for k in self.ITEM_DYNAMIC_FEAT:
            self.sparse_emb[k] = nn.Embedding(self.ITEM_DYNAMIC_FEAT[k] + 1, self.each_embedding[k], padding_idx=0)

        # SE 加权模块（按“特征组数”自适配
        # item：两套（历史含时间特征、plain 不含）
        n_item_feats_plain = 1 + len(self.ITEM_SPARSE_FEAT) + len(self.ITEM_ARRAY_FEAT)+ len(self.ITEM_CONTINUAL_STATIC_FEAT) + len(self.ITEM_EMB_FEAT)+len(self.ITEM_DYNAMIC_FEAT)+ len(self.ITEM_DYNAMIC_CONTINUAL_FEAT)
        n_item_feats_hist = n_item_feats_plain + len(self.ITEM_TIME_SPARSE_FEAT)+ len(self.ITEM_CONTINUAL_FEAT) + 1  # +1 是 action emb
        hid_plain = max(4, n_item_feats_plain // 2) if n_item_feats_plain > 0 else 4
        hid_hist = max(4, n_item_feats_hist // 2) if n_item_feats_hist > 0 else 4
        self.item_SE_plain = nn.Sequential(nn.Linear(max(1, n_item_feats_plain), hid_plain),
                                           nn.ReLU(), nn.Linear(hid_plain, max(1, n_item_feats_plain)), nn.Sigmoid())
        self.item_SE_hist = nn.Sequential(nn.Linear(max(1, n_item_feats_hist), hid_hist),
                                          nn.ReLU(), nn.Linear(hid_hist, max(1, n_item_feats_hist)), nn.Sigmoid())

        # user：一套
        n_user_feats = 0 + len(self.USER_SPARSE_FEAT) + len(self.USER_ARRAY_FEAT) + len(self.USER_CONTINUAL_FEAT)
        hid_user = max(4, n_user_feats // 2) if n_user_feats > 0 else 4
        self.user_SE = nn.Sequential(nn.Linear(max(1, n_user_feats), hid_user),
                                     nn.ReLU(), nn.Linear(hid_user, max(1, n_user_feats)), nn.Sigmoid())

        # 参数初始化
        self._init_embeddings()

    # ---------------- util: 统计各侧输入维度 ----------------
    def _sum_embedding_dims(self, keys):
        return sum(self.each_embedding[k] for k in keys)

    def _calc_item_input_dim(self, include_time: bool) -> int:
        dim = 2 * self.item_emb.embedding_dim # item id emb
        dim += self._sum_embedding_dims(self.ITEM_SPARSE_FEAT.keys())
        dim += self._sum_embedding_dims(self.ITEM_ARRAY_FEAT.keys())
        dim += len(self.ITEM_CONTINUAL_STATIC_FEAT)          # 若存在连续静态特征（float）
        dim += self._sum_embedding_dims(self.ITEM_DYNAMIC_FEAT.keys())  # 物品动态特征（稀疏）
        dim += len(self.ITEM_DYNAMIC_CONTINUAL_FEAT)         # 物品动态特征（连续）

        if include_time:
            dim += self._sum_embedding_dims(self.ITEM_TIME_SPARSE_FEAT.keys())
            dim += len(self.ITEM_CONTINUAL_FEAT)                 # 连续动态特征（float）
            dim += self.action_emb_append.embedding_dim  # action emb 维度
        dim += 32 * len(self.ITEM_EMB_FEAT)                 
        return dim

    def _calc_user_input_dim(self) -> int:
        #dim = self.user_emb.embedding_dim
        dim = 0
        dim += self._sum_embedding_dims(self.USER_SPARSE_FEAT.keys())
        dim += self._sum_embedding_dims(self.USER_ARRAY_FEAT.keys())
        dim += len(self.USER_CONTINUAL_FEAT)
        return dim
    # ------------------------------------------------------

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

        # 连续静态特征（float）这里默认无
        self.ITEM_CONTINUAL_STATIC_FEAT = []

        # 新增：时间离散特征（词表规模来自 dataset 的 feat_statistics）
        self.ITEM_TIME_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types.get('item_time_sparse', [])}

        # 新增：物品动态特征
        self.ITEM_DYNAMIC_FEAT = {k: feat_statistics[k] for k in feat_types.get('item_dynamic_sparse', [])}
        self.ITEM_DYNAMIC_CONTINUAL_FEAT = feat_types.get('item_dynamic_continual', [])

        # 每个特征的 embedding 维度（可按需调整）
        # 原始映射（保持你项目里的设定）
        self.each_embedding = {
            '103': 17, '104': 9, '105': 11, '109': 9,
            '106': 12, '107': 13, '108': 10, '110': 9,
            '100': 10, '117': 21, '111': 64, '118': 23, '101': 15, '102': 27,
            '119': 25, '120': 24, '114': 13, '112': 14, '121': 42, '115': 21,
            '122': 27, '116': 13
        }
        # 时间离散特征的 embedding 维度（轻量但区分度足够）
        time_dims = {
            '201': 9,  # day_of_week
            '202': 9,  # hour_of_day
            '203': 9,  # part_of_day
            '204': 32,  # delta_to_next_bucket（更重要更大维）
            '206': 16,  # recency_to_last_bucket
            '207': 11,  # is_new_session
            '208': 13,  # week_of_year
            '209': 9,  # month
            '210': 7,  # is_weekend
            '211': 21,  # hour_of_week
            '212': 8,  # is_holiday: 1=否, 2=节假日, 3=特殊事件
        }
        item_dynamic_dims = {
            '301': 13,  # expose_count
            '302': 13,  # click_count
            # '303': 13,  # ctr
            '306': 9,  # quantile_expose 4档：0=最低，1=75%+，2=95%+，3=99%+
            '307': 9,  # quantile_click  4档：0=最低，1=75%+，2=95%+，3=99%+
            '308': 7,  # abnormal_expose 2分类：0=否，1=是
            '309': 7,  # abnormal_click  2分类：0=否，1=是
            '310': 11,  # user_count
            '311': 10,  # expose_count_may
            '312': 11,  # click_count_may
            # '313': 11,  # ctr_may
            '314': 7,  # is_hot
        }
        self.each_embedding.update(time_dims)
        self.each_embedding.update(item_dynamic_dims)

    def _init_embeddings(self):
        
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

        for layer in self.attention_layers:
            nn.init.zeros_(layer.PinRec_FiLM[1].weight)
            nn.init.zeros_(layer.PinRec_FiLM[1].bias)
        
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

    # --------- 聚合 item / user 的特征 ----------
    def _gather_item_feats(self, seq, feature_tensors, include_time: bool):
        #item_embedding = self.item_emb(seq)
        item_embedding = torch.cat([self.item_emb(seq), self.item_emb_(seq)], dim=-1)
        item_feat_list = [item_embedding]

        # item 稀疏
        for k in self.ITEM_SPARSE_FEAT.keys():
            if k in feature_tensors:
                tensor_feature = feature_tensors[k].to(self.dev).long()
                item_feat_list.append(self.sparse_emb[k](tensor_feature))

        # item 数组（均值池化）
        for k in self.ITEM_ARRAY_FEAT.keys():
            if k in feature_tensors:
                tensor_feature = feature_tensors[k].to(self.dev).long()  # [B,S,L]
                emb = self.sparse_emb[k](tensor_feature)                 # [B,S,L,D]
                mask = (tensor_feature != 0).unsqueeze(-1)
                emb_sum = (emb * mask).sum(dim=2)
                counts = mask.sum(dim=2).clamp_min(1).to(emb_sum.dtype)
                item_feat_list.append(emb_sum / counts)                  # [B,S,D]

        # item 连续静态（float）——当前为空，但留接口
        for k in self.ITEM_CONTINUAL_STATIC_FEAT:
            if k in feature_tensors:
                tensor_feature = feature_tensors[k].to(self.dev).float().unsqueeze(2)  # [B,S,1]
                item_feat_list.append(tensor_feature)

        # 多模态
        for k in self.ITEM_EMB_FEAT:
            if k in feature_tensors:
                tensor_feature = feature_tensors[k].to(self.dev).float()
                item_feat_list.append(self.emb_transform[k](tensor_feature))            # [B,S,32]

        # 时间离散特征（仅在 include_time=True 时注入）
        if include_time:
            for k in self.ITEM_TIME_SPARSE_FEAT.keys():
                if k in feature_tensors:
                    tensor_feature = feature_tensors[k].to(self.dev).long()
                    item_feat_list.append(self.sparse_emb[k](tensor_feature))           # [B,S,Dk]
            for k in self.ITEM_CONTINUAL_FEAT:
                if k in feature_tensors:
                    tensor_feature = feature_tensors[k].to(self.dev).float().unsqueeze(2)  # [B,S,1]
                    item_feat_list.append(tensor_feature)
        # 物品动态特征
        for k in self.ITEM_DYNAMIC_FEAT.keys():
            if k in feature_tensors:
                tensor_feature = feature_tensors[k].to(self.dev).long()
                item_feat_list.append(self.sparse_emb[k](tensor_feature))           # [B,S,Dk]

        for k in self.ITEM_DYNAMIC_CONTINUAL_FEAT:
            if k in feature_tensors:
                tensor_feature = feature_tensors[k].to(self.dev).float().unsqueeze(2)  # [B,S,1]
                item_feat_list.append(tensor_feature)

        return item_embedding, item_feat_list

    def _gather_user_feats(self, seq, feature_tensors):
        user_feat_list = []

        for k in self.USER_SPARSE_FEAT:
            if k in feature_tensors:
                tensor_feature = feature_tensors[k].to(self.dev).long()
                user_feat_list.append(self.sparse_emb[k](tensor_feature))
        for k in self.USER_ARRAY_FEAT:
            if k in feature_tensors:
                tensor_feature = feature_tensors[k].to(self.dev).long()
                emb = self.sparse_emb[k](tensor_feature)
                mask_arr = (tensor_feature != 0).unsqueeze(-1)
                emb_sum = (emb * mask_arr).sum(dim=2)
                counts = mask_arr.sum(dim=2).clamp_min(1).to(emb_sum.dtype)
                user_feat_list.append(emb_sum / counts)
        for k in self.USER_CONTINUAL_FEAT:
            if k in feature_tensors:
                tensor_feature = feature_tensors[k].to(self.dev).float().unsqueeze(2)
                user_feat_list.append(tensor_feature)
        return None, user_feat_list

    def feat2emb(self, seq, feature_tensors, mask=None, include_user=False,action_type=None):
        """
        include_user=True ：历史序列分支（注入时间离散特征 + 用户侧）
        include_user=False：plain 分支（无时间特征，不混入用户侧），用于 pos/neg/ANN 目标
        """
        seq = seq.to(self.dev)
        if mask is not None:
            item_mask = (mask == 1).to(self.dev)
            item_seq = item_mask * seq  # [B,S]，屏蔽用户 ID
        else:
            item_seq = seq
        # item 分支
        if include_user:
            feature_item = feature_tensors[1]
            feature_user = feature_tensors[0]
        else:
            feature_item = feature_tensors
            feature_user = None

        _, item_feat_list = self._gather_item_feats(item_seq, feature_item, include_time=include_user)

        # SE + DNN（两套）
        if include_user:
            if action_type is not None:
                item_action_mask = (mask == 1).to(self.dev)
                action_ids = (action_type.to(self.dev) + 1) * item_action_mask
                action_embs = self.action_emb_append(action_ids)
                item_feat_list.append(action_embs)
            all_item_emb = self.apply_se_weighting(item_feat_list[0], item_feat_list[1:], self.item_SE_hist)
            all_item_emb = F.silu(self.itemdnn_hist(all_item_emb))
        else:
            all_item_emb = self.apply_se_weighting(item_feat_list[0], item_feat_list[1:], self.item_SE_plain)
            all_item_emb = F.silu(self.itemdnn_plain(all_item_emb))

        # user 分支（仅 include_user=True 时）
        if include_user:
            # 通过mask找出user所在位置的索引
            user_pos = (mask >= 2).nonzero(as_tuple=True)[1]
            # 依据user_pos从seq中取出user_id,形状为[B]
            user_id = seq.gather(1, user_pos.unsqueeze(1))
            user_embedding, user_feat_list = self._gather_user_feats(user_id, feature_user)
            all_user_emb = self.apply_se_weighting(user_feat_list[0], user_feat_list[1:], self.user_SE)
            all_user_emb = F.silu(self.userdnn(all_user_emb))
            # seqs_emb = all_item_emb + all_user_emb
            # 依据user_pos使用scatter_add_将user_emb加到对应位置
            seqs_emb = all_item_emb
            seqs_emb.scatter_add_(1, user_pos.unsqueeze(1).unsqueeze(2).expand(-1, -1, all_user_emb.size(2)), all_user_emb)
        else:
            seqs_emb = all_item_emb

        return seqs_emb  # [B,S,H]

    def log2feats(self, log_seqs, mask, seq_user_feature,seq_item_feature,input_interval=None, action_type=None, next_action_type=None, next_mask=None):
        
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]

        # Get sequence embeddings
        seqs = self.feat2emb(log_seqs, (seq_user_feature,seq_item_feature), mask=mask, include_user=True,action_type=action_type)

        # Add positional embeddings
        positions = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        positions *= (log_seqs != 0).long()
        pos_side1 = self.pos_emb(positions)  # [B, L, D]

        ltr_pos = (log_seqs != 0).long().cumsum(1)  # [B, L]
        pos_side2 = self.pos_emb_right(ltr_pos)   # [B, L, D]

        action_ids = (action_type.to(self.dev) + 1) * (mask == 1)
        action_side = self.action_emb(action_ids)  # [B, L, D]

        action_side = torch.cat([pos_side1, pos_side2, action_side], dim=-1)
        action_side = self.action_position_embedding(action_side)

        seqs *= self.pos_emb.embedding_dim ** 0.5
        seqs += action_side

        # Apply dropout
        # seqs = self.emb_dropout(seqs)

        # Create attention masks
        attention_mask_pad = ((mask>0)&(mask<=2)).to(self.dev)
        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask = (
            attention_mask_tril.unsqueeze(0)
            & attention_mask_pad.unsqueeze(1)
            & attention_mask_pad.unsqueeze(2)
        )

        # Ensure input_interval is the right type
        if input_interval is not None:
            input_interval = input_interval.long().to(self.dev)

        for i, layer in enumerate(self.attention_layers):
            if self.use_checkpoint and (i % self.ckpt_every == 0):
                def _layer_forward(x, input_interval, attention_mask, next_action_type, next_mask, _layer=layer):
                    with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                        y, _ = _layer(x, input_interval, attention_mask, next_action_type, next_mask)
                        return y

                seqs = checkpoint(
                    _layer_forward,
                    seqs, input_interval if input_interval is not None else torch.empty((), device=seqs.device),
                    attention_mask,
                    next_action_type, 
                    next_mask,
                    use_reentrant=False,  # 推荐用新 API，性能更好；含 Dropout 仍可用
                    preserve_rng_state=True  # 默认 True，保证 Dropout 前后重算时一致
                )
            else:
                seqs, _ = layer(seqs, input_interval, attention_mask, next_action_type, next_mask)
        
        seqs = self.last_layernorm(seqs)

        return seqs

    def forward(
        self,
        user_item,
        pos_seqs,
        neg_seqs,
        mask,
        next_mask,
        action_type,
        seq_user_feature,
        seq_item_feature,
        pos_feature,
        neg_feature,
        input_interval=None,
        next_action_type=None
    ):
        """Forward pass for training"""
        log_feats = self.log2feats(user_item, mask, seq_user_feature,seq_item_feature, input_interval, action_type, next_action_type, next_mask)  # [B, L, D]
        
        loss_mask = (next_mask == 1).to(self.dev)

        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)

        # 归一化
        log_feats = log_feats / log_feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        pos_embs = pos_embs / pos_embs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        neg_embs = neg_embs / neg_embs.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        neg_embs = neg_embs.reshape(-1, log_feats.shape[-1])
        neg_embs = neg_embs[loss_mask.view(-1)]

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        pos_logits = pos_logits * loss_mask

        q_valid  = log_feats[loss_mask]  # [M, D]
        neg_bank = neg_embs

        return pos_logits, None, (q_valid, neg_bank)

    def predict(self, log_seqs,seq_user_feat, seq_item_feat, mask, action_type, input_interval=None,target_action_type=None):
        """Prediction method"""
        next_action_type = torch.empty((log_seqs.size(0), log_seqs.size(1)), dtype=torch.long, device=log_seqs.device)
        next_action_type[:, :-1] = action_type[:, 1:]
        next_action_type[:, -1]  = target_action_type
        next_mask = torch.empty((log_seqs.size(0), log_seqs.size(1)), dtype=torch.long, device=log_seqs.device)
        next_mask[:, :-1] = mask[:, 1:]
        next_mask[:, -1]  = 1

        log_feats = self.log2feats(log_seqs, mask, seq_user_feat,seq_item_feat, input_interval, action_type, next_action_type, next_mask)
        final_feat = log_feats[:, -1, :]
        final_feat = final_feat / final_feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return final_feat