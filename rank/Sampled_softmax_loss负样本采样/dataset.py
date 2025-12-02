import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

def feat2tensor(seq_feature, k, feature_types):
    """
    Convert features to tensors (moved from model for multiprocessing support)
    
    Args:
        seq_feature: List of feature sequences
        k: Feature key/ID
        feature_types: Dictionary containing feature type classifications
        
    Returns:
        torch.Tensor: Converted feature tensor (without device placement)
    """
    batch_size = len(seq_feature)
    max_seq_len = len(seq_feature[0])
    item_array_feat = feature_types.get('item_array', [])
    user_array_feat = feature_types.get('user_array', [])
    item_emb_feat = feature_types.get('item_emb', [])

    if k in item_array_feat or k in user_array_feat:
        # Array features - ensure we handle lists properly
        max_array_len = 0
        max_seq_len = 0

        for i in range(batch_size):
            seq_data = [item.get(k, []) for item in seq_feature[i]]  # Use get with default
            max_seq_len = max(max_seq_len, len(seq_data))
            for item_data in seq_data:
                if isinstance(item_data, (list, np.ndarray)) and len(item_data) > 0:
                    max_array_len = max(max_array_len, len(item_data))

        batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
        try:
            for i in range(batch_size):
                seq_data = [item.get(k, []) for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    if isinstance(item_data, (list, np.ndarray)) and len(item_data) > 0:
                        actual_len = min(len(item_data), max_array_len)
                        batch_data[i, j, :actual_len] = item_data[:actual_len]
        except Exception as e:
            print(f"Error processing array feature {k}: {e}")
        return torch.from_numpy(batch_data)
    elif k in item_emb_feat:
        # Multimodal embeddings - these are already float vectors
        # We need to determine the embedding dimension from the first non-None value
        if k == '81':
            emb_dim = 32
        elif k == '82':
            emb_dim = 1024

        batch_emb_data = np.zeros((batch_size, max_seq_len, emb_dim), dtype=np.float32)

        for i in range(batch_size):
            for j in range(len(seq_feature[i])):
                item = seq_feature[i][j]
                if k in item and item[k] is not None:
                    if isinstance(item[k], (list, np.ndarray)):
                        batch_emb_data[i, j] = item[k]
                    else:
                        batch_emb_data[i, j] = item[k]

        return torch.from_numpy(batch_emb_data)
    else:
        # Sparse features - ensure integer type

        batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)

        for i in range(batch_size):
            seq_data = [item.get(k, 0) for item in seq_feature[i]]  # Use get with default
            # Ensure values are integers
            seq_data = [int(x) if isinstance(x, (int, float, np.number)) else 0 for x in seq_data]
            batch_data[i, :len(seq_data)] = seq_data

        return torch.from_numpy(batch_data)

class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id
        self.load_time_intervals = args.load_time_intervals  # 是否加载时间间隔数据

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
            print(f'用户数量为{self.usernum}, 物品数量为{self.itemnum}')
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()
        self.all_feat_ids = []
        for feat_type in self.feature_types.values():
            self.all_feat_ids.extend(feat_type)


    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        self.data_file_path = self.data_dir / "seq.jsonl"
        self.data_file = None
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _ensure_data_file(self):
        """
        Ensure data file is open for the current process (multiprocessing-safe)
        """
        if self.data_file is None:
            self.data_file = open(self.data_file_path, 'rb')
    
    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        self._ensure_data_file()
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = json.loads(line)
        return data

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)  
        # type=1 代表这是一个“物品事件”（用户点击/购买了某个物品）。type=2 代表这是一个“用户事件”（可能代表用户画像的更新或一次会话的开始）。
        ext_user_sequence = []
        for record_tuple in user_sequence:  # 一个列表包含多个tuple，可能是交互的物品信息，也可能是用户信息
            if self.load_time_intervals:
                u, i, user_feat, item_feat, action_type, timestamp = record_tuple
                if u and user_feat:
                    ext_user_sequence.insert(0, (u, user_feat, 2, action_type, timestamp))
                if i and item_feat:
                    ext_user_sequence.append((i, item_feat, 1, action_type, timestamp))
            else:
                u, i, user_feat, item_feat, action_type, _ = record_tuple
                if u and user_feat:
                    ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
                if i and item_feat:
                    ext_user_sequence.append((i, item_feat, 1, action_type))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)  
        pos = np.zeros([self.maxlen + 1], dtype=np.int32) 
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)  # 区分 seq 中的每个 ID 是用户（2）还是物品（1）
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        if self.load_time_intervals:
            cur_timestamp = np.zeros([self.maxlen + 1], dtype=np.int32)
            next_timestamp = np.zeros([self.maxlen + 1], dtype=np.int32)

        nxt = ext_user_sequence[-1]
        idx = self.maxlen  # 从最后一位开始插入信息

        total_item = set()  # 收集用户交互过的所有物品，用于负采样
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                total_item.add(record_tuple[0])

       
        input_sequence = ext_user_sequence[:-1]
        if len(input_sequence) > self.maxlen:
            user_info_tuple = input_sequence[0]
            item_sequence_tuples = input_sequence[1:]
            truncated_item_sequence = item_sequence_tuples[-(self.maxlen - 1):]
            input_sequence = [user_info_tuple] + truncated_item_sequence

        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(input_sequence):
            if self.load_time_intervals:
                i, feat, type_, act_type, ts = record_tuple
                next_i, next_feat, next_type, next_act_type, next_ts = nxt
            else:
                i, feat, type_, act_type = record_tuple
                next_i, next_feat, next_type, next_act_type = nxt
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            seq[idx] = i
            token_type[idx] = type_
            
            if type_ == 1:
                if act_type is not None:
                    action_type[idx] = act_type
                else:
                    action_type[idx] = 0  # 如果action_type为None，则设置为0
                if self.load_time_intervals:
                    cur_timestamp[idx] = ts
                    
            next_token_type[idx] = next_type
            
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
                if self.load_time_intervals:
                    next_timestamp[idx] = next_ts
                    
            seq_feat[idx] = feat  
            
            #  如果下一个事件是物品，则创建正负样本对
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                neg_id = self._random_neq(1, self.itemnum + 1, total_item)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break
        # 用户的历史序列比 maxlen短，那么填充循环结束后，seq_feat 等数组的左侧部分会是空的（None）。这行代码将所有 None 替换为一个包含所有特征默认值的字典 self.feature_default_value。这确保了输入给模型的数据是干净、无 None 的
        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)
        if self.load_time_intervals:
            return seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, cur_timestamp, next_timestamp
        else:
            return seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100',
            '117',
            #'111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            #'121',
            '115',
            '122',
            '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])

        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])

        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])

        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])

        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        missing_fields = set(self.all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    def collate_fn(self, batch):
        """
        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            seq_feat: 用户序列特征, 预处理为tensor字典形式
            pos_feat: 正样本特征, 预处理为tensor字典形式  
            neg_feat: 负样本特征, 预处理为tensor字典形式
        """
        if len(batch[0]) == 12:  # 如果加载了时间间隔数据
            seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, cur_timestamp, next_timestamp = zip(
                *batch)
            cur_timestamp = torch.from_numpy(np.array(cur_timestamp))
            next_timestamp = torch.from_numpy(np.array(next_timestamp))
        else:
            seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(
                *batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        action_type = torch.from_numpy(np.array(action_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        # Convert features to tensors using feat2tensor function

        # Process each feature type
        seq_feat_tensors = {}
        pos_feat_tensors = {}  
        neg_feat_tensors = {}

        for k in self.all_feat_ids:
            # Convert seq_feat
            seq_feat_tensors[k] = feat2tensor(list(seq_feat), k, self.feature_types)
            # Convert pos_feat  
            pos_feat_tensors[k] = feat2tensor(list(pos_feat), k, self.feature_types)
            # Convert neg_feat
            neg_feat_tensors[k] = feat2tensor(list(neg_feat), k, self.feature_types)
        if len(batch[0]) == 12:  # 如果加载了时间间隔数据
            return seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat_tensors, pos_feat_tensors, neg_feat_tensors, cur_timestamp, next_timestamp
        else:
            return seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat_tensors, pos_feat_tensors, neg_feat_tensors


class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        # Store path instead of file handle to support multiprocessing
        self.data_file_path = self.data_dir / "predict_seq.jsonl"
        self.data_file = None  # Will be opened in each worker
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            action_type: 动作类型
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
            [cur_timestamp]: 当前时间戳 (如果load_time_intervals为True)
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        for record_tuple in user_sequence:
            if self.load_time_intervals:
                # 如果需要加载时间间隔数据，则需要将时间间隔信息也加入到用户序列中
                u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            else:
                # 如果不需要加载时间间隔数据，则将时间戳设置为None
                u, i, user_feat, item_feat, action_type, _ = record_tuple

            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]

            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                if self.load_time_intervals:
                    ext_user_sequence.insert(0, (u, user_feat, 2, action_type, timestamp))
                else:
                    ext_user_sequence.insert(0, (u, user_feat, 2, action_type))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                if self.load_time_intervals:
                    ext_user_sequence.append((i, item_feat, 1, action_type, timestamp))
                else:
                    ext_user_sequence.append((i, item_feat, 1, action_type))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)

        # 添加时间戳数组初始化
        if self.load_time_intervals:
            cur_timestamp = np.zeros([self.maxlen + 1], dtype=np.int32)

        input_sequence = ext_user_sequence
        if len(input_sequence) > self.maxlen:
            user_info_tuple = input_sequence[0]
            item_sequence_tuples = input_sequence[1:]
            # 对物品序列进行截断，只保留最新的 maxlen-1 个
            truncated_item_sequence = item_sequence_tuples[-(self.maxlen - 1):]
            # 将用户信息和截断后的物品序列重新组合
            input_sequence = [user_info_tuple] + truncated_item_sequence

        idx = self.maxlen
        for record_tuple in reversed(input_sequence):
            if self.load_time_intervals:
                i, feat, type_, act_type, ts = record_tuple
            else:
                i, feat, type_, act_type = record_tuple

            feat = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            if type_ == 1:
                if act_type is not None:
                    action_type[idx] = act_type
                else:
                    action_type[idx] = 0  # 如果action_type为None，则设置为0
                if self.load_time_intervals:
                    cur_timestamp[idx] = ts
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        if self.load_time_intervals:
            cur_timestamp = np.where(cur_timestamp == 0, -1, cur_timestamp)  # 将未填充的时间戳设置为-1
            return seq, token_type, action_type, seq_feat, user_id, cur_timestamp
        else:
            return seq, token_type, action_type, seq_feat, user_id

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    def collate_fn(self, batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            action_type: 动作类型, torch.Tensor形式
            seq_feat: 用户序列特征, 预处理为tensor字典形式
            user_id: user_id, list of strings
            [cur_timestamp]: 当前时间戳, torch.Tensor形式 (如果有时间戳数据)
        """
        if len(batch[0]) == 6:  # 如果加载了时间间隔数据
            seq, token_type, action_type, seq_feat, user_id, cur_timestamp = zip(*batch)
            cur_timestamp = torch.from_numpy(np.array(cur_timestamp))
        else:
            seq, token_type, action_type, seq_feat, user_id = zip(*batch)

        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        action_type = torch.from_numpy(np.array(action_type))
        # Convert features to tensors using feat2tensor function
        # Get all unique feature keys

        # Process each feature type
        seq_feat_tensors = {}

        for k in self.all_feat_ids:
            # Convert seq_feat
            seq_feat_tensors[k] = feat2tensor(list(seq_feat), k, self.feature_types)

        if len(batch[0]) == 6:  # 如果加载了时间间隔数据
            return seq, token_type, action_type, seq_feat_tensors, user_id, cur_timestamp
        else:
            return seq, token_type, action_type, seq_feat_tensors, user_id


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in feat_ids:
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        # 依据pkl文件是否存在选择加载方式，而不是依据feat_id
        if not Path(mm_path, f'emb_{feat_id}_{shape}.pkl').exists():
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('part-*'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            if 'emb' not in data_dict_origin:
                                continue
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        else:
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict