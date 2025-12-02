from copy import deepcopy
import json
import os
import pickle
import struct
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np
import torch
from tqdm import tqdm
from tools import LEGAL_2024_2025,E_COMMERCE
      
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
    item_continual_feat = feature_types.get('item_continual', [])
    user_continual_feat = feature_types.get('user_continual', [])
    item_dynamic_continual_feat = feature_types.get('item_dynamic_continual', [])

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
        if k in user_array_feat:
            batch_data = np.zeros((batch_size, 1, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item.get(k, []) for item in seq_feature[i]]
                if len(seq_data) > 0:
                    item_data = seq_data[0]
                    if isinstance(item_data, (list, np.ndarray)) and len(item_data) > 0:
                        actual_len = min(len(item_data), max_array_len)
                        batch_data[i, 0, :actual_len] = item_data[:actual_len]
            return torch.from_numpy(batch_data)
        batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
        for i in range(batch_size):
            seq_data = [item.get(k, []) for item in seq_feature[i]]
            for j, item_data in enumerate(seq_data):
                if isinstance(item_data, (list, np.ndarray)) and len(item_data) > 0:
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]
        return torch.from_numpy(batch_data)
    elif k in item_emb_feat:
        # Multimodal embeddings - these are already float vectors
        # We need to determine the embedding dimension from the first non-None value
        if k == '81':
            emb_dim = 32
        elif k == '82':
            emb_dim = 1024
        elif k == '83':
            emb_dim = 3584
        elif k == '84':
            emb_dim = 4096
        elif k == '85':
            emb_dim = 3584
        elif k == '86':
            emb_dim = 3584
        else:
            raise ValueError(f"Unknown item_emb feat id {k}")

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
    elif k in item_continual_feat or k in user_continual_feat or k in item_dynamic_continual_feat:
        if k in user_continual_feat:
            batch_data = np.zeros((batch_size, 1), dtype=np.float32)
            for i in range(batch_size):
                seq_data = [item.get(k, 0.0) for item in seq_feature[i]]
                if len(seq_data) > 0:
                    val = seq_data[0]
                    batch_data[i, 0] = float(val) if isinstance(val, (int, float, np.number)) else 0.0
            return torch.from_numpy(batch_data)

        batch_data = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        for i in range(batch_size):
            seq_data = [item.get(k, 0.0) for item in seq_feature[i]]
            seq_data = [float(x) if isinstance(x, (int, float, np.number)) else 0.0 for x in seq_data]
            batch_data[i, :len(seq_data)] = seq_data
        return torch.from_numpy(batch_data)
    else:
        if k in feature_types.get('user_sparse', []):
            batch_data = np.zeros((batch_size, 1), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item.get(k, 0) for item in seq_feature[i]]
                if len(seq_data) > 0:
                    val = seq_data[0]
                    batch_data[i, 0] = int(val) if isinstance(val, (int, float, np.number)) else 0
            return torch.from_numpy(batch_data)

        batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        for i in range(batch_size):
            seq_data = [item.get(k, 0) for item in seq_feature[i]]
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

        # cache_dir = os.environ.get('USER_CACHE_PATH', '.')
        # combine_path = os.path.join(cache_dir, "item_expose.json") # 这个文件用于记录曝光次数小于等于1的物品的creative_id

        # if not os.path.exists(combine_path):
        #     print(f"文件 {combine_path} 不存在，请先运行 create_feat() 生成物品曝光文件")
        #     raise FileNotFoundError(f"{combine_path} not found")
        # with open(combine_path, "r", encoding="utf-8") as f:
        #     self.item_expose = json.load(f)

        # 判断Path(data_dir, "creative_emb")是否存在
        if not os.path.exists(Path(data_dir, "creative_emb")):
            self.mm_emb_dict = load_mm_emb(Path('TencentGR_1k_new',"creative_emb"), self.mm_emb_ids)
        else:
            self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
            print(f'用户数量为{self.usernum}, 物品数量为{self.itemnum}')
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        # 各离散时间特征的词表规模（不含 padding 的 0）
        self.TIME_SPARSE_VOCAB = {
                '201': 7,                        # dow
                '202': 24,                       # hour
                '203': 6,                        # part of day
                '204': 129,                     # delta_to_next
                '206': 129,                     # recency_to_last
                '207': 4,                        # is_new_session
                '208': 53,                       # week of year
                '209': 12,                       # month
                '210': 2,                        # is_weekend
                '211': 168,                      # 
                '212': 3,                        # is_holiday: 1=否, 2=节假日, 3=特殊事件
            }
        self.ITEM_DYNAMIC_FEAT_VOCAB = {
                '301': 6,    # expose_count
                '302': 6,    # click_count
                '303': 6,    # ctr
                # '304': ,    # rank_expose
                # '305': ,    # rank_click
                '306': 4,    # quantile_expose 4档：0=最低，1=75%+，2=95%+，3=99%+
                '307': 4,    # quantile_click  4档：0=最低，1=75%+，2=95%+，3=99%+
                '308': 2,    # abnormal_expose 2分类：0=否，1=是
                '309': 2,    # abnormal_click  2分类：0=否，1=是
                '310': 6,    # user_count
                '311': 5,    # expose_count_may
                '312': 6,    # click_count_may
                '313': 6,    # ctr_may
                '314': 2,    # is_hot
            }
        self.feat_top3_dict = {
                "121": [1733228, 4406404, 2407089],
                "116": [16, 19, 20],
                "122": [8, 2, 6],
                "115": [370, 895, 534],
                "112": [14, 10, 1],
                "114": [32, 22, 10],
                "120": [2783, 1370, 1810],
                "119": [3415, 206, 3378],
                "102": [17473, 147455, 120933],
                "101": [11, 10, 18],
                "118": [1173, 1519, 1342],
                "117": [67, 245, 350],
                "100": [5, 6, 3]
            }
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()
        self.all_feat_ids = []
        for feat_type in self.feature_types.values():
            self.all_feat_ids.extend(feat_type)
        self.user_feat_ids = []
        self.item_feat_ids = []
        self.user_default_value = {}
        self.item_default_value = {}
        # 划分用户和物品特征
        for k,v in self.feature_types.items():
            if k.startswith('user_'):
                self.user_feat_ids.extend(v)
                self.user_default_value.update({key: self.feature_default_value[key] for key in v})
            elif k.startswith('item_'):
                self.item_feat_ids.extend(v)
                self.item_default_value.update({key: self.feature_default_value[key] for key in v})
        self.item_orgin_feat = self.feature_types['item_sparse'] + self.feature_types['item_array'] + self.feature_types['item_emb']

        self.num_neg = getattr(args, "num_neg", 3)

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
            '100', '117', '118', '101', '102', '119', '120', '114', '112', '122', '116', '121'
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = ['201_sin', '201_cos', '202_sin', '202_cos', '211_sin', '211_cos']

        # 新增：时间离散特征组（不依赖 indexer 统计）
        feat_types['item_time_sparse'] = ['201', '202', '203', '204', '206', '207', #'208', '209', 
                                          '210','211','212']
        # feat_types['item_dynamic_sparse'] = ['301', '302', '306',
        #                                      '307', '308', '309', '310', '311', '312',
        #                                      '314'] # 去除 '304', '305'（排名特征）
        # 将ctr和ctr_may从sparse改为continual
        # feat_types['item_dynamic_continual'] = ['303','313']
        feat_types['item_dynamic_continual'] = []
        feat_types['item_dynamic_sparse'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])

        for feat_id in feat_types['item_sparse']:
            # feat_default_value[feat_id] = 0
            # 不使用0作为缺失值，而是使用top3中出现频率最高的值
            feat_default_value[feat_id] = self.feat_top3_dict.get(feat_id, [0])[0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])

        # 时间离散特征的统计/默认值（手工词表）
        for feat_id in feat_types['item_time_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = self.TIME_SPARSE_VOCAB[feat_id]

        for feat_id in feat_types['item_dynamic_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = self.ITEM_DYNAMIC_FEAT_VOCAB[feat_id]

        for feat_id in feat_types['item_dynamic_continual']:
            feat_default_value[feat_id] = 0.0

        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])

        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])

        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0.0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0.0

        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id,is_user=False):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if is_user:
            all_feat_ids = self.user_feat_ids
        else:
            all_feat_ids = self.item_feat_ids
            for feat_id in self.feature_types['item_emb']:
                if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                    if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                        feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]
            missing_orginal_feats = set(self.item_orgin_feat) - set(feat.keys())
            for feat_id in missing_orginal_feats:
                if feat_id in self.item_feat_dict.get(str(item_id),{}):
                    feat[feat_id] = self.item_feat_dict[str(item_id)][feat_id]
                else:
                    feat[feat_id] = self.feature_default_value[feat_id]
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            feat[feat_id] = self.feature_default_value[feat_id]
        return feat

    def _part_of_day(self,hour: int) -> int:
        if 0 <= hour <= 4:
            return 0
        if 5 <= hour <= 7:
            return 1
        if 8 <= hour <= 10:
            return 2
        if 11 <= hour <= 16:
            return 3
        if 17 <= hour <= 20:
            return 4
        return 5  # 21..23
    
    def _is_holiday_or_event(self,dt):
        if LEGAL_2024_2025.get(dt, False):
            return 2  # 节假日
        if E_COMMERCE.get(dt, False):
            return 3  # 电商大促
        return 1      # 非节假日

    def _minutes_nonneg(self,delta_seconds: int | float | None) -> float:
        if delta_seconds is None:
            return 0.0
        return max(0.0, float(delta_seconds) / 60.0)

    def _bucket_id_from_sec(self, sec: float,k) -> int:
        # HSTU log分桶方式
        bucket_id = int(np.log(max(sec, 1)) / 0.301)
        return min(max(bucket_id, 0), 128) + 1  # 128为桶数量，可与模型参数一致

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

    def _inject_time_feats(self, feat: dict, ts: int | None,
                           next_ts_for_delta: int | None,
                           last_ts_for_recency: int | None) -> dict:
        """
        把时间戳衍生为离散特征 id（全部 embedding）。
        """
        if feat is None:
            feat = {}
        if ts is None or ts <= 0:
            return feat
        try:
            dt = datetime.fromtimestamp(ts, timezone(timedelta(hours=8))) # 可能是北京时间，也可能是 UTC 时间，后续做一下对比实验
            dow = dt.weekday()                # 0..6
            hour = dt.hour                    # 0..23
            pod = self._part_of_day(hour)     # 0..5
            week = int(dt.isocalendar().week) # 1..53
            month = dt.month                  # 1..12
            is_weekend = 1 if dow < 5 else 2  # 1=工作日 2=周末
            # 与下一条（更近事件）的时间差
            if next_ts_for_delta is not None:
                sec_diff = int(next_ts_for_delta) - int(ts)
                m_next = self._minutes_nonneg(int(next_ts_for_delta) - int(ts))
                bucket_next = self._bucket_id_from_sec(sec_diff,'204')
                # is_new_session = 2 if m_next >= 30.0 else 1  # 30min 作为会话边界
                # 多级会话桶用 4 桶替代二分类：≤30m/≤60m/≤120m/>120m，对应 id 2/3/4/5（id=1 保留“否”）
                if m_next < 30.0:
                    session_gap_bucket = 1
                elif m_next < 60.0:
                    session_gap_bucket = 2
                elif m_next < 120.0:
                    session_gap_bucket = 3
                else:
                    session_gap_bucket = 4

            else:
                bucket_next = 0
                session_gap_bucket = 0

            # 与“序列最新一次交互”的时间差（recency）
            if last_ts_for_recency is not None:
                sec_diff = int(last_ts_for_recency) - int(ts)
                bucket_last = self._bucket_id_from_sec(sec_diff,'206')
            else:
                bucket_last = 0

            for k in self.feature_types['item_time_sparse']:
                if k == '201':
                    feat[k] = int(dow + 1)     # 1..7
                elif k == '202':
                    feat[k] = int(hour + 1)    # 1..24
                elif k == '203':
                    feat[k] = int(pod + 1)    # 1..6
                elif k == '204':
                    feat[k] = int(bucket_next) # delta_to_next bucket
                elif k == '206':
                    feat[k] = int(bucket_last) # recency_to_last bucket
                elif k == '207':
                    feat[k] = int(session_gap_bucket)
                elif k == '208':
                    feat[k] = int(week)        # 1..53
                elif k == '209':
                    feat[k] = int(month)       # 1..12
                elif k == '210':
                    feat[k] = int(is_weekend)  # 1/2
                elif k == '211':
                    feat[k] = int(dow * 24 + hour + 1)  # hour_of_week 1..168
                elif k == '212':
                    is_holiday = self._is_holiday_or_event(dt.strftime("%Y-%m-%d"))
                    feat[k] = is_holiday # 1=否, 2=节假日, 3=电商大促
            for k in self.feature_types['item_continual']:
                if k == '201_sin':
                    feat[k] = np.sin(2 * np.pi * (dow + 1) / 7)
                elif k == '201_cos':
                    feat[k] = np.cos(2 * np.pi * (dow + 1) / 7)
                elif k == '202_sin':
                    feat[k] = np.sin(2 * np.pi * (hour + 1) / 24)
                elif k == '202_cos':
                    feat[k] = np.cos(2 * np.pi * (hour + 1) / 24)
                elif k == '211_sin':
                    feat[k] = np.sin(2 * np.pi * (dow * 24 + hour + 1) / 168)
                elif k == '211_cos':
                    feat[k] = np.cos(2 * np.pi * (dow * 24 + hour + 1) / 168)

        except Exception as e:
            print(f"Error processing timestamp {ts}: {e}")
            raise e
        return feat

    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)  
        # type=1 代表这是一个“物品事件”（用户点击/购买了某个物品）。type=2 代表这是一个“用户事件”（可能代表用户画像的更新或一次会话的开始）。
        ext_user_sequence = []
        have_user_info = False
        for record_tuple in user_sequence:  # 一个列表包含多个tuple，可能是交互的物品信息，也可能是用户信息
            if self.load_time_intervals:
                u, i, user_feat, item_feat, action_type, timestamp = record_tuple
                if u and user_feat:
                    have_user_info = True
                    ext_user_sequence.insert(0, (u, user_feat, 2, action_type, timestamp))
                if i and item_feat:
                    ext_user_sequence.append((i, item_feat, 1, action_type, timestamp))
        if not have_user_info:
            # 如果序列中没有用户信息，则添加一个默认的用户信息
            default_user_feat = {k: self.user_default_value[k] for k in self.user_feat_ids}
            if self.load_time_intervals:
                ext_user_sequence.insert(0, (0, default_user_feat, 3, 0, None))
            else:
                ext_user_sequence.insert(0, (0, default_user_feat, 3, 0))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)  
        pos = np.zeros([self.maxlen + 1], dtype=np.int32) 
        neg = np.zeros([self.maxlen + 1, self.num_neg], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)  # 区分 seq 中的每个 ID 是用户（2）还是物品（1）
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        # seq_feat = np.empty([self.maxlen + 1], dtype=object)
        # 分离用户和物品的特征，节省内存提高效率
        seq_feat_user = np.empty(1, dtype=object) 
        seq_feat_item = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1, self.num_neg], dtype=object)

        if self.load_time_intervals:
            cur_timestamp = np.zeros([self.maxlen + 1], dtype=np.int32)
            next_timestamp = np.zeros([self.maxlen + 1], dtype=np.int32)

        # 用于 recency 的“序列最新一次交互时间”，注意，不应该取ext_user_sequence的最后一条
        # 因为在训练的时候，最后一条通常是预测目标，不应该被用来计算 recency，而是取input_sequence中的最后一条
        input_sequence = ext_user_sequence[:-1]
        last_ts_for_recency = None
        if self.load_time_intervals:
            for tup in reversed(input_sequence):
                if tup[2] == 1 and tup[-1] is not None:
                    last_ts_for_recency = int(tup[-1])
                    break

        nxt = ext_user_sequence[-1]
        idx = self.maxlen  # 从最后一位开始插入信息

        total_item = set()  # 收集用户交互过的所有物品，用于负采样
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                total_item.add(record_tuple[0])

        next_ts = None

        if len(input_sequence) > self.maxlen:
            user_info_tuple = input_sequence[0]
            item_sequence_tuples = input_sequence[1:]
            truncated_item_sequence = item_sequence_tuples[-(self.maxlen - 1):]
            input_sequence = [user_info_tuple] + truncated_item_sequence

        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(input_sequence):
            if self.load_time_intervals:
                i, feat, type_, act_type, ts = record_tuple
                next_i, next_feat, next_type, next_act_type, _ = nxt
            else:
                i, feat, type_, act_type = record_tuple
                next_i, next_feat, next_type, next_act_type = nxt
                ts = None
                next_ts = None

            # 只对历史 item 事件注入离散时间特征
            if type_ == 1:
                self._inject_time_feats(feat, ts, next_ts, last_ts_for_recency)
                feat = self.fill_missing_feat(feat, i,is_user=False)
                seq_feat_item[idx] = feat
                # 使用重映射表进行ID转换
                # i = self.item_expose[str(i)]
            else:
                feat = self.fill_missing_feat(feat, i,is_user=True)
                seq_feat_user[0] = feat
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
                    next_timestamp[idx] = next_ts if next_ts is not None else -1
                    
            # seq_feat[idx] = feat  # 旧版，现在分离为用户和物品特征
            #  如果下一个事件是物品，则创建正负样本对
            if next_type == 1 and next_i != 0:
                next_feat = self.fill_missing_feat(next_feat, next_i,is_user=False)
                pos_feat[idx] = next_feat
                # 使用重映射表进行ID转换
                # next_i = self.item_expose[str(next_i)]
                pos[idx] = next_i
                for n in range(self.num_neg):
                    neg_id = self._random_neq(1, self.itemnum + 1, total_item)
                    neg_feat_i = deepcopy(self.item_feat_dict.get(str(neg_id), {}))
                    neg_feat[idx, n] = self.fill_missing_feat(neg_feat_i, neg_id, is_user=False)
                    neg[idx, n] = neg_id
            nxt = record_tuple
            next_ts = ts if ts is not None else next_ts
            idx -= 1
            if idx == -1:
                break
        # 用户的历史序列比 maxlen短，那么填充循环结束后，seq_feat 等数组的左侧部分会是空的（None）。这行代码将所有 None 替换为一个包含所有特征默认值的字典 self.feature_default_value。这确保了输入给模型的数据是干净、无 None 的
        seq_feat_item = np.where(seq_feat_item == None, self.item_default_value, seq_feat_item)
        pos_feat = np.where(pos_feat == None, self.item_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.item_default_value, neg_feat)
        if self.load_time_intervals:
            return seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat_user, seq_feat_item, pos_feat, neg_feat, cur_timestamp, next_timestamp
        else:
            return seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat_user, seq_feat_item, pos_feat, neg_feat
    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)



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

        if len(batch[0]) == 13:  # 如果加载了时间间隔数据
            seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat_user, seq_feat_item, pos_feat, neg_feat, cur_timestamp, next_timestamp = zip(
                *batch)
            cur_timestamp = torch.from_numpy(np.array(cur_timestamp))
            next_timestamp = torch.from_numpy(np.array(next_timestamp))
        else:
            seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat_user, seq_feat_item, pos_feat, neg_feat = zip(
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
        # seq_feat_tensors = {}
        seq_feat_item_tensors = {}
        seq_feat_user_tensors = {}
        pos_feat_tensors = {}  
        neg_feat_tensors = {}

        for k in self.user_feat_ids:
            # Convert seq_feat_user
            seq_feat_user_tensors[k] = feat2tensor(list(seq_feat_user), k, self.feature_types)
        for k in self.item_feat_ids:
            # Convert seq_feat_item
            seq_feat_item_tensors[k] = feat2tensor(list(seq_feat_item), k, self.feature_types)
            # Convert pos_feat  
            pos_feat_tensors[k] = feat2tensor(list(pos_feat), k, self.feature_types)
        # 关键：把 neg_feat 从 [B] 个 [L, N] 的 object 矩阵 → B*N 条“长度 L 的序列”列表
        B = len(neg_feat)
        L = neg.shape[1]
        N = neg.shape[2]
        neg_feat_seq_list = []
        for b in range(B):
            arr = neg_feat[b]  # ndarray [L, N]，每格是一个 dict
            for n in range(N):
                one_seq = [arr[t, n] for t in range(L)]  # 这一条负样本的“序列特征”
                neg_feat_seq_list.append(one_seq)
        for k in self.item_feat_ids:
            neg_feat_tensors[k] = feat2tensor(neg_feat_seq_list, k, self.feature_types)

        if len(batch[0]) == 13:
            return (seq, pos, neg, token_type, action_type, next_token_type, next_action_type,
                    seq_feat_user_tensors, seq_feat_item_tensors, pos_feat_tensors, neg_feat_tensors,
                    cur_timestamp, next_timestamp)
        else:
            return (seq, pos, neg, token_type, action_type, next_token_type, next_action_type,
                    seq_feat_user_tensors, seq_feat_item_tensors, pos_feat_tensors, neg_feat_tensors)
class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

        # 读取data_path下的user_action_types.json文件，获取预测目标的action_type
        action_type_path = data_dir / 'user_action_type.json'
        with open(action_type_path, 'r') as f:
            user_action_types = json.load(f)
        self.target_action = user_action_types
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
        have_user_info = False
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
                    have_user_info = True
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
        if not have_user_info:
            # 如果序列中没有用户信息，则添加一个默认的用户信息
            default_user_feat = {k: self.user_default_value[k] for k in self.user_feat_ids}
            if self.load_time_intervals:
                ext_user_sequence.insert(0, (0, default_user_feat, 3, 0, None))
            else:
                ext_user_sequence.insert(0, (0, default_user_feat, 3, 0))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        # seq_feat = np.empty([self.maxlen + 1], dtype=object)
        seq_user_feat = np.empty(1, dtype=object)
        seq_item_feat = np.empty([self.maxlen + 1], dtype=object)
        target_act_type = self.target_action.get(user_id, 1)

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

        # 找出“该序列最新一次 item 交互”时间（用于 recency）
        last_ts_for_recency = None
        if self.load_time_intervals:
            for tup in reversed(input_sequence):
                if len(tup) >= 5 and tup[2] == 1 and tup[-1] is not None:
                    last_ts_for_recency = int(tup[-1])
                    break

        idx = self.maxlen
        nxt_ts = None  # 用于 delta_to_next（与训练一致）

        for record_tuple in reversed(input_sequence):
            if self.load_time_intervals:
                i, feat, type_, act_type, ts = record_tuple
            else:
                i, feat, type_, act_type = record_tuple
                ts = None

            if type_ == 1:
                self._inject_time_feats(feat, ts, nxt_ts, last_ts_for_recency)


            token_type[idx] = type_
            if type_ == 1:
                if act_type is not None:
                    action_type[idx] = act_type
                else:
                    action_type[idx] = 0  # 如果action_type为None，则设置为0
                if self.load_time_intervals:
                    cur_timestamp[idx] = ts
                self.fill_missing_feat(feat, i,is_user=False)
                seq_item_feat[idx] = feat
                # 使用重映射表进行ID转换
                # i = self.item_expose[str(i)]
            else:
                self.fill_missing_feat(feat, i,is_user=True)
                seq_user_feat[0] = feat

            # 更新下一次“更近事件”时间
            nxt_ts = ts if ts is not None else nxt_ts
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        seq_user_feat = np.where(seq_user_feat == None, self.user_default_value, seq_user_feat)
        seq_item_feat = np.where(seq_item_feat == None, self.item_default_value, seq_item_feat)

        if self.load_time_intervals:
            cur_timestamp = np.where(cur_timestamp == 0, -1, cur_timestamp)  # 将未填充的时间戳设置为-1
            return seq, token_type, action_type, seq_user_feat, seq_item_feat, user_id, cur_timestamp, target_act_type
        else:
            return seq, token_type, action_type, seq_user_feat, seq_item_feat, user_id

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
        if len(batch[0]) == 8:  # 如果加载了时间间隔数据
            # seq, token_type, action_type, seq_feat, user_id, cur_timestamp = zip(*batch)
            seq, token_type, action_type, seq_feat_user, seq_feat_item, user_id, cur_timestamp,target_actions = zip(*batch)
            cur_timestamp = torch.from_numpy(np.array(cur_timestamp))
        else:
            seq, token_type, action_type, seq_feat, user_id = zip(*batch)

        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        action_type = torch.from_numpy(np.array(action_type))
        target_actions = torch.from_numpy(np.array(target_actions))

        seq_feat_user_tensors = {}
        seq_feat_item_tensors = {}
        for k in self.user_feat_ids:
            # Convert seq_feat_user
            seq_feat_user_tensors[k] = feat2tensor(list(seq_feat_user), k, self.feature_types)
        for k in self.item_feat_ids:
            # Convert seq_feat_item
            seq_feat_item_tensors[k] = feat2tensor(list(seq_feat_item), k, self.feature_types)


        if len(batch[0]) == 8:  # 如果加载了时间间隔数据
            return seq, token_type, action_type, seq_feat_user_tensors, seq_feat_item_tensors, user_id, cur_timestamp, target_actions
        else:
            return seq, token_type, action_type, seq_feat_user_tensors, seq_feat_item_tensors, user_id


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
