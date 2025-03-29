import random
from collections import defaultdict

from tqdm import tqdm


class TripletCollector:
    def __init__(
        self,
        names_pub: dict,
        author_pub_labels: dict,
        packing_size: int,
        positive_ratio: float = None,
        positive_num: int = None,
        shuffle: bool = True,
        random_seed: int = None,
    ):
        assert positive_ratio is not None or positive_num is not None, (
            "positive_ratio and positive_num cannot be both None"
        )
        assert positive_ratio is None or positive_num is None, (
            "positive_ratio and positive_num cannot be both provided"
        )

        # 保存参数以便重用
        self.packing_size = packing_size
        self.positive_ratio = positive_ratio
        self.positive_num = positive_num
        self.shuffle = shuffle
        self.random_seed = random_seed

        # 如果提供了随机种子，设置随机状态以确保可重现性
        if random_seed is not None:
            random.seed(random_seed)

        # 初始化数据结构
        self.pub_to_author = {}
        self.labels = {}
        self.author_label_map = defaultdict(lambda: defaultdict(list))
        self.label_author_map = defaultdict(lambda: defaultdict(list))
        self.global_pub_pools = []

        # 加载数据并构建索引
        for author, pubs in author_pub_labels.items():
            for pub, label in pubs.items():
                self.pub_to_author[pub] = author
                self.labels[pub] = label
                self.author_label_map[author][label].append(pub)
                self.label_author_map[label][author].append(pub)
                self.global_pub_pools.append((pub, author, label))

        # 获取所有可用的发布物列表
        self.all_pubs = list(self.labels.keys())

        # 构建数据集
        self.build_dataset()

    def build_dataset(self):
        """构建数据集，可以在需要时重新生成"""
        self.data = []

        # 确定要处理的发布物列表
        working_pubs = self.all_pubs.copy()
        if not self.shuffle:
            # 当不混洗时，按照确定性顺序处理发布物
            working_pubs.sort()  # 确保确定性顺序

        # 创建已使用的正样本跟踪
        used_pos_samples = set() if not self.shuffle else None

        # 对每个发布物构建三元组
        for pub in tqdm(working_pubs, desc="Constructing the triplet dataset"):
            label = self.labels[pub]
            triplet = self._get_triplet(pub, label, used_pos_samples)
            if triplet:  # 只有当能成功创建三元组时才添加
                self.data.append(triplet)

    def _get_triplet(self, pub, label, used_pos_samples=None):
        """为一个发布物生成三元组

        Args:
            pub: 查询发布物
            label: 查询发布物的标签
            used_pos_samples: 已使用的正样本集合（仅在不混洗时使用）

        Returns:
            包含查询、正样本和负样本的打包列表
        """
        author = self.pub_to_author[pub]
        packing = [pub]  # 第一个位置固定为query

        # Level 1: 同作者同标签正样本
        same_author_pos = [p for p in self.author_label_map[author][label] if p != pub]

        # 如果跟踪已使用样本，则排除它们
        if used_pos_samples is not None:
            same_author_pos = [p for p in same_author_pos if p not in used_pos_samples]

        # 确定要采样的正样本数量
        pos_sampling_num = (
            self.positive_num
            if self.positive_num is not None
            else max(1, int(len(same_author_pos) * self.positive_ratio))
        )
        pos_sampling_num = min(len(same_author_pos), pos_sampling_num)

        # 如果没有可用的正样本，则返回None
        if pos_sampling_num == 0:
            return None

        # 采样正样本
        selected_pos = random.sample(same_author_pos, pos_sampling_num)
        packing.extend(selected_pos)

        # 如果跟踪已使用样本，将选择的正样本标记为已使用
        if used_pos_samples is not None:
            for p in selected_pos:
                used_pos_samples.add(p)

        # Level 2: 同作者不同标签负样本
        if len(packing) < self.packing_size:
            neg_labels = [label_item for label_item in self.author_label_map[author] if label_item != label]
            same_author_neg_pools = []
            for neg_label in neg_labels:
                same_author_neg_pools.extend(self.author_label_map[author][neg_label])

            same_author_neg_count = min(
                len(same_author_neg_pools), self.packing_size - len(packing)
            )
            if same_author_neg_count > 0:
                same_author_neg = random.sample(
                    same_author_neg_pools, same_author_neg_count
                )
                packing.extend(same_author_neg)

        # Level 3: 跨作者不同标签负样本
        if len(packing) < self.packing_size:
            # 使用集合来快速检查已添加的发布物
            existing_pubs = set(packing)
            remaining_neg = self.packing_size - len(packing)
            cross_author_neg = []

            # 随机选择足够的全局负样本
            candidates = random.sample(
                self.global_pub_pools,
                min(len(self.global_pub_pools), remaining_neg * 2),
            )
            for candidate in candidates:
                if (
                    candidate[0] not in existing_pubs
                    and len(cross_author_neg) < remaining_neg
                ):
                    cross_author_neg.append(candidate[0])
                    existing_pubs.add(candidate[0])

            packing.extend(cross_author_neg)

        return packing[: self.packing_size]

    def sampling(self):
        """返回构建的数据集"""
        return self.data

    def reset(self, shuffle=None):
        """重置数据集，可选是否更改shuffle设置"""
        if shuffle is not None:
            self.shuffle = shuffle
        self.build_dataset()
