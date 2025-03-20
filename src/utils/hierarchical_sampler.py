import random
from collections import defaultdict

from tqdm import tqdm


class HierarchicalSampler:
    def __init__(
        self,
        names_pub: dict,
        author_pub_labels: dict,
        packing_size: int,
        positive_ratio: float,
    ):
        # Load the data
        self.pub_to_author = {}
        self.labels = {}
        for author, pubs in author_pub_labels.items():
            for pub, label in pubs.items():
                self.pub_to_author[pub] = author
                self.labels[pub] = label

        # 新增分层索引结构
        self.author_label_map = defaultdict(lambda: defaultdict(list))
        self.label_author_map = defaultdict(lambda: defaultdict(list))

        for author, pubs in author_pub_labels.items():
            for pub, label in pubs.items():
                # 构建作者->标签->论文的三级索引
                self.author_label_map[author][label].append(pub)
                # 构建标签->作者->论文的三级索引
                self.label_author_map[label][author].append(pub)

        self.global_pub_pools = []
        for author, pubs in author_pub_labels.items():
            for pub, label in pubs.items():
                self.global_pub_pools.append((pub, author, label))

        self.data = []
        # construct the data, containing query, pos and neg samples
        for pub, label in tqdm(
            self.labels.items(), desc="Constructing the triplet dataset"
        ):
            self.data.append(
                self._get_triplet(pub, label, packing_size, positive_ratio)
            )

    def sampling(self):
        return self.data

    def _get_triplet(self, pub, label, packing_size, pos_ratio):
        author = self.pub_to_author[pub]

        # 分层采样策略
        packing = [pub]  # 第一个位置固定为query

        # Level 1: 同作者同标签正样本
        same_author_pos = [p for p in self.author_label_map[author][label] if p != pub]
        selected_pos = random.sample(
            same_author_pos, min(len(same_author_pos), int(pos_ratio * packing_size))
        )
        packing.extend(selected_pos)

        # Level 2: 同作者不同标签负样本
        if len(packing) < packing_size:
            neg_labels = [l for l in self.author_label_map[author] if l != label]
            same_author_neg_pools = []
            for l in neg_labels:
                same_author_neg_pools.extend(self.author_label_map[author][l])
            same_author_neg = random.sample(
                same_author_neg_pools,
                min(len(same_author_neg_pools), packing_size - len(packing)),
            )
            packing.extend(same_author_neg)

        # Level 3: 跨作者不同标签负样本
        if len(packing) < packing_size:
            remaining_neg = packing_size - len(packing)
            cross_author_neg = []
            while remaining_neg > 0:
                candidate = random.choice(self.global_pub_pools)
                if candidate[0] not in packing:
                    cross_author_neg.append(candidate[0])
                    remaining_neg -= 1
            packing.extend(cross_author_neg)

        # 保证最终长度
        return packing[:packing_size]
