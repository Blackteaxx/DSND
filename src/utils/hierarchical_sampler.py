import random
from collections import defaultdict

from tqdm import tqdm


class HierarchicalSampler:
    def __init__(
        self,
        names_pub: dict,
        author_pub_labels: dict,
        packing_size: int,
        positive_ratio: float = None,
        positive_num: int = None,
        random_seed: int = None,
    ):
        assert positive_ratio is not None or positive_num is not None, (
            "positive_ratio and positive_num cannot be both None"
        )
        assert positive_ratio is None or positive_num is None, (
            "positive_ratio and positive_num cannot be both provided"
        )
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

        # Set the random seed, if provided, to ensure reproducibility
        if random_seed is not None:
            random.seed(random_seed)

        self.data = []
        # construct the data, containing query, pos and neg samples
        for pub, label in tqdm(
            self.labels.items(), desc="Constructing the triplet dataset"
        ):
            self.data.append(
                self._get_triplet(
                    pub, label, packing_size, positive_ratio, positive_num
                )
            )

    def sampling(self):
        return self.data

    def _get_triplet(self, pub, label, packing_size, pos_ratio, pos_num):
        author = self.pub_to_author[pub]

        # 分层采样策略
        packing = [pub]  # 第一个位置固定为query

        # Level 1: 同作者同标签正样本
        same_author_pos = [p for p in self.author_label_map[author][label] if p != pub]
        pos_sampling_num = (
            pos_num if pos_num is not None else int(len(same_author_pos) * pos_ratio)
        )
        pos_sampling_num = min(len(same_author_pos), pos_sampling_num)
        selected_pos = random.sample(same_author_pos, pos_sampling_num)
        packing.extend(selected_pos)

        # Level 2: 同作者不同标签负样本
        if len(packing) < packing_size:
            neg_labels = [l for l in self.author_label_map[author] if l != label]
            same_author_neg_pools = []
            for neg_label in neg_labels:
                same_author_neg_pools.extend(self.author_label_map[author][neg_label])
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
