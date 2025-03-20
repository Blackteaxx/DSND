# 2025.3.17

import json
import os
import random
import re
from typing import Literal

from logger import get_logger
from match_name import match_name
from tqdm import tqdm

logger = get_logger(__name__)

puncs = "[!“”\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+"
stopwords = [
    "at",
    "based",
    "in",
    "of",
    "for",
    "on",
    "and",
    "to",
    "an",
    "using",
    "with",
    "the",
    "by",
    "we",
    "be",
    "is",
    "are",
    "can",
]
stopwords_extend = [
    "university",
    "univ",
    "china",
    "department",
    "dept",
    "laboratory",
    "lab",
    "school",
    "al",
    "et",
    "institute",
    "inst",
    "college",
    "chinese",
    "beijing",
    "journal",
    "science",
    "international",
    "key",
    "sciences",
    "research",
    "academy",
    "state",
    "center",
]
stopwords_check = [
    "a",
    "was",
    "were",
    "that",
    "2",
    "key",
    "1",
    "technology",
    "0",
    "sciences",
    "as",
    "from",
    "r",
    "3",
    "academy",
    "this",
    "nanjing",
    "shanghai",
    "state",
    "s",
    "research",
    "p",
    "results",
    "peoples",
    "4",
    "which",
    "5",
    "high",
    "materials",
    "study",
    "control",
    "method",
    "group",
    "c",
    "between",
    "or",
    "it",
    "than",
    "analysis",
    "system",
    "sci",
    "two",
    "6",
    "has",
    "h",
    "after",
    "different",
    "n",
    "national",
    "japan",
    "have",
    "cell",
    "time",
    "zhejiang",
    "used",
    "data",
    "these",
]


def split_train_dev(
    train_names_pub: dict,
    dev_ratio: float = 0.2,
    random_seed: int = 42,
):
    """Split training data into training and validation data

    Args:
        train_names_pub (dict): _description_
        dev_ratio (float, optional): _description_. Defaults to 0.2.
        random_seed (int, optional): _description_. Defaults to 42.
    """
    random.seed(random_seed)

    author_names = list(train_names_pub.keys())
    random.shuffle(author_names)

    dev_author_names = author_names[: int(len(author_names) * dev_ratio)]
    train_author_names = author_names[int(len(author_names) * dev_ratio) :]

    dev_names_pub = {
        author_name: train_names_pub[author_name] for author_name in dev_author_names
    }
    train_names_pub = {
        author_name: train_names_pub[author_name] for author_name in train_author_names
    }

    return train_names_pub, dev_names_pub


def process_raw_data(
    src_dir: str,
    dst_dir: str,
    mode: Literal["train", "valid", "test"],
) -> None:
    """Process raw data, whose format is
        author: {author_name: {author_id: [paper_id]}} in training data, {author_name: [paper_id]} in validation and testing data
        paper:  {paper_id: features_dict}
        To the format:
        names-pub/{mode}/{author_name}.json: {paper_id: features_dict} in training data, {paper_id: features_dict} in validation and testing data

    Args:
        src_dir (str): the directory of raw data
        dst_dir (str): the directory to save processed data
        mode (List[str]): the mode of data, including 'train', 'valid', 'test'
    """
    assert mode in ["train", "valid", "test"]

    # Load raw data
    if mode == "train":
        author2pub = os.path.join(src_dir, mode, "train_author.json")
        paper2feature = os.path.join(src_dir, mode, "train_pub.json")
    else:
        author2pub = os.path.join(src_dir, mode, f"sna_{mode}_raw.json")
        paper2feature = os.path.join(src_dir, mode, f"sna_{mode}_pub.json")

    with open(author2pub, "r") as f:
        author2pub = json.load(f)
    with open(paper2feature, "r") as f:
        paper2feature = json.load(f)

    # Process raw data
    names_pub = {}
    if mode == "train":
        for author_name, author_id2pub in tqdm(
            author2pub.items(), desc=f"Processing {mode} raw data"
        ):
            for author_id, paper_ids in author_id2pub.items():
                if author_name not in names_pub:
                    names_pub[author_name] = {}
                for paper_id in paper_ids:
                    names_pub[author_name][paper_id] = paper2feature[paper_id]
    else:
        for author_name, paper_ids in tqdm(author2pub.items()):
            if author_name not in names_pub:
                names_pub[author_name] = {}
            for paper_id in paper_ids:
                names_pub[author_name][paper_id] = paper2feature[paper_id]

    # Save processed data
    if mode == "train":
        train_names_pub, valid_names_pub = split_train_dev(names_pub)
        train_target_file_name = os.path.join(dst_dir, "names-pub", "train.json")
        dev_target_file_name = os.path.join(dst_dir, "names-pub", "dev.json")
        if os.path.exists(train_target_file_name):
            os.remove(train_target_file_name)
        if os.path.exists(dev_target_file_name):
            os.remove(dev_target_file_name)
        with open(train_target_file_name, "w") as f:
            json.dump(train_names_pub, f, indent=4, ensure_ascii=False)
        with open(dev_target_file_name, "w") as f:
            json.dump(valid_names_pub, f, indent=4, ensure_ascii=False)
    else:
        target_file_name = os.path.join(dst_dir, "names-pub", f"{mode}.json")
        if os.path.exists(target_file_name):
            os.remove(target_file_name)
        with open(target_file_name, "w") as f:
            json.dump(names_pub, f, indent=4, ensure_ascii=False)


def get_text_features(
    names_pub: dict,
    dst_dir: str,
    mode: Literal["train", "dev", "valid", "test"],
) -> None:
    """Get text features from processed data.
        Text feature: publiactions' title, abstract, keywords, venue, authors' name and organization, year
        Text feature will be saved in features/{mode}/{author_name}/text_features.json

    Args:
        names_pub (dict): processed data
        dst_dir (str): the directory to save text features
        mode (List[str]): the mode of data, including 'train', 'valid', 'test'
    """
    assert mode in ["train", "dev", "valid", "test"]

    # Get text features
    for author_name, papers in tqdm(
        names_pub.items(), desc=f"Getting text features in {mode} data"
    ):
        author_dir = os.path.join(dst_dir, "features", mode, author_name)
        if not os.path.exists(author_dir):
            os.makedirs(author_dir)

        text_features_dict = {}
        for paper_id, paper in papers.items():
            text_features_dict[paper_id] = paper

        target_file_name = os.path.join(author_dir, "text_features.json")
        if os.path.exists(target_file_name):
            os.remove(target_file_name)
        with open(target_file_name, "w") as f:
            json.dump(text_features_dict, f, indent=4, ensure_ascii=False)


def unify_name_order(name):
    """
    unifying different orders of name.
    Args:
        name
    Returns:
        name and reversed name
    """
    token = name.split("_")
    name = token[0] + token[1]
    name_reverse = token[1] + token[0]
    if len(token) > 2:
        name = token[0] + token[1] + token[2]
        name_reverse = token[2] + token[0] + token[1]

    return name, name_reverse


def get_relation_features(
    names_pub: dict,
    dst_dir: str,
    mode: Literal["train", "dev", "valid", "test"],
) -> None:
    """Get relation features from processed data.
        Relation feature: papers' co-authorship, papers' org, papers' venue
        Relation feature will be saved in features/{mode}/{author_name}/paper_author.csv, paper_org.csv, paper_venue.csv

    Args:
        names_pub (dict): processed data
        dst_dir (str): the directory to save relation features
        mode (List[str]): the mode of data, including 'train', 'valid', 'test'
    """
    assert mode in ["train", "dev", "valid", "test"]

    r = "[!“”\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+"  # remove the punctuations and special characters

    for n, name in tqdm(
        enumerate(names_pub), desc="Generating publication features and relations..."
    ):
        file_path = os.path.join(dst_dir, "features", mode, name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        authorname_dict = {}  # maintain a author-name-dict
        ori_name = name
        name, name_reverse = unify_name_order(name)
        pubs_dict = names_pub[ori_name]

        with (
            open(os.path.join(file_path, "paper_author.txt"), "w") as coa_file,
            open(os.path.join(file_path, "paper_title.txt"), "w") as cot_file,
            open(os.path.join(file_path, "paper_venue.txt"), "w") as cov_file,
            open(os.path.join(file_path, "paper_org.txt"), "w") as coo_file,
        ):
            for i, pid in enumerate(pubs_dict):
                pub = pubs_dict[pid]

                # Save tokenized title (relations)
                title = pub["title"]
                pstr = title.strip()
                pstr = pstr.lower()
                pstr = re.sub(r, " ", pstr)
                pstr = re.sub(r"\s{2,}", " ", pstr).strip()
                pstr = pstr.split(" ")
                pstr = [word for word in pstr if len(word) > 1]
                pstr = [word for word in pstr if word not in stopwords]
                pstr = [word for word in pstr if word not in stopwords_check]
                for word in pstr:
                    cot_file.write(pid + "\t" + word + "\n")

                # Save tokenized keywords
                word_list = []
                if "keywords" in pub:
                    for word in pub["keywords"]:
                        word_list.append(word)
                    pstr = " ".join(word_list)
                    pstr = re.sub(" +", " ", pstr)
                keyword = pstr

                # Save tokenized org (relations) and co-author (relations)
                org = ""
                find_author = False
                for author in pub["authors"]:
                    authorname = "".join(filter(str.isalpha, author["name"])).lower()
                    token = authorname.split(" ")
                    if len(token) == 2:
                        authorname = token[0] + token[1]
                        authorname_reverse = token[1] + token[0]
                        # Find whether the author's name is a name of co-author
                        if authorname not in authorname_dict:
                            if authorname_reverse not in authorname_dict:
                                authorname_dict[authorname] = 1
                            else:
                                authorname = authorname_reverse
                    else:
                        authorname = authorname.replace(" ", "")

                    if authorname != name and authorname != name_reverse:
                        coa_file.write(
                            pid + "\t" + authorname + "\n"
                        )  # current name is a name of co-author
                    else:
                        # find the author's org
                        if "org" in author:
                            org = author[
                                "org"
                            ]  # current name is a name for disambiguating
                            find_author = True

                # if the author's org is not found, then find the org by the author's name
                if not find_author:
                    for author in pub["authors"]:
                        if match_name(author["name"], ori_name):
                            org = author["org"]
                            break

                pstr = org.strip()
                pstr = pstr.lower()
                pstr = re.sub(puncs, " ", pstr)
                pstr = re.sub(r"\s{2,}", " ", pstr).strip()
                pstr = pstr.split(" ")
                pstr = [word for word in pstr if len(word) > 1]
                pstr = [word for word in pstr if word not in stopwords]
                pstr = [word for word in pstr if word not in stopwords_extend]
                pstr = set(pstr)
                for word in pstr:
                    coo_file.write(pid + "\t" + word + "\n")

                # Save tokenized venue (relations)
                if pub["venue"]:
                    pstr = pub["venue"].strip()
                    pstr = pstr.lower()
                    pstr = re.sub(puncs, " ", pstr)
                    pstr = re.sub(r"\s{2,}", " ", pstr).strip()
                    pstr = pstr.split(" ")
                    pstr = [word for word in pstr if len(word) > 1]
                    pstr = [word for word in pstr if word not in stopwords]
                    pstr = [word for word in pstr if word not in stopwords_extend]
                    pstr = [word for word in pstr if word not in stopwords_check]
                    for word in pstr:
                        cov_file.write(pid + "\t" + word + "\n")
                    if len(pstr) == 0:
                        cov_file.write(pid + "\t" + "null" + "\n")


def get_text_and_relation_features(
    src_dir: str,
    dst_dir: str,
    mode: Literal["train", "dev", "valid", "test"],
) -> None:
    """Get text and relation features from processed data.
        Text feature: publiactions' title, abstract, keywords, venue, authors' name and organization, year
        Relation feature: papers' co-authorship, papers' org, papers' venue

        Text feature will be saved in features/{mode}/{author_name}/text_features.json
        Relation feature will be saved in features/{mode}/{author_name}/paper_author.csv, paper_org.csv, paper_venue.csv

    Args:
        src_dir (str): the directory of processed data
        dst_dir (str): the directory to save text and relation features
        mode (List[str]): the mode of data, including 'train', 'valid', 'test'
    """
    assert mode in ["train", "dev", "valid", "test"]

    # Load processed data
    with open(os.path.join(src_dir, "names-pub", f"{mode}.json"), "r") as f:
        names_pub = json.load(f)

    # Get text and relation features
    get_text_features(names_pub, dst_dir, mode)
    get_relation_features(names_pub, dst_dir, mode)


def get_labels(
    src_dir: str,
    dst_dir: str,
):
    """
    Get labels from training data.

    Args:
        src_dir (str): the directory of training data
        dst_dir (str): the directory to save labels
    """
    with open(os.path.join(src_dir, "train", "train_author.json"), "r") as f:
        author2pub = json.load(f)

    labels = {}
    label_idx = 0
    for author_name, author_id2pub in author2pub.items():
        labels[author_name] = {}
        for author_id, paper_ids in author_id2pub.items():
            for paper_id in paper_ids:
                labels[author_name][paper_id] = label_idx
            label_idx += 1

    with open(os.path.join(dst_dir, "train_labels.json"), "w") as f:
        json.dump(labels, f, indent=4, ensure_ascii=False)

    logger.info(f"Labels are saved in {os.path.join(dst_dir, 'train_labels.json')}")


if __name__ == "__main__":
    src_dir = "/data/name_disambiguation/data/SND/src"
    dst_dir = "/data/name_disambiguation/data/SND"
    process_raw_data(src_dir, dst_dir, "train")
    process_raw_data(src_dir, dst_dir, "valid")
    process_raw_data(src_dir, dst_dir, "test")
    get_text_and_relation_features(dst_dir, dst_dir, "train")
    get_text_and_relation_features(dst_dir, dst_dir, "dev")
    get_text_and_relation_features(dst_dir, dst_dir, "valid")
    get_text_and_relation_features(dst_dir, dst_dir, "test")

    get_labels(src_dir, dst_dir)
