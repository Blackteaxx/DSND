def format_paper_for_llm(paper_dict: dict, author_name: str):
    """将论文字典转换为LLM友好的结构化文本

    Args:
        paper_dict: 包含论文元数据的字典

    Returns:
        str: 适合LLM处理的自然语言格式文本
    """
    components = []

    prompt = "Given the following related infomation of the research paper: {}. Use one word to describe the research paper to separate it from others."

    # 1. 标题与作者信息强化
    components.append(f"Research Paper Title: {paper_dict['title']}")
    components.append("\nMain Author: " + author_name)
    components.append("\nAuthors:")
    components.extend(
        [
            f"- {author['name']} ({author['org'].rstrip(', ')})"
            for author in paper_dict["authors"]
        ]
    )

    # 2. 摘要结构化处理
    components.append("\nAbstract:")
    components.append(paper_dict["abstract"].replace("(Turcz.)", ""))  # 清理特殊符号

    # 3. 关键词增强表示
    components.append("\nKey Terms:")
    components.append("; ".join([f"[{kw}]" for kw in paper_dict["keywords"]]))

    # 4. 元数据整合
    metadata = []
    if "venue" in paper_dict:
        metadata.append(f"Published in: {paper_dict['venue']}")
    if "year" in paper_dict:
        metadata.append(f"Year: {paper_dict['year']}")
    if metadata:
        components.append("\n" + " | ".join(metadata))

    pub_info = "\n".join(components)
    return prompt.format(pub_info)
