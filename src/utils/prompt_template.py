from ..arguments import SpecialToken


def format_paper_for_bert(paper_dict: dict, author_name: str, use_graph: bool = False):
    """将论文字典转换为LLM友好的结构化文本

    Args:
        paper_dict: 包含论文元数据的字典
        author_name: 主要作者姓名
        use_graph: 是否使用图数据（默认为False）

    Returns:
        str: 适合LLM处理的自然语言格式文本
    """
    components = []

    # 改进的Prompt引导语（保持原有变量位置）
    prompt = """Task Description: Generate a discriminative embedding vector for author name disambiguation by analyzing:
        Paper Metadata:
        {}"""

    # 完全保留您原有的内容生成逻辑
    if use_graph:
        components.append(f"\nGraph Information: {SpecialToken.GRAPH_TOKEN} ")
    components.append(f"Research Paper Title: {paper_dict['title']}")
    components.append("\nMain Author: " + author_name)
    components.append("\nAuthors:")
    components.extend([f"- {author['name']}" for author in paper_dict["authors"]])
    components.append("\nKey Terms:")
    components.append("; ".join([f"[{kw}]" for kw in paper_dict["keywords"]]))

    metadata = []
    if "venue" in paper_dict:
        metadata.append(f"Published in: {paper_dict['venue']}")
    if "year" in paper_dict:
        metadata.append(f"Year: {paper_dict['year']}")
    if metadata:
        components.append("\n" + " | ".join(metadata))

    pub_info = "\n".join(components)
    return prompt.format(pub_info)


def format_paper_for_llm(
    paper_dict: dict, author_name: str, use_graph: bool = False
) -> str:
    """将论文字典转换为LLM友好的结构化文本

    Args:
        paper_dict: 包含论文元数据的字典
        author_name: 主要作者姓名
        use_graph: 是否使用图数据（默认为False）

    Returns:
        str: 适合LLM处理的自然语言格式文本
    """
    components = []

    # 改进的Prompt引导语（保持原有变量位置）
    prompt = """Task Description: Generate a discriminative embedding vector for author name disambiguation by analyzing:
        1. Author identity clues (institutions, co-authors)
        2. Technical content fingerprints (methods, keywords)
        3. Temporal-spatial patterns (year, venue)
        4. High-level graph structual features (if applicable)
            
        Output Requirements:
        - Encode features that distinguish papers from different authors with the same name
        - Preserve all original field labels (e.g., "Research Paper Title:")
        - Focus on quantifiable characteristics
            
        Paper Metadata:
        {}"""

    # 完全保留您原有的内容生成逻辑
    if use_graph:
        components.append(f"\nGraph Information: {SpecialToken.GRAPH_TOKEN} ")
    components.append(f"Research Paper Title: {paper_dict['title']}")
    components.append("\nMain Author: " + author_name)
    components.append("\nAuthors:")
    components.extend(
        [
            f"- {author['name']} ({author['org'].rstrip(', ')})"
            for author in paper_dict["authors"]
            # f"- {author['name']}"
            # for author in paper_dict["authors"]
        ]
    )
    # components.append("\nAbstract:")
    # components.append(paper_dict["abstract"].replace("(Turcz.)", ""))
    components.append("\nKey Terms:")
    components.append("; ".join([f"[{kw}]" for kw in paper_dict["keywords"]]))

    metadata = []
    if "venue" in paper_dict and paper_dict["venue"] != "":
        metadata.append(f"Published in: {paper_dict['venue']}")
    if "year" in paper_dict and paper_dict["year"] != "":
        metadata.append(f"Year: {paper_dict['year']}")
    if metadata:
        components.append("\n" + " | ".join(metadata))

    pub_info = "\n".join(components)
    return prompt.format(pub_info)
