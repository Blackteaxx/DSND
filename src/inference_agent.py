import os
from multiprocessing import Process, Queue
import multiprocessing as mp
from typing import List, Optional

import torch
from tqdm import tqdm

from .modeling import Qwen2ModelForSNDPubEmbedding
from .utils.logger import get_logger

logger = get_logger(__name__)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class InferenceAgent:
    def __init__(self, model: Qwen2ModelForSNDPubEmbedding, num_gpus: int = 1):
        self.model = model
        self.num_gpus = num_gpus

    def _worker(
        self, gpu_id: int, task_queue: Queue, result_queue: Queue, progress_queue: Queue
    ):
        """worder function for each GPU"""
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        model = self.model.to(device)
        model.eval()
        while not task_queue.empty():
            try:
                batch = task_queue.get()
                pub_ids = batch["pub_ids"]
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                with torch.no_grad():
                    outputs = model(**batch)
                    sentence_embeddings = outputs.sentence_embeddings

                result = {
                    "embeddings": sentence_embeddings.cpu(),
                    "pub_ids": pub_ids,
                }
                result_queue.put(result)  # 将结果放回结果队列
                progress_queue.put(1)  # 更新进度
            except Exception as e:
                print(f"Error in GPU {gpu_id}: {e}")
                result_queue.put(None)

    def inference(
        self, sentences_dict: List[dict], batch_size: Optional[int] = 1
    ) -> List[float]:
        if batch_size is None:
            batch_size = len(sentences_dict)  # 默认 batch_size 为全部句子

        logger.info(
            f"Start inferencing with batch size {batch_size} on {self.num_gpus} GPUs"
        )
        logger.info(
            f"Total packings: {len(sentences_dict)}, each packing has {len(sentences_dict[0])} sentences"
        )

        ctx = mp.get_context("spawn")
        # 创建任务队列、结果队列和进度队列
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        progress_queue = ctx.Queue()
        # 将数据批次放入任务队列
        for i in range(0, len(sentences_dict), batch_size):
            batch = sentences_dict[i : i + batch_size]  # batch has been tokenized
            task_queue.put(batch)
        # 创建多个进程，每个进程负责一个 GPU
        processes = []
        for gpu_id in range(self.num_gpus):
            p = Process(
                target=self._worker,
                args=(gpu_id, task_queue, result_queue, progress_queue),
            )
            p.start()
            processes.append(p)
        # 使用 tqdm 显示进度条
        total_tasks = task_queue.qsize()  # 总任务数
        with tqdm(total=total_tasks, desc="Inference Progress") as pbar:
            while any(p.is_alive() for p in processes):
                while not progress_queue.empty():
                    progress_queue.get()  # 从进度队列中获取进度更新
                    pbar.update(1)  # 更新进度条
        # 等待所有进程完成
        for p in processes:
            p.join()
        # 从结果队列中收集所有结果
        results = []
        while not result_queue.empty():
            result = result_queue.get()
            if result is not None:
                results.append(result)
        return results
