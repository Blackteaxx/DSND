import math
import multiprocessing as mp
import queue
from multiprocessing import Queue
from typing import Any, Dict, List, Literal, Union

import torch
from tqdm import tqdm, trange

from .modeling import Qwen2ModelForSNDPubEmbedding
from .utils.logger import get_logger

logger = get_logger(__name__)


class InferenceAgent:
    def __init__(self, model: Qwen2ModelForSNDPubEmbedding):
        self.model = model
        self.target_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    def encode(self, sentences):
        """encode sentences into embeddings, with multi-accelerators support

        Args:
            sentences (_type_): _description_
            batch_size (_type_): _description_
        """
        if len(sentences) == 1 or len(self.target_devices) == 1:
            # single accelerator
            return self._encode_single_accelerator(sentences, self.target_devices[0])

        self.pool = self.start_multi_process_pool(
            process_target_func=InferenceAgent._encode_multi_process_worker,
        )
        return self.encode_multi_process(
            sentences,
            pool=self.pool,
        )

    # adapted from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L807
    def start_multi_process_pool(
        self,
        process_target_func: Any,
    ) -> Dict[Literal["input", "output", "processes"], Any]:
        """
        Starts a multi-process pool to process the encoding with several independent processes
        via :meth:`SentenceTransformer.encode_multi_process <sentence_transformers.SentenceTransformer.encode_multi_process>`.

        This method is recommended if you want to encode on multiple GPUs or CPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process
        and stop_multi_process_pool.

        Returns:
            Dict[str, Any]: A dictionary with the target processes, an input queue, and an output queue.
        """
        if self.model is None:
            raise ValueError("Model is not initialized.")

        logger.info(
            "Start multi-process pool on devices: {}".format(
                ", ".join(map(str, self.target_devices))
            )
        )

        self.model.to("cpu")
        self.model.share_memory()
        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for device_id in tqdm(self.target_devices, desc="initial target device"):
            p = ctx.Process(
                target=process_target_func,
                args=(device_id, self.model, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {"input": input_queue, "output": output_queue, "processes": processes}

        # adapted from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L877

    def encode_multi_process(
        self,
        sentences: List[str],
        pool: Dict[Literal["input", "output", "processes"], Any],
        **kwargs,
    ):
        chunk_size = math.ceil(len(sentences) / len(pool["processes"]))

        input_queue = pool["input"]
        last_chunk_id = 0
        chunk = []

        # split the sentences into chunks, for multi-devices processing
        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, chunk, kwargs])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, chunk, kwargs])
            last_chunk_id += 1

        output_queue = pool["output"]
        results_list = sorted(
            [output_queue.get() for _ in trange(last_chunk_id, desc="Chunks")],
            key=lambda x: x[0],
        )
        results = [res[1] for res in results_list]

        # concatenate the results from all the processes
        embeddings = [res["embeddings"] for res in results]
        pub_ids = [res["pub_ids"] for res in results]
        res_embeddings = torch.cat(embeddings, dim=0)
        res_pub_ids = []
        for pub_id in pub_ids:
            res_pub_ids.extend(pub_id)

        results = {
            "embeddings": res_embeddings,
            "pub_ids": res_pub_ids,
        }
        return results

    # adapted from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L976
    @staticmethod
    def _encode_multi_process_worker(
        target_device: str,
        model,
        input_queue: Queue,
        results_queue: Queue,
    ) -> None:
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                chunk_id, sentences, kwargs = input_queue.get()
                results = InferenceAgent._encode_single_accelerator(
                    model, sentences, target_device
                )
                results_queue.put([chunk_id, results])
            except queue.Empty:
                break

    @staticmethod
    @torch.no_grad()
    def _encode_single_accelerator(model, sentences, target_device):
        """encode sentences into embeddings, with single accelerator support

        Args:
            sentences (_type_): _description_
        """
        model.eval()
        model.to(target_device)
        results = {
            "embeddings": [],
            "pub_ids": [],
        }
        for sentence in tqdm(sentences):
            pub_ids = sentence["pub_ids"]
            for k, v in sentence.items():
                if isinstance(v, torch.Tensor):
                    sentence[k] = v.to(target_device)
            outputs = model(
                **sentence,
            )
            embeddings = outputs.embeddings.detach().cpu()
            results["embeddings"].append(embeddings)
            results["pub_ids"].extend(pub_ids)

        # stack the embeddings
        results["embeddings"] = torch.cat(results["embeddings"], dim=0)
        return results

    def _concatenate_results_from_multi_process(
        self, results_list: List[Union[torch.Tensor, Any]]
    ):
        """concatenate and return the results from all the processes

        Args:
            results_list (List[Union[torch.Tensor, np.ndarray, Any]]): A list of results from all the processes.

        Raises:
            NotImplementedError: Unsupported type for results_list

        Returns:
            Union[torch.Tensor, np.ndarray]: return the embedding vectors in a numpy array or tensor.
        """
        if isinstance(results_list[0], torch.Tensor):
            # move all tensors to the same device
            results_list = [res.to(self.target_devices[0]) for res in results_list]
            return torch.cat(results_list, dim=0)
        else:
            raise NotImplementedError("Unsupported type for results_list")
