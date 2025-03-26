import logging

import torch


def distributed_logging(logger, *args):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            for arg in args:
                logger.info(arg)
    else:
        for arg in args:
            logger.info(arg)


def get_logger(name, log_file="/data/name_disambiguation/train.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file is provided)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
