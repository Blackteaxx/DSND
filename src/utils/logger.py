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

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
