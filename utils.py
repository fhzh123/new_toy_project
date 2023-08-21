# Import modules
import os
import sys
import time
import tqdm
import random
import logging
import argparse
import numpy as np
# Import PyTorch
import torch
import torch.nn.functional as F

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def path_check(args):
    # Preprocessing Path Checking
    #(args.preprocess_path, args.task, args.data_name, args.tokenizer)
    if not os.path.exists(args.preprocess_path):
        os.makedirs(args.preprocess_path)

    if not os.path.exists(os.path.join(args.preprocess_path, args.data_name)):
        os.makedirs(os.path.join(args.preprocess_path, args.data_name))

    if not os.path.exists(os.path.join(args.preprocess_path, args.data_name, args.encoder_model_type)):
        os.makedirs(os.path.join(args.preprocess_path, args.data_name, args.encoder_model_type))

    # Model Checkpoint Path Checking
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    if not os.path.exists(os.path.join(args.model_save_path, args.data_name)):
        os.makedirs(os.path.join(args.model_save_path, args.data_name))

    if not os.path.exists(os.path.join(args.model_save_path, args.data_name, args.encoder_model_type)):
        os.makedirs(os.path.join(args.model_save_path, args.data_name, args.encoder_model_type))

    # Testing Results Path Checking
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

def input_to_device(batch_iter, device):

    src_sequence = batch_iter[0]
    src_att = batch_iter[1]
    trg_sequence = batch_iter[2]
    trg_att = batch_iter[3]

    src_sequence = src_sequence.to(device, non_blocking=True)
    src_att = src_att.to(device, non_blocking=True)
    trg_sequence = trg_sequence.to(device, non_blocking=True)
    trg_att = trg_att.to(device, non_blocking=True)

    return src_sequence, src_att, trg_sequence, trg_att

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.stream = sys.stdout

    def flush(self):
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, "flush"):
                self.stream.flush()
        finally:
            self.release()

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit, RecursionError):
            raise
        except Exception:
            self.handleError(record)

def write_log(logger, message):
    if logger:
        logger.info(message)