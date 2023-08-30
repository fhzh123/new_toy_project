import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import numpy as np
from tqdm import tqdm
from time import time
# Import PyTorch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
# Custom Modules
from model.dataset import CustomDataset
from model.model import TransformerModel
from utils.tqdm import TqdmLoggingHandler, write_log
from utils.model_utils import return_model_name
from utils.train_utils import input_to_device

def testing(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    #===================================#
    #=============Data Load=============#
    #===================================#

    write_log(logger, "Load data...")

    total_src_list, total_trg_list = data_load(data_path=args.data_path, data_name=args.data_name)

    # tokenizer load
    src_tokenizer_name = return_model_name(args.src_tokenizer_type)
    src_tokenizer = AutoTokenizer.from_pretrained(src_tokenizer_name)
    src_vocab_num = src_tokenizer.vocab_size

    trg_tokenizer_name = return_model_name(args.trg_tokenizer_type)
    trg_tokenizer = AutoTokenizer.from_pretrained(trg_tokenizer_name)
    trg_vocab_num = trg_tokenizer.vocab_size

    test_data = CustomDataset(src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer, 
                              src_list=total_src_list['train'], trg_list=total_trg_list['train'], 
                              src_max_len=args.src_max_len, trg_max_len=args.trg_max_len)
    test_dataloader = DataLoader(test_data, drop_last=False, batch_size=args.batch_size, shuffle=False, 
                                 pin_memory=True, num_workers=args.num_workers)
    write_log(logger, f"Total number of trainingsets iterations - {len(test_data)}, {len(test_dataloader)}")

    # Model load
    model = TransformerModel(encoder_model_type=args.encoder_model_type, decoder_model_type=args.decoder_model_type,
                             src_vocab_num=src_vocab_num, trg_vocab_num=trg_vocab_num,
                             isPreTrain=args.isPreTrain, dropout=args.dropout)
    
    save_file_name = os.path.join(args.model_save_path, args.data_name, args.encoder_model_type,
                                  f'checkpoint_pca_{args.pca_reduction}_seed_{args.random_seed}.pth.tar')
    checkpoint = torch.load(save_file_name)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    for i, batch_iter in enumerate(tqdm(test_dataloader, bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

        # Input setting
        b_iter = input_to_device(batch_iter, device=device)
        src_sequence, src_att, _, _  = b_iter

        with torch.no_grad():

            # Encoding
            encoder_out = model.encode(src_input_ids=src_sequence, src_attention_mask=src_att)

            # PCA
            if args.pca_reduction:
                encoder_out = model.pca_reduction(encoder_hidden_states=encoder_out, encoder_attention_mask=src_att)
                src_att = None

            # Decoding
            decoding_dict = return_decoding_dict(args)
            predicted = model.generate(decoding_dict=decoding_dict, encoder_hidden_states=encoder_out, encoder_attention_mask=src_att)