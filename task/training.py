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
from utils.optimizer_utils import optimizer_select, scheduler_select

def training(args):

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

    total_src_list, total_trg_list = dict(), dict()

    data_path = os.path.join(args.data_path,'WMT/2016/multi_modal')

    # 1) Train data load
    with open(os.path.join(data_path, 'train.de'), 'r') as f:
        total_src_list['train'] = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(data_path, 'train.en'), 'r') as f:
        total_trg_list['train'] = [x.replace('\n', '') for x in f.readlines()]

    # 2) Valid data load
    with open(os.path.join(data_path, 'val.de'), 'r') as f:
        total_src_list['valid'] = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(data_path, 'val.en'), 'r') as f:
        total_trg_list['valid'] = [x.replace('\n', '') for x in f.readlines()]

    # 3) Test data load
    with open(os.path.join(data_path, 'test.de'), 'r') as f:
        total_src_list['test'] = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(data_path, 'test.en'), 'r') as f:
        total_trg_list['test'] = [x.replace('\n', '') for x in f.readlines()]

    # tokenizer load
    src_tokenizer_name = return_model_name(args.src_tokenizer_type)
    src_tokenizer = AutoTokenizer.from_pretrained(src_tokenizer_name)
    src_vocab_num = src_tokenizer.vocab_size

    trg_tokenizer_name = return_model_name(args.trg_tokenizer_type)
    trg_tokenizer = AutoTokenizer.from_pretrained(trg_tokenizer_name)
    trg_vocab_num = trg_tokenizer.vocab_size

    dataset_dict = {
        'train': CustomDataset(src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer,
                               src_list=total_src_list['train'], trg_list=total_trg_list['train'], 
                               src_max_len=args.src_max_len, trg_max_len=args.trg_max_len),
        'valid': CustomDataset(src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer,
                               src_list=total_src_list['train'], trg_list=total_trg_list['train'], 
                               src_max_len=args.src_max_len, trg_max_len=args.trg_max_len),
        'test': CustomDataset(src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer,
                               src_list=total_src_list['train'], trg_list=total_trg_list['train'], 
                               src_max_len=args.src_max_len, trg_max_len=args.trg_max_len),
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True,
                            pin_memory=True, num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=True, 
                            pin_memory=True, num_workers=args.num_workers),
        'test': DataLoader(dataset_dict['test'], drop_last=False,
                           batch_size=args.batch_size, shuffle=False, 
                           pin_memory=True, num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    model = TransformerModel(encoder_model_type=args.encoder_model_type, decoder_model_type=args.decoder_model_type,
                             src_vocab_num=src_vocab_num, trg_vocab_num=trg_vocab_num,
                             isPreTrain=args.isPreTrain, dropout=args.dropout)
    model.to(device)

    # 3) Optimizer & Learning rate scheduler setting
    optimizer = optimizer_select(optimizer_model=args.optimizer, model=model, lr=args.lr, w_decay=args.w_decay)
    scheduler = scheduler_select(scheduler_model=args.scheduler, optimizer=optimizer, dataloader_len=len(dataloader_dict['train']), args=args)

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps, ignore_index=model.pad_idx).to(device)

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        write_log(logger, 'Resume model...')
        save_path = os.path.join(args.model_save_path, args.data_name)
        save_file_name = os.path.join(save_path,
                                        f'checkpoint_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}_v_{args.variational_mode}_p_{args.parallel}.pth.tar')
        checkpoint = torch.load(save_file_name)
        start_epoch = checkpoint['epoch'] - 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    write_log(logger, 'Traing start!')
    best_val_loss = 1e+7

    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        start_time_e = time()

        write_log(logger, 'Training start...')
        model.train()

        for i, batch_iter in enumerate(tqdm(dataloader_dict['train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

            optimizer.zero_grad(set_to_none=True)

            # Input setting
            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, trg_sequence, trg_att  = b_iter

            # Encoding
            encoder_out = model.encode(src_input_ids=src_sequence, src_attention_mask=src_att)

            # PCA
            if args.pca_reduction:
                encoder_out = model.pca_reduction(encoder_hidden_states=encoder_out, encoder_attention_mask=src_att)
                src_att = None

            # Decoding
            decoder_out = model.decode(trg_input_ids=trg_sequence, encoder_hidden_states=encoder_out,
                                       encoder_attention_mask=src_att)

            # Loss Backward
            decoder_out_view = decoder_out.view(-1, trg_vocab_num)
            trg_sequence_view = trg_sequence.view(-1)

            train_loss = criterion(decoder_out_view, trg_sequence_view)
            train_loss.backward()
            if args.clip_grad_norm > 0:
                clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            scheduler.step()

            # Print loss value only training
            if i == 0 or i % args.print_freq == 0 or i == len(dataloader_dict['train'])-1:
                train_acc = (decoder_out_view.argmax(dim=1)[trg_sequence_view != 0] == trg_sequence_view[trg_sequence_view != 0]).sum() / (trg_sequence_view != 0).sum()
                iter_log = "[Epoch:%03d][%03d/%03d] train_loss:%03.2f | train_accuracy:%03.2f | learning_rate:%1.6f |spend_time:%02.2fmin" % \
                    (epoch, i, len(dataloader_dict['train'])-1, train_loss.item(), train_acc.item(), optimizer.param_groups[0]['lr'], (time() - start_time_e) / 60)
                write_log(logger, iter_log)

            if args.debugging_mode:
                break

        write_log(logger, 'Validation start...')
        model.eval()
        val_loss = 0
        val_acc = 0

        for batch_iter in tqdm(dataloader_dict['valid'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):

            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, trg_sequence, trg_att  = b_iter

            with torch.no_grad():

                # Encoding
                encoder_out = model.encode(src_input_ids=src_sequence, src_attention_mask=src_att)

                # PCA
                if args.pca_reduction:
                    encoder_out = model.pca_reduction(encoder_hidden_states=encoder_out, encoder_attention_mask=src_att)
                    src_att = None

                # Decoding
                decoder_out = model.decode(trg_input_ids=trg_sequence, encoder_hidden_states=encoder_out,
                                        encoder_attention_mask=src_att)

            # Loss and Accuracy Check
            decoder_out_view = decoder_out.view(-1, trg_vocab_num)
            trg_sequence_view = trg_sequence.view(-1)

            val_acc += (decoder_out_view.argmax(dim=1)[trg_sequence_view != 0] == trg_sequence_view[trg_sequence_view != 0]).sum() / (trg_sequence_view != 0).sum()
            val_loss += criterion(decoder_out_view, trg_sequence_view)

            if args.debugging_mode:
                break

        # val_mmd_loss /= len(dataloader_dict['valid'])
        val_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])
        write_log(logger, 'Augmenter Classifier Validation CrossEntropy Loss: %3.3f' % val_loss)
        write_log(logger, 'Augmenter Classifier Validation Accuracy: %3.2f%%' % (val_acc * 100))

        save_file_name = os.path.join(args.model_save_path, args.data_name, args.encoder_model_type, f'checkpoint_pca_{args.pca_reduction}_seed_{args.random_seed}.pth.tar')
        if val_loss < best_val_loss:
            write_log(logger, 'Model checkpoint saving...')
            torch.save({
                'cls_training_done': False,
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, save_file_name)
            best_val_loss = val_loss
            best_epoch = epoch
        else:
            else_log = f'Still {best_epoch} epoch Loss({round(best_val_loss.item(), 4)}) is better...'
            write_log(logger, else_log)