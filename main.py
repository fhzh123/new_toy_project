# Import modules
import os
import time
import argparse
# Import custom modules
from task.training import training
# from task.testing import testing
# Utils
from utils.argparser_utils import str2bool, path_check, set_random_seed

def main(args):

    # Time setting
    total_start_time = time.time()

    # Seed setting
    set_random_seed(args.random_seed)

    # Path setting
    path_check(args)

    if args.training:
        training(args)

    # Time calculate
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')

if __name__=='__main__':
    user_name = os.getlogin()
    parser = argparse.ArgumentParser(description='Parsing Method')
    # Task setting
    parser.add_argument('--data_name', default='WMT2016', type=str,
                        help='')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--resume', action='store_true')
    # Path setting
    parser.add_argument('--data_path', default='/HDD/dataset', type=str,
                        help='Original data path')
    parser.add_argument('--preprocess_path', default=f'/HDD/{user_name}/preprocessed/pca', type=str,
                        help='Pre-processing data path')
    parser.add_argument('--model_save_path', default=f'/HDD/{user_name}/model_checkpoint/pca', type=str,
                        help='Model checkpoint file path')
    parser.add_argument('--result_path', default=f'/HDD/{user_name}/results/pca', type=str,
                        help='Results file path')
    # Preprocessing setting
    parser.add_argument('--src_max_len', default=150, type=str,
                        help='Source input maximum length; Default is 150')
    parser.add_argument('--trg_max_len', default=150, type=str,
                        help='Target input maximum length; Default is 150')
    # Model setting
    parser.add_argument('--pca_reduction', action='store_true')
    parser.add_argument('--src_tokenizer_type', default='T5', type=str,
                        help='Source input tokenizer setting; Default is T5')
    parser.add_argument('--trg_tokenizer_type', default='T5', type=str,
                        help='Target input tokenizer setting; Default is T5')
    parser.add_argument('--encoder_model_type', default='T5', type=str,
                        help='Encoder model setting; Default is T5')
    parser.add_argument('--decoder_model_type', default='T5', type=str,
                        help='Decoder model setting; Default is T5')
    parser.add_argument('--isPreTrain', default=True, type=str2bool,
                        help='Use pre-trained language model; Default is True')
    parser.add_argument('--dropout', default=0.2, type=float,
                        help='Decoder model setting; Default is T5')
    # Train setting
    parser.add_argument('--random_seed', default=42, type=int,
                        help='Random seed setting; Default is 42')
    parser.add_argument('--num_epochs', default=200, type=int,
                        help='Number of epochs; Default is 200')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Training batch size; Default is 16')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Num of CPU workers; Default is 8')
    parser.add_argument('--optimizer', default='Ralamb', type=str,
                        help='Gradient descent optimizer setting; Default is Ralamb')
    parser.add_argument('--scheduler', default='warmup', type=str,
                        help='Gradient descent scheduler setting; Default is warmup')
    parser.add_argument('--n_warmup_epochs', default=2, type=float,
                        help='Wamrup epochs when using warmup scheduler; Default is 2')
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Learning rate setting; Default is 8')
    parser.add_argument('--w_decay', default=1e-5, type=float,
                        help="Ralamb's weight decay; Default is 1e-5")
    parser.add_argument('--label_smoothing_eps', default=0.01, type=float,
                        help='Label smoothing epsilon; Default is 0.01')
    parser.add_argument('--clip_grad_norm', default=5, type=int,
                        help='Gradient clipping norm setting; Default is 5')
    # Checking setting
    parser.add_argument('--debugging_mode', action='store_true',
                        help='')
    parser.add_argument('--print_freq', default=300, type=int,
                        help='Training process printing frequency; Default is 300')
    args = parser.parse_args()

    main(args)