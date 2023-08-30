import os

def data_load(data_path:str = None, data_name:str = None):

    total_src_list, total_trg_list = dict(), dict()

    if data_name == 'WMT2016_Multimodal':

        data_path = os.path.join(data_path,'WMT/2016/multi_modal')

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

    return total_src_list, total_trg_list