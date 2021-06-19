import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer
from utils import *
from sklearn.model_selection import train_test_split
import json

from tqdm import tqdm

class SentencePairDataset(Dataset):
    def __init__(self, file_dir, is_train, pretrained, shuffle_order=False, aug_data=False, len_limit=512, clip='tail'):
        self.is_train = is_train
        self.shuffle_order = shuffle_order
        self.aug_data = aug_data
        self.total_input_ids = []
        self.total_input_types = []

        # use AutoTokenzier instead of BertTokenizer to support speice.model (AlbertTokenizer-like)
        tokenizer = AutoTokenizer.from_pretrained(pretrained)

        # read json lines and convert to dict / df
        json_lines = []
        for single_file_dir in file_dir:
            with open(single_file_dir, 'r', encoding='utf-8') as f_in:
                json_lines += [line.strip() for line in f_in.readlines()]
        lines = [json.loads(line) for line in json_lines]

        content = pd.DataFrame(lines)
        content.columns = ['source', 'target', 'label']

        # utilize labelB=1-->A positive, labelA=0-->B negative 
        if self.is_train and self.aug_data:
            content = augment_data(content)

        sources = content['source'].values.tolist()
        targets = content['target'].values.tolist()

        if self.is_train:
            self.labels = content['label'].values.tolist()
        else:
            self.ids = content['label'].values.tolist()

        # shuffle_order is only allowed for training mode
        if self.shuffle_order and self.is_train:
            sources += content['target'].values.tolist()
            targets += content['source'].values.tolist()
            self.labels += self.labels

        len_limit_s = (len_limit-3)//2
        len_limit_t = (len_limit-3)-len_limit_s
        # print('len_limit_s: ', len_limit_s)
        # print('len_limit_t: ', len_limit_t)
        for source, target in tqdm(zip(sources, targets), total=len(sources)):
            # tokenize before clipping
            source = tokenizer.encode(source)[1:-1]
            target = tokenizer.encode(target)[1:-1]

            # clip the sentences if too long
            # TODO: different strategies to clip long sequences
            if clip == 'head' and len(source)+len(target)+3 > len_limit:
                if len(source)>len_limit_s and len(target)>len_limit_t:
                    source = source[0:len_limit_s]
                    target = target[0:len_limit_t]
                elif len(source)>len_limit_s:
                    source = source[0:len_limit-3-len(target)]
                elif len(target)>len_limit_t:
                    target = target[0:len_limit-3-len(source)]

            if clip == 'tail' and len(source)+len(target)+3 > len_limit:
                if len(source)>len_limit_s and len(target)>len_limit_t:
                    source = source[-len_limit_s:]
                    target = target[-len_limit_t:]
                elif len(source)>len_limit_s:
                    source = source[-(len_limit-3-len(target)):]
                elif len(target)>len_limit_t:
                    target = target[-(len_limit-3-len(source)):]

            assert len(source)+len(target)+3 <= len_limit
            
            # [CLS]:101, [SEP]:102
            input_ids = [101] + source + [102] + target + [102]
            input_types = [0]*(len(source)+2) + [1]*(len(target)+1)

            assert len(input_ids) <= len_limit and len(input_types) <= len_limit
            self.total_input_ids.append(input_ids)
            self.total_input_types.append(input_types)
    
        self.max_input_len = max([len(s) for s in self.total_input_ids])
        print("max length: ", self.max_input_len)

    def __len__(self):
        return len(self.total_input_ids)


    def __getitem__(self, idx):
        if self.is_train:
            input_ids = pad_to_maxlen(self.total_input_ids[idx], self.max_input_len)
            input_types = pad_to_maxlen(self.total_input_types[idx], self.max_input_len)
            label = int(self.labels[idx])
            # print(len(input_ids), len(input_types), label)
            return torch.LongTensor(input_ids), torch.LongTensor(input_types), torch.LongTensor([label])
            
        else:
            input_ids = pad_to_maxlen(self.total_input_ids[idx], self.max_input_len)
            input_types = pad_to_maxlen(self.total_input_types[idx], self.max_input_len)
            index  = self.ids[idx]
            return torch.LongTensor(input_ids), torch.LongTensor(input_types), index


class SentencePairDatasetForSBERT(Dataset):
    def __init__(self, file_dir, is_train, pretrained, shuffle_order=False, aug_data=False, len_limit=512, clip='head'):
        self.is_train = is_train
        self.shuffle_order = shuffle_order
        self.aug_data = aug_data
        self.total_source_input_ids = []
        # token_types are no longer neccessary if not concat into one text
        # self.total_source_input_types = []
        self.total_target_input_ids = []
        # self.total_target_input_types = []
        self.sample_types = []

        # use AutoTokenzier instead of BertTokenizer to support speice.model (AlbertTokenizer-like)
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        lines = []
        for single_file_dir in file_dir:
            with open(single_file_dir, 'r', encoding='utf-8') as f_in:
                content = f_in.readlines()
                for item in content:
                    line = json.loads(item.strip())
                    # BUG FIXED, order MATTERS!
                    # mannually add key 'type' to distinguish the origin of samples
                    # 0 for A, 1 for B
                    if 'A' in single_file_dir:
                        if self.is_train:
                            line['label'] = line.pop('labelA')
                        line['type'] = 0
                    else:
                        if self.is_train:
                            line['label'] = line.pop('labelB')
                        line['type'] = 1
                    lines.append(line)

        content = pd.DataFrame(lines)
        # print(content.head())
        content.columns = ['source', 'target', 'label', 'type']

        # utilize labelB=1-->A positive, labelA=0-->B negative 
        if self.is_train and self.aug_data:
            content = augment_data(content)

        sources = content['source'].values.tolist()
        targets = content['target'].values.tolist()

        self.sample_types = content['type'].values.tolist()
        if self.is_train:
            self.labels = content['label'].values.tolist()
        else:
            self.ids = content['label'].values.tolist()

        # shuffle_order is only allowed for training mode
        if self.shuffle_order and self.is_train:
            sources += content['target'].values.tolist()
            targets += content['source'].values.tolist()
            self.labels += self.labels
            self.sample_types += self.sample_types

        for source, target in tqdm(zip(sources, targets), total=len(sources)):
            # tokenize before clipping
            source = tokenizer.encode(source)[1:-1]
            target = tokenizer.encode(target)[1:-1]

            # clip the sentences if too long
            # TODO: different strategies to clip long sequences
            if clip == 'head':
                if len(source)+2 > len_limit:
                    source = source[0: len_limit-2]
                if len(target)+2 > len_limit:
                    target = target[0: len_limit-2]

            if clip == 'tail':
                if len(source)+2 > len_limit:
                    source = source[-len_limit+2:]
                if len(target)+2 > len_limit:
                    target = target[-len_limit+2:]

            # check if the length is within the limit
            assert len(source)+2 <= len_limit and len(target)+2 <= len_limit
            
            # [CLS]:101, [SEP]:102
            source_input_ids = [101] + source + [102]
            # source_input_types = [0]*(len(source)+2)

            target_input_ids = [101] + target + [102]
            # target_input_types = [0]*(len(target)+2)

            assert len(source_input_ids) <= len_limit and len(target_input_ids) <= len_limit
            # assert len(source_input_types) <= len_limit and len(target_input_types) <= len_limit
            
            self.total_source_input_ids.append(source_input_ids)
            # self.total_source_input_types.append(source_input_types)
            self.total_target_input_ids.append(target_input_ids)
            # self.total_target_input_types.append(target_input_types)
    
        self.max_source_input_len = max([len(s) for s in self.total_source_input_ids])
        self.max_target_input_len = max([len(s) for s in self.total_target_input_ids])
        print("max source length: ", self.max_source_input_len)
        print("max target length: ", self.max_target_input_len)

    def __len__(self):
        return len(self.total_target_input_ids)

    def __getitem__(self, idx):
        source_input_ids = pad_to_maxlen(self.total_source_input_ids[idx], self.max_source_input_len)
        # source_input_types = pad_to_maxlen(self.total_source_input_types[idx], self.max_source_input_len)
        target_input_ids = pad_to_maxlen(self.total_target_input_ids[idx], self.max_target_input_len)
        # target_input_types = pad_to_maxlen(self.total_target_input_types[idx], self.max_target_input_len)
        sample_type = int(self.sample_types[idx])

        if self.is_train:
            label = int(self.labels[idx])
            return torch.LongTensor(source_input_ids), torch.LongTensor(target_input_ids), torch.LongTensor([label]), sample_type
        
        else:
            index = self.ids[idx]
            return torch.LongTensor(source_input_ids), torch.LongTensor(target_input_ids), index, sample_type 


class SentencePairDatasetWithType(Dataset):
    def __init__(self, file_dir, is_train, pretrained, shuffle_order=False, aug_data=False, len_limit=512, clip='head'):
        self.is_train = is_train
        self.shuffle_order = shuffle_order
        self.aug_data = aug_data
        self.total_input_ids = []
        self.total_input_types = []
        self.sample_types = []

        # use AutoTokenzier instead of BertTokenizer to support speice.model (AlbertTokenizer-like)
        tokenizer = AutoTokenizer.from_pretrained(pretrained)

        # read json lines and convert to dict / df
        lines = []
        for single_file_dir in file_dir:
            with open(single_file_dir, 'r', encoding='utf-8') as f_in:
                content = f_in.readlines()
                for item in content:
                    line = json.loads(item.strip())
                    # BUG FIXED, order MATTERS!
                    # mannually add key 'type' to distinguish the origin of samples
                    # 0 for A, 1 for B
                    if 'A' in single_file_dir:
                        if self.is_train:
                            line['label'] = line.pop('labelA')
                        line['type'] = 0
                    else:
                        if self.is_train:
                            line['label'] = line.pop('labelB')
                        line['type'] = 1
                    lines.append(line)
            print(single_file_dir, len(lines))
        content = pd.DataFrame(lines)
        # print(content.head())
        content.columns = ['source', 'target', 'label', 'type']

        # utilize labelB=1-->A positive, labelA=0-->B negative 
        if self.is_train and self.aug_data:
            print("augmenting data...")
            content = augment_data(content)

        sources = content['source'].values.tolist()
        targets = content['target'].values.tolist()

        self.sample_types = content['type'].values.tolist()
        if self.is_train:
            self.labels = content['label'].values.tolist()
        else:
            self.ids = content['label'].values.tolist()

        # shuffle_order is only allowed for training mode
        if self.shuffle_order and self.is_train:
            sources += content['target'].values.tolist()
            targets += content['source'].values.tolist()
            self.labels += self.labels
            self.sample_types += self.sample_types
            
        len_limit_s = (len_limit-3)//2
        len_limit_t = (len_limit-3)-len_limit_s
        # print('len_limit_s: ', len_limit_s)
        # print('len_limit_t: ', len_limit_t)
        for source, target in tqdm(zip(sources, targets), total=len(sources)):
            # tokenize before clipping
            source = tokenizer.encode(source)[1:-1]
            target = tokenizer.encode(target)[1:-1]

            # clip the sentences if too long
            # TODO: different strategies to clip long sequences
            if clip == 'head' and len(source)+len(target)+3 > len_limit:
                if len(source)>len_limit_s and len(target)>len_limit_t:
                    source = source[0:len_limit_s]
                    target = target[0:len_limit_t]
                elif len(source)>len_limit_s:
                    source = source[0:len_limit-3-len(target)]
                elif len(target)>len_limit_t:
                    target = target[0:len_limit-3-len(source)]
            
            if clip == 'tail' and len(source)+len(target)+3 > len_limit:
                if len(source)>len_limit_s and len(target)>len_limit_t:
                    source = source[-len_limit_s:]
                    target = target[-len_limit_t:]
                elif len(source)>len_limit_s:
                    source = source[-(len_limit-3-len(target)):]
                elif len(target)>len_limit_t:
                    target = target[-(len_limit-3-len(source)):]

            # check if the total length is within the limit
            assert len(source)+len(target)+3 <= len_limit
            
            # [CLS]:101, [SEP]:102
            input_ids = [101] + source + [102] + target + [102]
            input_types = [0]*(len(source)+2) + [1]*(len(target)+1)

            assert len(input_ids) <= len_limit and len(input_types) <= len_limit
            self.total_input_ids.append(input_ids)
            self.total_input_types.append(input_types)
    
        self.max_input_len = max([len(s) for s in self.total_input_ids])
        print("max length: ", self.max_input_len)

    def __len__(self):
        return len(self.total_input_ids)

    def __getitem__(self, idx):
        if self.is_train:
            input_ids = pad_to_maxlen(self.total_input_ids[idx], self.max_input_len)
            input_types = pad_to_maxlen(self.total_input_types[idx], self.max_input_len)
            label = int(self.labels[idx])
            sample_type = int(self.sample_types[idx])
            # print(len(input_ids), len(input_types), label)
            return torch.LongTensor(input_ids), torch.LongTensor(input_types), torch.LongTensor([label]), sample_type
            
        else:
            input_ids = pad_to_maxlen(self.total_input_ids[idx], self.max_input_len)
            input_types = pad_to_maxlen(self.total_input_types[idx], self.max_input_len)
            index  = self.ids[idx]
            sample_type = int(self.sample_types[idx])
            return torch.LongTensor(input_ids), torch.LongTensor(input_types), index, sample_type


if __name__ == '__main__':
    data_dir = '../data/sohu2021_open_data/'
    task_a = ['短短匹配A类',  '短长匹配A类', '长长匹配A类']
    task_b = ['短短匹配B类',  '短长匹配B类', '长长匹配B类']

    train_data_dir = []
    for task in task_a:
        train_data_dir.append(data_dir + task + '/train.txt')
        train_data_dir.append(data_dir + task + '/train_r2.txt')
    print(train_data_dir)

    train_data_dir = ['../data/sohu2021_open_data/短短匹配A类/valid.txt', '../data/sohu2021_open_data/短短匹配B类/valid.txt']
    test_data_dir = ['../data/sohu2021_open_data/短短匹配A类/test_with_id.txt', '../data/sohu2021_open_data/短短匹配B类/test_with_id.txt']
    
    pretrained = '/data1/wangchenyue/Downloads/bert-base-chinese/'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    print("loading bert tokenizer successfully")
    
    print("Testing dataset for SentencePairDatasetWithType")
    train_dataset = SentencePairDatasetWithType(file_dir=train_data_dir, is_train=True, pretrained=pretrained, aug_data=False)
    print(len(train_dataset))
    for idx in range(10):
        print(train_dataset[idx][-2])

    test_dataset = SentencePairDatasetWithType(file_dir=test_data_dir, is_train=False, pretrained=pretrained)
    print(len(test_dataset))
    # for idx in range(1):
    #     print(test_dataset[idx])

    # print("Testing dataloader for SentencePairDatasetWithType")
    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # for idx, batch in enumerate(train_dataloader):
    #     input_ids, input_types, labels, types = batch

    # total_ids_a, total_ids_b = [], []
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # for idx, batch in enumerate(test_dataloader):
    #     input_ids, input_types, ids, types = batch
    #     mask_a, mask_b = (types==0).numpy(), (types==1).numpy()

    #     total_ids_a += [id for id in ids if id.endswith('a')]
    #     total_ids_b += [id for id in ids if id.endswith('b')]

    # 0504, SentencePairDatasetForSBERT
    # print("Testing dataset for SentencePairDatasetForSBERT")
    # train_dataset = SentencePairDatasetForSBERT(file_dir=train_data_dir, is_train=True, pretrained=pretrained, shuffle_order=False)
    # for idx in range(1):
    #     print(train_dataset[idx])

    # test_dataset = SentencePairDatasetForSBERT(file_dir=test_data_dir, is_train=False, pretrained=pretrained)
    # for idx in range(1):
    #     print(test_dataset[idx])

    # print("Testing dataloader for SentencePairDatasetForSBERT")
    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # for idx, batch in enumerate(train_dataloader):
    #     source_input_ids, target_input_types, labels, types = batch

    # total_ids_a, total_ids_b = [], []
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # for idx, batch in enumerate(test_dataloader):
    #     source_input_ids, target_input_types, ids, types = batch
    #     mask_a, mask_b = (types==0).numpy(), (types==1).numpy()

    #     total_ids_a += [id for id in ids if id.endswith('a')]
    #     total_ids_b += [id for id in ids if id.endswith('b')]
    