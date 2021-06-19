from transformers import BertModel, BertTokenizer
from data import *
import torch
import torch.nn as nn

# import files for customized NEZHA model
from NEZHA.model_nezha import BertConfig, BertForSequenceClassification, NEZHAModel
from NEZHA import nezha_utils

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class SBERTSingleModelBase(nn.Module):
    def __init__(self, bert_dir, hidden_size=768, mid_size=512, freeze = False):
        super(SBERTSingleModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = 20
        
        self.bert = BertModel.from_pretrained(bert_dir)
        
        self.dropout = nn.Dropout(0.5)
        # self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm1d(mid_size)
        
        # self.linear_a = nn.Linear(hidden_size*3, mid_size)
        self.classifier_a = nn.Linear(hidden_size*3, 2)

        # self.linear_b = nn.Linear(hidden_size*3, mid_size)
        self.classifier_b = nn.Linear(hidden_size*3, 2)

    def forward(self, source_input_ids, target_input_ids):
        # 0 for [PAD], mask out the padded values
        source_attention_mask = torch.ne(source_input_ids, 0)
        target_attention_mask = torch.ne(target_input_ids, 0)
        
        # get bert output
        source_embedding = self.bert(source_input_ids, attention_mask=source_attention_mask)
        target_embedding = self.bert(target_input_ids, attention_mask=target_attention_mask)
        
        # simply take out the [CLS] represention
        # TODO: try different pooling strategies
        source_embedding = source_embedding[1]
        target_embedding = target_embedding[1]

        # concat the source embedding, target embedding and abs embedding as in the original SBERT paper
        abs_embedding = torch.abs(source_embedding-target_embedding)
        context_embedding = torch.cat([source_embedding, target_embedding, abs_embedding], -1)
        context_embedding = self.dropout(context_embedding)

        # get probs for type A
        # output_a = self.linear_a(context_embedding)
        # output_a = self.bn(output_a)
        # output_a = self.relu(output_a)
        # output_a = self.dropout(output_a)
        probs_a = self.classifier_a(context_embedding)
        
        # get probs for type B
        # output_b = self.linear_b(context_embedding)
        # output_b = self.bn(output_b)
        # output_b = self.relu(output_b)
        # output_b = self.dropout(output_b)
        probs_b = self.classifier_b(context_embedding)

        return probs_a, probs_b

class SBERTSingleModel(nn.Module):
    def __init__(self, bert_dir, hidden_size=768, mid_size=512, freeze = False):
        super(SBERTSingleModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = 2
        
        self.bert = BertModel.from_pretrained(bert_dir)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(mid_size)
        
        self.linear_a = nn.Linear(hidden_size*3, mid_size)
        self.classifier_a = nn.Linear(mid_size, 2)

        self.linear_b = nn.Linear(hidden_size*3, mid_size)
        self.classifier_b = nn.Linear(mid_size, 2)

    def forward(self, source_input_ids, target_input_ids):
        # 0 for [PAD], mask out the padded values
        source_attention_mask = torch.ne(source_input_ids, 0)
        target_attention_mask = torch.ne(target_input_ids, 0)
        
        # get bert output
        source_embedding = self.bert(source_input_ids, attention_mask=source_attention_mask)
        target_embedding = self.bert(target_input_ids, attention_mask=target_attention_mask)
        
        # simply take out the [CLS] represention
        # TODO: try different pooling strategies
        source_embedding = source_embedding[1]
        target_embedding = target_embedding[1]

        # concat the source embedding, target embedding and abs embedding as in the original SBERT paper
        abs_embedding = torch.abs(source_embedding-target_embedding)
        context_embedding = torch.cat([source_embedding, target_embedding, abs_embedding], -1)
        context_embedding = self.dropout(context_embedding)

        # get probs for type A
        output_a = self.linear_a(context_embedding)
        output_a = self.bn(output_a)
        output_a = self.relu(output_a)
        output_a = self.dropout(output_a)
        probs_a = self.classifier_a(output_a)
        
        # get probs for type B
        output_b = self.linear_b(context_embedding)
        output_b = self.bn(output_b)
        output_b = self.relu(output_b)
        output_b = self.dropout(output_b)
        probs_b = self.classifier_b(output_b)

        return probs_a, probs_b

class SBERTDoubleModel(nn.Module):
    def __init__(self, bert_dir, hidden_size=768, mid_size=512, freeze = False):
        super(SBERTDoubleModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = 2
        
        self.source_bert = BertModel.from_pretrained(bert_dir)
        self.target_bert = BertModel.from_pretrained(bert_dir)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(mid_size)
        
        self.linear_a = nn.Linear(hidden_size*3, mid_size)
        self.classifier_a = nn.Linear(mid_size, 2)

        self.linear_b = nn.Linear(hidden_size*3, mid_size)
        self.classifier_b = nn.Linear(mid_size, 2)

    def forward(self, source_input_ids, target_input_ids):
        # 0 for [PAD], mask out the padded values
        source_attention_mask = torch.ne(source_input_ids, 0)
        target_attention_mask = torch.ne(target_input_ids, 0)
        
        # get bert output
        source_embedding = self.source_bert(source_input_ids, attention_mask=source_attention_mask)
        target_embedding = self.target_bert(target_input_ids, attention_mask=target_attention_mask)
        
        # simply take out the [CLS] represention
        # TODO: try different pooling strategies
        source_embedding = source_embedding[1]
        target_embedding = target_embedding[1]

        # concat the source embedding, target embedding and abs embedding as in the original SBERT paper
        abs_embedding = torch.abs(source_embedding-target_embedding)
        context_embedding = torch.cat([source_embedding, target_embedding, abs_embedding], -1)
        context_embedding = self.dropout(context_embedding)

        # get probs for type A
        output_a = self.linear_a(context_embedding)
        output_a = self.bn(output_a)
        output_a = self.relu(output_a)
        output_a = self.dropout(output_a)
        probs_a = self.classifier_a(output_a)
        
        # get probs for type B
        output_b = self.linear_b(context_embedding)
        output_b = self.bn(output_b)
        output_b = self.relu(output_b)
        output_b = self.dropout(output_b)
        probs_b = self.classifier_b(output_b)

        return probs_a, probs_b


class BertClassifierSingleModel(nn.Module):
    def __init__(self, bert_dir, hidden_size = 768, mid_size=512, freeze = False):
        super(BertClassifierSingleModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = 2
        
        self.bert = BertModel.from_pretrained(bert_dir)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(mid_size)

        self.linear_a = nn.Linear(hidden_size, mid_size)
        self.classifier_a = nn.Linear(mid_size, 2)

        self.linear_b = nn.Linear(hidden_size, mid_size)
        self.classifier_b = nn.Linear(mid_size, 2)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, input_ids, input_types):
        # get shared BERT model output
        mask = torch.ne(input_ids, 0)
        bert_output = self.bert(input_ids, token_type_ids=input_types, attention_mask=mask)
        cls_embed = bert_output[1]
        output = self.dropout(cls_embed)

        # get probs for type A
        output_a = self.linear_a(output)
        output_a = self.bn(output_a)
        output_a = self.relu(output_a)
        output_a = self.dropout(output_a)
        probs_a = self.classifier_a(output_a)

        # get probs for type B
        output_b = self.linear_b(output)
        output_b = self.bn(output_b)
        output_b = self.relu(output_b)
        output_b = self.dropout(output_b)
        probs_b = self.classifier_b(output_b)

        return probs_a, probs_b

class BertClassifierTextCNNSingleModel(nn.Module):
    def __init__(self, bert_dir, hidden_size = 768, mid_size=512, freeze = False):
        super(BertClassifierTextCNNSingleModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = 2
        
        self.bert = BertModel.from_pretrained(bert_dir)
        
        # for TextCNN
        filter_num = 128
        filter_sizes = [2,3,4]
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (size, hidden_size)) for size in filter_sizes])

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(mid_size)

        self.linear_a = nn.Linear(len(filter_sizes) * filter_num, mid_size)
        self.classifier_a = nn.Linear(mid_size, 2)

        self.linear_b = nn.Linear(len(filter_sizes) * filter_num, mid_size)
        self.classifier_b = nn.Linear(mid_size, 2)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, input_ids, input_types):
        # get shared BERT model output
        mask = torch.ne(input_ids, 0)
        bert_output = self.bert(input_ids, token_type_ids=input_types, attention_mask=mask)
        bert_hidden = bert_output[0]
        output = self.dropout(bert_hidden)

        tcnn_input = output.unsqueeze(1)
        tcnn_output = [F.relu(conv(tcnn_input)).squeeze(3) for conv in self.convs]
        # max pooling in TextCNN
        # TODO: support avg pooling
        tcnn_output = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in tcnn_output]
        tcnn_output = torch.cat(tcnn_output, 1)
        tcnn_output = self.dropout(tcnn_output)

        # get probs for type A
        output_a = self.linear_a(tcnn_output)
        output_a = self.bn(output_a)
        output_a = self.relu(output_a)
        output_a = self.dropout(output_a)
        probs_a = self.classifier_a(output_a)

        # get probs for type B
        output_b = self.linear_b(tcnn_output)
        output_b = self.bn(output_b)
        output_b = self.relu(output_b)
        output_b = self.dropout(output_b)
        probs_b = self.classifier_b(output_b)

        return probs_a, probs_b

class NezhaClassifierSingleModel(nn.Module):
    def __init__(self, bert_dir, hidden_size = 768, mid_size=512, freeze = False):
        super(NezhaClassifierSingleModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = 2
        
        # self.bert = BertModel.from_pretrained(bert_dir)
        self.bert_config = BertConfig.from_json_file(bert_dir+'config.json')
        self.bert = NEZHAModel(config=self.bert_config)
        nezha_utils.torch_init_model(self.bert, bert_dir+'pytorch_model.bin')

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(mid_size)

        self.linear_a = nn.Linear(hidden_size, mid_size)
        self.classifier_a = nn.Linear(mid_size, 2)

        self.linear_b = nn.Linear(hidden_size, mid_size)
        self.classifier_b = nn.Linear(mid_size, 2)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, input_ids, input_types):
        # get shared BERT model output
        mask = torch.ne(input_ids, 0)
        bert_output = self.bert(input_ids, token_type_ids=input_types, attention_mask=mask)
        cls_embed = bert_output[1]
        output = self.dropout(cls_embed)

        # get probs for type A
        output_a = self.linear_a(output)
        output_a = self.bn(output_a)
        output_a = self.relu(output_a)
        output_a = self.dropout(output_a)
        probs_a = self.classifier_a(output_a)

        # get probs for type B
        output_b = self.linear_b(output)
        output_b = self.bn(output_b)
        output_b = self.relu(output_b)
        output_b = self.dropout(output_b)
        probs_b = self.classifier_b(output_b)

        return probs_a, probs_b


if __name__  == '__main__':
    train_data_dir = ['../data/sohu2021_open_data/短短匹配A类/valid.txt', '../data/sohu2021_open_data/短短匹配B类/valid.txt']
    test_data_dir = ['../data/sohu2021_open_data/短短匹配A类/test_with_id.txt']
    
    # pretrained = '/data1/wangchenyue/Downloads/bert-base-chinese/'
    pretrained = '/data1/wangchenyue/Downloads/nezha-base-wwm/'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    print("loading bert tokenizer successfully")
    
    print("loading data...")
    train_dataset = SentencePairDatasetWithType(train_data_dir, True, pretrained)
    # train_dataset = SentencePairDatasetForSBERT(train_data_dir, True, pretrained)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # # 0505: testing NEZHA model
    print("loading pretrained model...")
    # bert_model = BertClassifierSingleModel(pretrained)
    # bert_model = NezhaClassifierSingleModel(pretrained)
    bert_model = BertClassifierTextCNNSingleModel(pretrained)
    bert_model.to('cuda')

    for idx, batch in enumerate(train_dataloader):
        print('\n', idx)
        input_ids, input_types, labels, types = batch
        input_ids = input_ids.to('cuda')
        input_types = input_types.to('cuda')
        # labels should be flattened
        labels = labels.to('cuda').view(-1)

        probs_a, probs_b = bert_model(input_ids, input_types)
        print("probs_a.shape: ", probs_a.shape)
        print("probs_b.shape: ", probs_b.shape)
        print("labels.shape: ", labels.shape)
        print("types: ", types)

        mask_a = (types==0).numpy()
        mask_b = (types==1).numpy()
        print("mask_a: ", mask_a)
        print("mask_b: ", mask_b)

        output_a, labels_a = probs_a[mask_a], labels[mask_a]
        output_b, labels_b = probs_b[mask_b], labels[mask_b]
        # print(output_a, labels_a)
        # print(output_b, labels_b)

        criterion = nn.CrossEntropyLoss()
        loss_a = criterion(output_a, labels_a)
        loss_b = criterion(output_b, labels_b)
        print(loss_a.item(), loss_b.item())

        (loss_a + loss_b).backward()


    # 0504 testing SBERTSingleModel
    # 0506 testing SBERTDoubleModel
    # print("loading pretrained model...")
    # bert_model = SBERTDoubleModel(pretrained)
    # bert_model = bert_model.to('cuda')
    # for idx, batch in enumerate(train_dataloader):
    #     print('\n', idx)
    #     source_input_ids, target_input_ids, labels, types = batch
    #     source_input_ids = source_input_ids.to('cuda')
    #     target_input_ids = target_input_ids.to('cuda')
    #     # labels should be flattened
    #     labels = labels.to('cuda').view(-1)

    #     probs_a, probs_b = bert_model(source_input_ids, target_input_ids)
    #     print("probs_a.shape: ", probs_a.shape)
    #     print("probs_b.shape: ", probs_b.shape)
    #     print("labels.shape: ", labels.shape)
    #     print("types: ", types)

    #     mask_a = (types==0).numpy()
    #     mask_b = (types==1).numpy()
    #     print("mask_a: ", mask_a)
    #     print("mask_b: ", mask_b)

    #     output_a, labels_a = probs_a[mask_a], labels[mask_a]
    #     output_b, labels_b = probs_b[mask_b], labels[mask_b]
    #     # print(output_a, labels_a)
    #     # print(output_b, labels_b)

    #     criterion = nn.CrossEntropyLoss()
    #     loss_a = criterion(output_a, labels_a)
    #     loss_b = criterion(output_b, labels_b)
    #     print(loss_a.item(), loss_b.item())

    #     (loss_a + loss_b).backward()
