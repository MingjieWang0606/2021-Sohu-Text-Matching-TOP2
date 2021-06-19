# from model import BertClassifier
from model import BertClassifierSingleModel, NezhaClassifierSingleModel, BertClassifierTextCNNSingleModel
from data import SentencePairDatasetWithType
from utils import pad_to_maxlen
from config import Config
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

from sklearn import metrics
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

def infer(model, device, dev_dataloader, test_dataloader, search_thres=True, threshold_fixed_a=0.5, threshold_fixed_b=0.5, save_valid=True):
    print("Inferring")
    model.eval()
    model = torch.nn.DataParallel(model)

    total_gt_a, total_preds_a, total_probs_a =  [], [], []
    total_gt_b, total_preds_b, total_probs_b = [], [], []

    print("Model running on dev set...")
    for idx, batch in enumerate(tqdm(dev_dataloader)):
        input_ids, input_types, labels, types = batch
        input_ids = input_ids.to(device)
        input_types = input_types.to(device)
        # labels should be flattened
        labels = labels.to(device).view(-1)

        with torch.no_grad():
            probs_a, probs_b = model(input_ids, input_types)
            mask_a, mask_b = (types==0).numpy(), (types==1).numpy()
            outputs_a = nn.functional.softmax(probs_a[mask_a], dim=1).cpu().numpy().tolist()
            outputs_b = nn.functional.softmax(probs_b[mask_b], dim=1).cpu().numpy().tolist()
            labels_a, labels_b = labels[mask_a], labels[mask_b]

            gt_a = labels_a.cpu().numpy().tolist()
            total_gt_a += gt_a
            preds_a = probs_a[mask_a].argmax(axis=1).cpu().numpy().tolist() if len(gt_a)!=0 else []
            # preds_a = probs_a.argmax(axis=1).cpu().numpy().tolist() if len(gt_a)!=0 else []
            total_preds_a += preds_a
            total_probs_a += [output[-1] for output in outputs_a]

            gt_b = labels_b.cpu().numpy().tolist()
            total_gt_b += gt_b
            preds_b = probs_b[mask_b].argmax(axis=1).cpu().numpy().tolist() if len(gt_b)!=0 else []
            # preds_b = probs_b.argmax(axis=1).cpu().numpy().tolist() if len(gt_b)!=0 else []
            total_preds_b += preds_b
            total_probs_b += [output[-1] for output in outputs_b]
            

    if search_thres:
        # search for the optimal threshold
        print("Searching for the best threshold on valid dataset...")
        thresholds = np.arange(0.2, 0.9, 0.01)
        fscore_a = np.zeros(shape=(len(thresholds)))
        fscore_b = np.zeros(shape=(len(thresholds)))
        print('Length of sequence: {}'.format(len(thresholds)))
        
        print("Original F1 Score for Task A: {}".format(str(metrics.f1_score(total_gt_a, total_preds_a, zero_division=0))))
        if len(total_gt_a) != 0:
            print("\tClassification Report\n")
            print(metrics.classification_report(total_gt_a, total_preds_a))

        print("Original F1 Score for Task B: {}".format(str(metrics.f1_score(total_gt_b, total_preds_b, zero_division=0))))
        if len(total_gt_b) != 0:
            print("\tClassification Report\n")
            print(metrics.classification_report(total_gt_b, total_preds_b))
            
        for index, thres in enumerate(tqdm(thresholds)):
            y_pred_prob_a = (np.array(total_probs_a) > thres).astype('int')
            fscore_a[index] = metrics.f1_score(total_gt_a, y_pred_prob_a.tolist(), zero_division=0)

            y_pred_prob_b = (np.array(total_probs_b) > thres).astype('int')
            fscore_b[index] = metrics.f1_score(total_gt_b, y_pred_prob_b.tolist(), zero_division=0)

        # record the optimal threshold for task A
        # print(fscore_a)
        index_a = np.argmax(fscore_a)
        threshold_opt_a = round(thresholds[index_a], ndigits=4)
        f1_score_opt_a = round(fscore_a[index_a], ndigits=6)
        print('Best Threshold for Task A: {} with F-Score: {}'.format(threshold_opt_a, f1_score_opt_a))
        # print("\nThreshold Classification Report\n")
        # print(metrics.classification_report(total_gt_a, (np.array(total_probs_a) > threshold_opt_a).astype('int').tolist()))

        # record the optimal threshold for task B
        index_b = np.argmax(fscore_b)
        threshold_opt_b = round(thresholds[index_b], ndigits=4)
        f1_score_opt_b = round(fscore_b[index_b], ndigits=6)
        print('Best Threshold for Task B: {} with F-Score: {}'.format(threshold_opt_b, f1_score_opt_b))
        # print("\nThreshold Classification Report\n")
        # print(metrics.classification_report(total_gt_b, (np.array(total_probs_b) > threshold_opt_b).astype('int').tolist()))

        if save_valid:
            y_pred_prob_a = (np.array(total_probs_a) > threshold_opt_a).astype('int')
            y_pred_prob_b = (np.array(total_probs_b) > threshold_opt_b).astype('int')
            ssa, sla, lla = y_pred_prob_a[0:1645], y_pred_prob_a[1645:3301], y_pred_prob_a[3301:]
            gt_ssa, gt_sla, gt_lla = total_gt_a[0:1645], total_gt_a[1645:3301], total_gt_a[3301:]
            ssb, slb, llb = y_pred_prob_b[0:1643], y_pred_prob_b[1643:3299], y_pred_prob_b[3299:]
            gt_ssb, gt_slb, gt_llb = total_gt_b[0:1643], total_gt_b[1643:3299], total_gt_b[3299:]
            print("f1 on ssa: ", metrics.f1_score(gt_ssa, ssa))
            print("f1 on sla: ", metrics.f1_score(gt_sla, sla))
            print("f1 on lla: ", metrics.f1_score(gt_lla, lla))
            print("f1 on ssb: ", metrics.f1_score(gt_ssb, ssb))
            print("f1 on slb: ", metrics.f1_score(gt_slb, slb))
            print("f1 on llb: ", metrics.f1_score(gt_llb, llb))
            
            np.save('../valid_output/pred_a.npy', y_pred_prob_a)
            np.save('../valid_output/pred_b.npy', y_pred_prob_b)
            np.save('../valid_output/gt_a.npy', np.array(total_gt_a))
            np.save('../valid_output/gt_b.npy', np.array(total_gt_b))

    total_ids_a, total_probs_a = [], []
    total_ids_b, total_probs_b = [], []
    for idx, batch in enumerate(tqdm(test_dataloader)):
        input_ids, input_types, ids, types = batch
        input_ids = input_ids.to(device)
        input_types = input_types.to(device)
        
        # the probs given by the model, without grads 
        with torch.no_grad():
            probs_a, probs_b = model(input_ids, input_types)
            mask_a, mask_b = (types==0).numpy(), (types==1).numpy()
            total_ids_a += [id for id in ids if id.endswith('a')]
            total_ids_b += [id for id in ids if id.endswith('b')]
            probs_a = nn.functional.softmax(probs_a[mask_a], dim=1).cpu().numpy().tolist()
            probs_b = nn.functional.softmax(probs_b[mask_b], dim=1).cpu().numpy().tolist()
            total_probs_a += [prob[-1] for prob in probs_a]
            total_probs_b += [prob[-1] for prob in probs_b]

    # positive if the prob passes the original threshold of 0.5
    total_fixed_preds_a = (np.array(total_probs_a) > threshold_fixed_a).astype('int').tolist()
    total_fixed_preds_b = (np.array(total_probs_b) > threshold_fixed_b).astype('int').tolist()
    
    if search_thres:
        # positive if the prob passes the optimal threshold
        total_preds_a = (np.array(total_probs_a) > threshold_opt_a).astype('int').tolist()
        total_preds_b = (np.array(total_probs_b) > threshold_opt_b).astype('int').tolist()
    else:
        total_preds_a = None
        total_preds_b = None

    return total_ids_a, total_preds_a, total_fixed_preds_a, \
           total_ids_b, total_preds_b, total_fixed_preds_b 


if __name__=='__main__':
    config = Config()
    device = config.device
    pretrained = config.pretrained
    model_type = config.infer_model_name

    save_dir = config.infer_model_dir
    model_name = config.infer_model_name
    hidden_size = config.hidden_size
    output_dir= config.infer_output_dir
    output_filename = config.infer_output_filename
    data_dir = config.data_dir
    task_a = ['短短匹配A类',  '短长匹配A类', '长长匹配A类']
    task_b = ['短短匹配B类',  '短长匹配B类', '长长匹配B类']
    task_type = config.infer_task_type

    infer_bs = config.infer_bs
    search_thres = config.infer_search_thres
    threshold_fixed_a = config.infer_fixed_thres_a
    threshold_fixed_b = config.infer_fixed_thres_b
    # method for clipping long seqeunces, 'head' or 'tail'
    clip_method = config.infer_clip_method
    
    dev_data_dir, test_data_dir = [], []
    if 'a' in task_type:
        for task in task_a:
            dev_data_dir.append(data_dir + task + '/valid.txt')
            test_data_dir.append(data_dir + task + '/test_with_id.txt')
    if 'b' in task_type:
        for task in task_b:
            dev_data_dir.append(data_dir + task + '/valid.txt')
            test_data_dir.append(data_dir + task + '/test_with_id.txt')

    print("Loading Bert Model from {}...".format(save_dir + model_name))
    # this may be problematic when loaded into different GPUs
    # model = torch.load(save_dir + model_name)

    # recommended: initialize the model first, then load the state_dict
    # distinguish model architectures or pretrained models according to model_type
    if 'nezha' in model_type.lower():
        print("Using NEZHA pretrained model")
        model = NezhaClassifierSingleModel(bert_dir=pretrained, hidden_size=hidden_size)
    elif 'cnn' in model_type.lower():
        print("Adding TextCNN after BERT output")
        model = BertClassifierTextCNNSingleModel(bert_dir=pretrained, hidden_size=hidden_size)
    else:
        print("Using conventional BERT model with linears")
        model = BertClassifierSingleModel(bert_dir=pretrained, hidden_size=hidden_size)
    # '.module' could raise weird behavior
    model_dict = torch.load(save_dir + model_name).module.state_dict()
    model.load_state_dict(model_dict)
    model.to(device)

    print("Loading Dev Data...")
    dev_dataset = SentencePairDatasetWithType(dev_data_dir, True, pretrained, clip=clip_method)
    dev_dataloader = DataLoader(dev_dataset, batch_size=infer_bs, shuffle=False)

    print("Loading Test Data...")
    # for test dataset, is_train should be set to False, thus get ids instead of labels
    test_dataset = SentencePairDatasetWithType(test_data_dir, False, pretrained, clip=clip_method)
    test_dataloader = DataLoader(test_dataset, batch_size=infer_bs, shuffle=False)

    total_ids_a, total_preds_a, total_fixed_preds_a, total_ids_b, total_preds_b, total_fixed_preds_b = infer(model, device, dev_dataloader, test_dataloader, search_thres, threshold_fixed_a, threshold_fixed_b)
    
    with open(output_dir + 'fixed_' + output_filename, 'w') as f_out:
        for id, pred in zip(total_ids_a, total_fixed_preds_a):   
            f_out.writelines(str(id) + ',' + str(pred) + '\n')
        for id, pred in zip(total_ids_b, total_fixed_preds_b):   
            f_out.writelines(str(id) + ',' + str(pred) + '\n')

    if total_preds_a is not None:
        with open(output_dir + output_filename, 'w') as f_out:
            for id, pred in zip(total_ids_a, total_preds_a):   
                f_out.writelines(str(id) + ',' + str(pred) + '\n')
            for id, pred in zip(total_ids_b, total_preds_b):   
                f_out.writelines(str(id) + ',' + str(pred) + '\n')