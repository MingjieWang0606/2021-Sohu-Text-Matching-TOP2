import numpy as np
from sklearn.metrics import f1_score
# from collections import defaultdict

if __name__ == '__main__':
    # 我们对以下五组模型的输出结果进行了投票集成并提交测试集榜单
    # 本文件中，我们同样将各组模型的验证集结果进行投票，计算各类f1
    model_names = ['0505_nezha_test_epoch_1_ab_loss',
                   '0505_nezha_ce_epoch_1_ab_loss',
                   '0503_macbert_base_epoch_1_ab_loss',
                   '0503_roberta_60K_singlemodel_epoch_1_ab_loss',
                   '0503_sbert_roberta_epoch_1_ab_loss']
    
    # len of valid_a: 4971, len of valid_b: 4969
    total_preds_a, total_preds_b = [0]*4971, [0]*4969
    # positive if the vote exceeds the threshold (>)
    threshold = len(model_names)/2
    for model in model_names:
        print("processing model {}".format(model))
        preds_a, preds_b = np.load('{}_pred_a.npy'.format(model)), np.load('{}_pred_b.npy'.format(model))
        preds_a, preds_b = preds_a.tolist(), preds_b.tolist()
        assert len(total_preds_a)==len(preds_a) and len(total_preds_b)==len(preds_b)
        for idx, pred_a in enumerate(preds_a):
            total_preds_a[idx] += pred_a
        for idx, pred_b in enumerate(preds_b):
            total_preds_b[idx] += pred_b
        # print(len(preds_a), len(preds_b))
        # print(type(preds_b))

    total_preds_a, total_preds_b = np.array(total_preds_a), np.array(total_preds_b)
    vote_a, vote_b = (total_preds_a>threshold).astype('int'), (total_preds_b>threshold).astype('int')
    gt_a, gt_b = np.load('gt_a.npy'), np.load('gt_b.npy')
    # print(len(vote_a), len(vote_b))
    # print(len(gt_a), len(gt_b))

    f1a, f1b = f1_score(gt_a, vote_a), f1_score(gt_b, vote_b)
    ssa, ssb = f1_score(gt_a[:1645], vote_a[:1645]), f1_score(gt_b[:1643], vote_b[:1643])
    sla, slb = f1_score(gt_a[1645:3301], vote_a[1645:3301]), f1_score(gt_b[1643:3299], vote_b[1643:3299])
    lla, llb = f1_score(gt_a[3301:], vote_a[3301:]), f1_score(gt_b[3299:], vote_b[3299:])
    print("f1a: {}, f1b: {}".format(f1a, f1b))
    print("ssa: {}, ssb: {}".format(ssa, ssb))
    print("sla: {}, slb: {}".format(sla, slb))
    print("lla: {}, llb: {}".format(lla, llb))