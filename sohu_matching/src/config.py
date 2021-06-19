class Config():
    def __init__(self):
        # 指定device，一直都是'cuda'
        self.device= 'cuda'
        # 模型名称，主要用于区分模型
        # 如果使用nezha，请在模型名称中包含‘nezha’
        # 如果加入TextCNN，请在模型名称中包含‘cnn’
        self.model_type = '0503_macbert_base'
        # 任务类型，'a'仅加载A类任务，'b'仅加载B类任务，'ab'同时加载
        self.task_type = 'ab'

        # 保存模型的文件夹，注意最后的/不能少
        # self.save_dir = '../checkpoints/0503/'
        self.save_dir = '/data1/wangchenyue/sohu_matching/checkpoints/0503/'

        # 保存数据的文件夹，注意最后的/不能少
        # self.data_dir  = '../data/sohu2021_open_data/'
        self.data_dir = '/data1/wangchenyue/sohu_matching/data/sohu2021_open_data/'
        # 是否仅加载部分数据集用于调试
        self.load_toy_dataset = False

        # 预训练模型文件夹
        self.pretrained = '/data1/wangchenyue/Downloads/chinese-macbert-base'
        # 训练轮数，基本上2个epoch就收敛，随后开始过拟合
        self.epochs = 3
        # 学习率，BERT基本是2e-5
        self.lr = 2e-5
        # AdamW优化器中的权重衰减系数
        self.weight_decay = 1e-3
        # 是否使用学习率规划器，目前实现为get_linear_schedule_with_warmup
        self.use_scheduler = True
        # 学习率规划器中warmup的步数，以线性逐步增加到指定学习率
        self.num_warmup_steps = 2000
        
        # BERT模型的输出维度，base规模的为768
        self.hidden_size = 768
        # 训练过程中的batch_size，对于SBERT基本需要减半
        self.train_bs = 32
        # 评估过程中的batch_size，对于SBERT基本需要减半
        self.eval_bs = 64
        # 损失函数，'CE'为交叉熵，'FL'为FocalLoss
        self.criterion = 'CE'
        # 每隔多少step输出训练信息
        self.print_every = 50
        # 每隔多少step在评估集上评估
        self.eval_every = 200

        # 是否交换source与target位置，作为一种数据增强
        self.shuffle_order = False        
        # 是否将B类任务的正样例加入A类任务正样、A类负样例加入B类负样例，作为一种数据增强
        self.aug_data = False
        # 截断方式，'head'保留前几句话，'tail'保留最后几句话
        self.clip_method = 'head'

        # 是否使用fgm对抗训练
        self.use_fgm = False

        ### 推理相关设定 ###
        # 推理时加载的模型所在文件夹
        # self.infer_model_dir = '../checkpoints/0502/'
        self.infer_model_dir = '/data1/wangchenyue/sohu_matching/checkpoints/0503/'
        # 推理时加载的模型名，模型的保存名最后包含了任务类型和保存指标
        self.infer_model_name = '0503_roberta_original_epoch_1_ab_loss'
        # 推理时对应的任务类型
        self.infer_task_type = self.infer_model_name.split('_')[-2]
        # 推理时的文件输出地址
        # self.infer_output_dir = '../results/0502/'
        self.infer_output_dir = '/data1/wangchenyue/sohu_matching/results/0503/'
        self.infer_output_filename = '{}.csv'.format(self.infer_model_name)
        # 推理时的截断方式
        self.infer_clip_method = 'head'
        # 推理时的batch_size
        self.infer_bs = 256
        # 推理时对两类任务的指定阈值，结果保存在以fixed_开头的文件中
        self.infer_fixed_thres_a = 0.45
        self.infer_fixed_thres_b = 0.35
        # 推理时是否优化阈值，False则只以指定阈值推理
        self.infer_search_thres = True

if __name__ == '__main__':
    config = Config()
    print(config)