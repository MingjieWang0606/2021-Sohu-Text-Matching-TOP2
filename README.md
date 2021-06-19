### sohu_matching

#### 小组：**分比我们低的都是帅哥**

#### 简介

本项目包含了小组**分比我们低的都是帅哥**在2021搜狐校园文本匹配初赛环节的PyTorch版本代码，在初赛Public排行榜上排名第一，复赛排行榜第二，决赛排行榜第二。测评的F1分数为0.751548057294007，其中A类任务0.8032378580323787，B类任务0.6998582565556343。

我们采用了联合训练的方式，在A、B两个任务上采用一个共同的基于预训练语言模型的encoder，而后分别为两个任务采用两组简单的全连接结构作为classifier。我们使用了不同的预训练模型（如NEZHA、MacBert、ROBERTA、ERNIE等），设计了选择了两种文本匹配的技术路线（通过[SEP]拼接source与target作为输入、类似SBERT的句子向量编码进行比较），并尝试了多种上分策略（如在给定语料上继续mlm预训练、focal loss损失函数、不同的pooling策略、加入TextCNN、fgm对抗训练、数据增强等）。我们选取了多组差异较大的模型的输出，通过投票的方式进行集成，得到最好成绩。

在验证集上的各组F1值分别为：

|          |    A类     |    B类     |
| :------: | :--------: | :--------: |
| 短短匹配 | 0.79593909 | 0.74812968 |
| 短长匹配 | 0.79614767 | 0.65986395 |
| 长长匹配 | 0.84605598 | 0.74776786 |
|   全部   | 0.81672364 | 0.72042440 |

#### 项目结构

```bash
│  README.md				# README
│  test.yaml				# conda环境配置
│  							# 基本上安装pytorch>=1.6和transformer即可复现
├─checkpoints				# 用于保存模型
├─data						# 用于保存数据
│  └─sohu2021_open_data
│      ├─短短匹配A类			# 包括train.txt, train_r2.txt, train_r3.txt, 
│      ├─短短匹配B类			# valid.txt, test_with_id.txt
│      ├─短长匹配A类
│      ├─短长匹配B类
│      ├─长长匹配A类
│      └─长长匹配B类
├─logs						# 用于保存日志，例：python train.py > log_dir
│  └─0503
│          0503_roberta_60K_singlemodel.log
│          0503_roberta_original.log
│          
├─results					# 用于保存测试集推理结果
│  └─0503					# fixed_表示按指定阈值推理，其他为最优阈值推理
│          0503_roberta_original_epoch_1_ab_f1.csv
│          0503_roberta_original_epoch_1_ab_loss.csv
│          fixed_0503_roberta_original_epoch_1_ab_f1.csv
│          fixed_0503_roberta_original_epoch_1_ab_loss.csv
│ 
├─valid_output				# 记录模型在valid上的输出，并计算各类f1  
└─src						# 主要代码文件夹
    │  config.py			# 模型与训练等参数统一通过config.py设置
    │  data.py				# 数据读取，DataLoader等
    │  infer.py				# 测试集推理代码
    │  infer_sbert.py		# 测试集推理代码（SBERT）
    │  model.py				# 模型定义
    │  train.py				# 训练代码
    │  train_sbert.py		# 训练代码（SBERT）
    │  utils.py				# 其他
    │  
    ├─new_runs				# tensorboard事件目录，用于可视化损失函数等指标
    │  └─0503_roberta_original_ab
    │          events.out.tfevents.1620106925.ubuntu
    ├─NEZHA					# nezha相关的模型结构定义等
    │  │  model_nezha.py	
    │  │  nezha_utils.py     
    └─__pycache__
```

#### 运行示例

补充训练数据后，在`config.py`文件中设置训练相关参数，进入到src文件夹下，运行`train.py`进行训练（默认多卡训练，在`train.py`调整设备卡数），可通过重定向将输出保存为日志。训练结束后，在`config.py`中设置推理相关参数，进入到src文件夹下，运行`infer.py`进行推理（默认多卡推理，在`infer.py`调整设备卡数）。

```bash
python train.py > ../logs/0503/0503_roberta_original.log	# 训练并保存输出
python infer.py		# 推理
```

SBERT的训练与推理暂时未整合至`train.py`与`infer.py`，通过`train_sbert.py`和`infer_sbert.py`训练、推理。

