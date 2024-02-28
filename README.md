# 科大讯飞
### 1.LLM训练过程的超参数怎么设置的？
    微调方式采用lora
    lora的r设置为8
    学习率1e-5
    batch需要根据显存占用调整，一般为2的N次方，可以用梯度累积来增加batch-size
    epoch一般设置为20，在训练过程中观察验证集的损失变化情况，进行早停处理
    精度采用FP16
    显存不充足时候 gradient-checkpoint设置为 True
    warm-up-step 酌情设置
    是否使用deepspeed，以及stage-几，和cpu-offload,根据显存设置
### 2.怎么预估LLM训练的batch-size?
    
### 3.LLM-SFT的数据来源，数据量，训练过程你是怎么迭代的？

### 4.逻辑回归是线性的么？
    不是，逻辑回归是线性回归+sigmoid，线性回归是线性的，但是sigmoid带来了非线性的因子。尽管它号称广义线性模型
### 5.bert模型做分类，你是怎么使用的？
    bert的输出有last_hidden_state, pool_output
    last_hidden_state对应cls和每个token的输出[batch, seq_len, 768]，pool_output是cls接一个全连接层 + tanh激活层的输出[batch, 768]
    如果分类任务，一般采用pool_output直接加一个线性层
    序列标注任务，取last_hidden_state每个token的输出
### 6.bert有什么缺点/局限性
    1.预训练时候的【mask】在下游任务微调时候并不出现，这就造成了一定的信息偏差，mask的811策略（80%mask，10%随机替换，10%不变）只能缓解
    2.预训练的MLM任务，每个batch只有15%的token参与训练，收敛速度慢
### 7.bert做分类时候，如果遇到过拟合怎么办？
    1.观察验证集损失信息，早停
    2.增加训练数据
    3.dropout
    4.选用小点的模型，减少参数量
    5.数据清洗
### 8.bert数据不均衡怎么处理？其他模型怎么处理
    1.数据层面，增加量少类别的样本（转译，同义词替换，LLM做相同语义文本生成），量多类别样本欠采样。
    2.模型层面，损失函数可以改为带权重的交叉熵损失
    3.评估指标，关注较少类别的召回率（所有该类别的样本，被正确找出来的比例）
### 9.负采样怎么做？可以用fasttext或word2vec的负采样举例
    负采样是一种用于词嵌入模型训练的采样方式，如果使用softmax进行全词表的计算，计算量很大。因此只使用正样本和采样的几个负样本计算损失，更新他们的权重。
    采样方式：随机或者按词频分布以一定概率采样
### 10.你的强化学习DPO是怎么训练的？为什么不用PPO？
    1.数据，一个input对应一个chosen和一个reject；来源是推理产生的Badcase作为reject，人工修改后的座位chosen；数据量5k；
    2.模型,策略模型Policy和参照模型Reference,Policy参与训练，Reference参数固定
    3.损失，Policy输出chosen和reject的logits_prob之差作为Policy得分,Ref输出chosen和reject的logits_prob之差作为Ref得分,loss = -logsigmoid(系数β * （Pilicy得分-Ref得分）
    4.参数，epoch=1，学习率1e-5, β=0.1

    PPO需要用到4个模型，显存开销大于DPO，以训练过程不稳定和效果不稳定著称；
    PPO的Reward-model难训练，而DPO也能起效果，从结果论上选择DPO

### 11.GLM,Baichuan,Qwen,LLama这些模型哪个更好用，你怎么评估哪个更好用的，他们的结构说一下。
    开源时间从开始的LLama到GLM，再到Baichuan，Qwen。后出来的会比先出来的好用，目前感觉第一梯队是GLM4，Qwen

    评估模型：
        1.针对专利和期刊中的高频技术词进行提问，判断模型训练数据是否包含该知识点
        2.采用prompt或few-shot方式，引导模型按指令要求输出，判断模型对指令的遵循能力
        3.同样数据sft之后的模型，人工对其输出进行评估

    模型结构：
        


### 12.Batch-Norm和Layer-Norm的区别，为什么CV用BN，NLP用LN？

### 13.常见损失函数