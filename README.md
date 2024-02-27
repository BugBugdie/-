# 科大讯飞
# 1.LLM训练过程的超参数怎么设置的？
    微调方式采用lora
    lora的r设置为8
    学习率1e-5
    batch需要根据显存占用调整，一般为2的N次方，可以用梯度累积来增加batch-size
    epoch一般设置为20，在训练过程中观察验证集的损失变化情况，进行早停处理
    精度采用FP16
    显存不充足时候 gradient-checkpoint设置为 True
    warm-up-step 酌情设置
    是否使用deepspeed，以及stage-几，和cpu-offload,根据显存设置
# 2.怎么预估LLM训练的batch-size?
    
# 3.LLM-SFT的数据来源，数据量，训练过程你是怎么迭代的？

# 4.逻辑回归是线性的么？
    不是，逻辑回归是线性回归+sigmoid，线性回归是线性的，但是sigmoid带来了非线性的因子。尽管它号称广义线性模型
# 5.bert模型做分类，你是怎么使用的？
    bert的输出有last_hidden_state, pool_output
    last_hidden_state对应cls和每个token的输出[batch, seq_len, 768]，pool_output是cls接一个全连接层 + tanh激活层的输出[batch, 768]
    如果分类任务，一般采用pool_output直接加一个线性层
    序列标注任务，取last_hidden_state每个token的输出
# 6.bert有什么缺点/局限性
    1.预训练时候的【mask】在下游任务微调时候并不出现，这就造成了一定的信息偏差，mask的811策略（80%mask，10%随机替换，10%不变）只能缓解
    2.预训练的MLM任务，每个batch只有15%的token参与训练，收敛速度慢
# 7.bert做分类时候，如果遇到过拟合怎么办？
    1.观察验证集损失信息，早停
    2.增加

# 8.bert数据不均衡怎么处理？其他模型怎么处理

# 9.负采样怎么做？可以用fasttext的负采样举例

# 10.你的强化学习DPO是怎么训练的？为什么不用PPO？

# 11.GLM,Baichuan,Qwen,LLama这些模型哪个更好用，你怎么评估哪个更好用的，他们的结构说一下。

# 12.Batch-Norm和Layer-Norm的区别，为什么CV用BN，NLP用LN？

# 13.常见损失函数