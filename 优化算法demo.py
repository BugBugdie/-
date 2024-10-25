"""
w：权重
lr: 学习率
g: 梯度
v: 动量
θ: 防止分母为0的补偿
"""
import math

# 随机梯度下降，每次只使用一个样本或一个批次的样本（mini-batch）计算梯度，容易陷入局部最优解
def Sgd(w,lr,g):
    w = w - lr * g
    return w

# 动量算法，不仅采用当前的梯度信息，也保留了历史的梯度信息，摆脱局部最优解和跨过平坦区域，加速训练过程。
def Momentum(w, lr, Vt, g, beta):
    Vt = beta * Vt + (1-beta) * g
    w = w - lr * Vt
    return w, Vt

# 自适应的学习率，根据参数的梯度大小适配学习率，大的梯度搭配小的学习率，防止跨过最优解产生的震荡
def Adagrad(w, lr, g, Gt):
    θ = 0.01 # 补偿
    Gt = Gt + g**2
    w = w - lr/(math.sqrt(Gt) + θ) * g
    return w, Gt

# 在保留历史梯度信息的策略上更加合理
def Rmsprop(w, lr, g, beta, Gt):
    θ = 0.01
    Gt = beta * Gt + (1 - beta) * g**2
    w = w - lr / (math.sqrt(Gt) + θ) * g
    return w, Gt

# RMSprop和mountum的结合
def Adam(w, lr, g, beta1, beta2, Gt, Vt):
    θ = 0.01
    Vt = beta1 * Vt + (1 - beta1) * g ** 2
    Gt = beta2 * Gt + (1 - beta2) * g
    Vt_1 = Vt / (1-beta1)
    Gt_1 = Gt / (1-beta2)
    w = w - lr / (math.sqrt(Vt_1) + θ) * Gt_1
    return w, Gt, Vt

# Adamw和Adam区别是：权重衰减作用的位置不同，adam作用在梯度上，adamw作用在参数上
def Adamw(w, lr, g, beta1, beta2, Gt, Vt):
    θ = 0.01
    lamada = 0.001
    Vt = beta1 * Vt + (1 - beta1) * g ** 2
    Gt = beta2 * Gt + (1 - beta2) * g
    Vt_1 = Vt / (1-beta1)
    Gt_1 = Gt / (1-beta2)
    w = w - lr / (math.sqrt(Vt_1) + θ) *  (Gt_1 + lamada * w)
    return w, Gt, Vt





