"""
w：权重
lr: 学习率
g: 梯度
v: 动量
θ: 防止分母为0的补偿
"""
import math
def Sgd(w,lr,g):
    w = w - lr * g
    return w

def Momentum(w, lr, Vt, g, beta):
    Vt = beta * Vt + (1-beta) * g
    w = w - lr * Vt
    return w, Vt

def Adagrad(w, lr, g, Gt):
    θ = 0.01 # 补偿
    Gt = Gt + g**2
    w = w - lr/(math.sqrt(Gt) + θ) * g
    return w, Gt

def Rmsprop(w, lr, g, beta, Gt):
    θ = 0.01
    Gt = beta * Gt + (1 - beta) * g**2
    w = w - lr / (math.sqrt(Gt) + θ) * g
    return w, Gt

def Adam(w, lr, g, beta1, beta2, Gt, Vt):
    θ = 0.01
    Vt = beta1 * Vt + (1 - beta1) * g ** 2
    Gt = beta2 * Gt + (1 - beta2) * g
    Vt_1 = Vt / (1-beta1)
    Gt_1 = Gt / (1-beta2)
    w = w - lr / (math.sqrt(Vt_1) + θ) * Gt_1
    return w, Gt, Vt

def Adamw(w, lr, g, beta1, beta2, Gt, Vt):
    θ = 0.01
    lamada = 0.001
    Vt = beta1 * Vt + (1 - beta1) * g ** 2
    Gt = beta2 * Gt + (1 - beta2) * g
    Vt_1 = Vt / (1-beta1)
    Gt_1 = Gt / (1-beta2)
    w = w - lr / (math.sqrt(Vt_1) + θ) *  (Gt_1 + lamada * w)
    return w, Gt, Vt





