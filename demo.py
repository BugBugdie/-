# 实现一个梯度下降

def func_grad(x0, x1):
    return 2*x0+ x1*x1, x0*x0+2*x1


def func(x0, x1):
    return x0*x0+x1*x1

a, b, lr = 2,3, 0.001

for i in range(10000):
    gd1, gd2 = func_grad(a, b)
    a, b = a - lr*gd1, b - lr*gd2
    res = func(a,b)
    print(round(a,2),round(b,2),round(res,2))

