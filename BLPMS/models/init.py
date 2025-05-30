import math
import torch
import torch.nn as nn

from args import args
"""
该文档用于module参数的初始化
"""

def signed_constant(module):
    """
    该函数用于对给定的模块（module）进行权重初始化，使用了“signed constant”方法。具体步骤如下：

    根据传入的模块（module）的权重（weight），计算出正确的扇出值（fan）。
    根据给定的非线性函数（args.nonlinearity），计算出增益值（gain）。
    将增益值（gain）除以扇出值（fan）的平方根，得到标准差（std）。
    将模块（module）的权重（weight）数据更新为权重的符号乘以标准差（std）。
    """
    fan = nn.init._calculate_correct_fan(module.weight, args.mode)
    #args.mode权重初始化模式，default:fan_in
    gain = nn.init.calculate_gain(args.nonlinearity)#args.nonlinearity激活函数，default:relu
    std = gain / math.sqrt(fan)
    module.weight.data = module.weight.data.sign() * std

def unsigned_constant(module):
    fan = nn.init._calculate_correct_fan(module.weight, args.mode)
    gain = nn.init.calculate_gain(args.nonlinearity)
    std = gain / math.sqrt(fan)
    module.weight.data = torch.ones_like(module.weight.data) * std

def kaiming_normal(module):
    nn.init.kaiming_normal_(
        module.weight, mode=args.mode, nonlinearity=args.nonlinearity
    )

def kaiming_uniform(module):
    nn.init.kaiming_uniform_(
        module.weight, mode=args.mode, nonlinearity=args.nonlinearity
    )

def xavier_normal(module):
    nn.init.xavier_normal_(
        module.weight
    )

def glorot_uniform(module):
    nn.init.xavier_uniform_(
        module.weight
    )

def xavier_constant(module):
    fan = nn.init._calculate_correct_fan(module.weight, args.mode)
    gain = 1.0
    std = gain / math.sqrt(fan)
    module.weight.data = module.weight.data.sign() * std

def default(module):
    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
