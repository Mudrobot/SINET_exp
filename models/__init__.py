from __future__ import absolute_import

from .resnet import *
__factory = {
    'resnet50': ResNet50,
    'sanet': SANet,
    'sinet': SINet,
    'idnet': IDNet,
    'sbnet': SBNet,
    ############# Self defined Net #############
    'ctnet': CTNet,
    'mynet': MyNet,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))

    m = __factory[name](*args, **kwargs)

    return m
