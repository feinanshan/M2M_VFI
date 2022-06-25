import copy
from vfi.dataset import VimeoDataset
from vfi.loss import L1, Charbonnier, Ternary, SOBEL, VGGPerceptualLoss
from vfi.model.m2m import M2M_PWC
from vfi.optimizer import Optimizer


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "vimeo90k": VimeoDataset,
    }[name]
#end


def get_loss(cfg):
    """get_loss

    :param name:
    """
    key2loss = {
    "l1": L1,
    "char": Charbonnier,
    "ter": Ternary,
    "sobel": SOBEL,
    "vggp": VGGPerceptualLoss
    }

    loss_list = []

    for los, wei in zip(cfg['name'],cfg['weights']):
        loss_list.append([los,wei,key2loss[los]()])
    #end

    return loss_list
#end


def get_model(cfg):
    """get_model

    :param name:
    """
    key2model = {
    "m2m_pwc": M2M_PWC,
    }
    name = cfg["arch"]
    model = key2model[name](cfg["ratio"],cfg["branch"])

    return model
#end


def get_optimizer(cfg, model):
    param_dict = copy.deepcopy(cfg)
    param_dict['model'] = model
    optimizer = Optimizer(**param_dict) 

    return optimizer
#end
