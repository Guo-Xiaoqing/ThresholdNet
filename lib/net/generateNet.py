# ----------------------------------------
# Written by Xiaoqing Guo
# ----------------------------------------

import torch
import torch.nn as nn
from net.deeplabv3plus import deeplabv3plus
from net.fcn import FCN8s
from net.unetplusplus import NestedUNet
from net.SegNet import SegNet

def generate_net(cfg):
	if cfg.MODEL_NAME == 'deeplabv3plus' or cfg.MODEL_NAME == 'deeplabv3+':
		return deeplabv3plus(cfg)
	if cfg.MODEL_NAME == 'fcn' or cfg.MODEL_NAME == 'FCN':
		return FCN8s(cfg)
	if cfg.MODEL_NAME == 'UNetplusplus' or cfg.MODEL_NAME == 'UNet++':
		return NestedUNet(cfg)
	if cfg.MODEL_NAME == 'segnet' or cfg.MODEL_NAME == 'SegNet':
		return SegNet(cfg)
	# if cfg.MODEL_NAME == 'supernet' or cfg.MODEL_NAME == 'SuperNet':
	# 	return SuperNet(cfg)
	# if cfg.MODEL_NAME == 'eanet' or cfg.MODEL_NAME == 'EANet':
	# 	return EANet(cfg)
	# if cfg.MODEL_NAME == 'danet' or cfg.MODEL_NAME == 'DANet':
	# 	return DANet(cfg)
	# if cfg.MODEL_NAME == 'deeplabv3plushd' or cfg.MODEL_NAME == 'deeplabv3+hd':
	# 	return deeplabv3plushd(cfg)
	# if cfg.MODEL_NAME == 'danethd' or cfg.MODEL_NAME == 'DANethd':
	# 	return DANethd(cfg)
	else:
		raise ValueError('generateNet.py: network %s is not support yet'%cfg.MODEL_NAME)
