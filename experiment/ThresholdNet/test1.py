# ----------------------------------------
# Written by Xiaoqing Guo
# ----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2

from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from net.sync_batchnorm.replicate import patch_replication_callback

from torch.utils.data import DataLoader

def test_net():
	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test')
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TEST_BATCHES, 
				shuffle=False, 
				num_workers=cfg.DATA_WORKERS)
	
	net = generate_net(cfg)
	print('net initialize')
	if cfg.TEST_CKPT is None:
		raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')
	

	print('Use %d GPU'%cfg.TEST_GPUS)
	device = torch.device('cuda')
	if cfg.TEST_GPUS > 1:
		net = nn.DataParallel(net)
		patch_replication_callback(net)
	net.to(device)

	print('start loading model %s'%cfg.TEST_CKPT)
	model_dict = torch.load(cfg.TEST_CKPT,map_location=device)
	net.load_state_dict(model_dict)
	
	net.eval()	
	IoU_array = 0.
	sample_num = 0.
	result_list = []
	with torch.no_grad():
		for i_batch, sample_batched in enumerate(dataloader):
			name_batched = sample_batched['name']
			row_batched = sample_batched['row']
			col_batched = sample_batched['col']

			[batch, channel, height, width] = sample_batched['image'].size()
			multi_avg = torch.zeros((batch, cfg.MODEL_NUM_CLASSES, height, width), dtype=torch.float32).to(1)
			for rate in cfg.TEST_MULTISCALE:
				inputs_batched = sample_batched['image_%f'%rate]
				predicts = net(inputs_batched).to(1)
				predicts_batched = predicts.clone()
				del predicts
				if cfg.TEST_FLIP:
					inputs_batched_flip = torch.flip(inputs_batched,[3]) 
					predicts_flip = torch.flip(net(inputs_batched_flip),[3]).to(1)
					predicts_batched_flip = predicts_flip.clone()
					del predicts_flip
					predicts_batched = (predicts_batched + predicts_batched_flip) / 2.0
			
				predicts_batched = F.interpolate(predicts_batched, size=None, scale_factor=1/rate, mode='bilinear', align_corners=True)
				multi_avg = multi_avg + predicts_batched
				del predicts_batched
			
			multi_avg = multi_avg / len(cfg.TEST_MULTISCALE)
			result = torch.argmax(multi_avg, dim=1).cpu().numpy().astype(np.uint8)
			
			labels_batched = sample_batched['segmentation'].cpu().numpy()
			for i in range(batch):
				row = row_batched[i]
				col = col_batched[i]
				p = result[i,:,:]					
				l = labels_batched[i,:,:]
			#	thres = threshold[i,1,:,:]
			#	p = cv2.resize(p, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
			#	l = cv2.resize(l, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
				predict = np.int32(p)
				gt = np.int32(l)
			#	print(np.unique(predict), np.unique(gt))
				cal = gt<255
				mask = (predict==gt) * cal 
				TP = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)
				P = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)
				T = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)               
				for cc in range(cfg.MODEL_NUM_CLASSES):
					P[cc] = np.sum((predict==cc))
					T[cc] = np.sum((gt==cc))
					TP[cc] = np.sum((gt==cc)*mask)
				TP = TP.astype(np.float64)
				T = T.astype(np.float64)
				P = P.astype(np.float64)
				IoU = TP/(T+P-TP)
				#	print(TP, T, P)
				IoU_array += IoU
				sample_num += 1
			#	p = cv2.resize(p, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
			#	l = cv2.resize(l, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
			#	result_list.append({'predict':np.uint8(p*255), 'threshold':np.uint8(thres*255), 'label':np.uint8(l*255), 'IoU':IoU, 'name':name_batched[i]})

		for i in range(cfg.MODEL_NUM_CLASSES):
			if i == 0:
				print('%15s:%7.3f%%'%('backbound',IoU_array[i]*100/sample_num))
			else:
				print('%15s:%7.3f%%'%('melanoma',IoU_array[i]*100/sample_num))
		miou = np.mean(IoU_array/sample_num)
		print('==================================')
		print('%15s:%7.3f%%\n'%('mIoU',miou*100))	

if __name__ == '__main__':
	test_net()


