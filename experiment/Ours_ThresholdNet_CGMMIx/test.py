# ----------------------------------------
# Written by Yude Wang
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
	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test_cross0')
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
				predicts, threshold = net(inputs_batched)
				predicts = predicts.to(1) 
				threshold = threshold.to(1)
				predicts_batched = predicts.clone()
				threshold_batched = threshold.clone()
				del predicts
				del threshold
				if cfg.TEST_FLIP:
					inputs_batched_flip = torch.flip(inputs_batched,[3]) 
					predicts_flip, threshold_flip = net(inputs_batched_flip)
					predicts_flip = torch.flip(predicts_flip,[3]).to(1)
					threshold_flip = torch.flip(threshold_flip,[3]).to(1)
					predicts_batched_flip = predicts_flip.clone()
					threshold_batched_flip = threshold_flip.clone()
					del predicts_flip
					del threshold_flip
					predicts_batched = (predicts_batched + predicts_batched_flip) / 2.0
					threshold_batched = (threshold_batched + threshold_batched_flip) / 2.0
			
				predicts_batched = F.interpolate(predicts_batched, size=None, scale_factor=1/rate, mode='bilinear', align_corners=True)
				threshold_batched = F.interpolate(threshold_batched, size=None, scale_factor=1/rate, mode='bilinear', align_corners=True)
				multi_avg = multi_avg + predicts_batched
				del predicts_batched
			
			multi_avg = multi_avg / len(cfg.TEST_MULTISCALE)
		#	print(multi_avg.max(), multi_avg.min())
			multi_avg = nn.Softmax(dim=1)(multi_avg)
		#	print(multi_avg.max(), multi_avg.min())
			multi_avg = multi_avg - threshold_batched
			result = torch.argmax(multi_avg, dim=1).cpu().numpy().astype(np.uint8)
			labels_batched = sample_batched['segmentation'].cpu().numpy()
			threshold = threshold_batched.cpu().numpy()
			del threshold_batched
	
			for i in range(batch):
				row = row_batched[i]
				col = col_batched[i]
			#	max_edge = max(row,col)
			#	rate = cfg.DATA_RESCALE / max_edge
			#	new_row = row*rate
			#	new_col = col*rate
			#	s_row = (cfg.DATA_RESCALE-new_row)//2
			#	s_col = (cfg.DATA_RESCALE-new_col)//2
	 
			#	p = predicts_batched[i, s_row:s_row+new_row, s_col:s_col+new_col]
				p = result[i,:,:]
				p = cv2.resize(p, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
				result_list.append({'predict':np.uint8(p*255), 'name':name_batched[i]})

			#print('%d/%d'%(i_batch,len(dataloader)))
	dataset.save_result(result_list, cfg.MODEL_NAME)
	dataset.do_python_eval(cfg.MODEL_NAME)
	print('Test finished')

if __name__ == '__main__':
	test_net()


