# ----------------------------------------
# Written by Xiaoqing Guo
# ----------------------------------------

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import numpy as np
import torch.nn.functional as F
import cv2
from torch.optim import lr_scheduler

from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from net.loss import MaskCrossEntropyLoss, MaskBCELoss, MaskBCEWithLogitsLoss
from net.sync_batchnorm.replicate import patch_replication_callback
from scipy.spatial.distance import directed_hausdorff

def Jaccard_loss_cal(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = logits[:,1,:,:].cuda()
    true_1_hot = true_1_hot.type(logits.type())[:,1,:,:].cuda()
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dim=(1,2))
    cardinality = torch.sum(probas + true_1_hot, dim=(1,2))
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1. - jacc_loss)

def SoftCrossEntropy(inputs, target, reduction=True, eps=1e-6):
    inputs = nn.Softmax(dim=1)(inputs)
    log_likelihood = -torch.log(inputs+eps)
    if reduction:
        loss = torch.mean(torch.sum(torch.mul(log_likelihood, target), dim=1))
    else:
        loss = torch.sum(torch.mul(log_likelihood, target), dim=1)
    return loss

def make_one_hot(labels):
	target = torch.eye(cfg.MODEL_NUM_CLASSES)[labels]
	gt_1_hot = target.permute(0, 3, 1, 2).float().cuda()
	return gt_1_hot

def Label_smoothing(labels):
	one_hot = make_one_hot(labels)
	labels_ind = one_hot>0.5
	label_smoothing = torch.where(labels_ind, one_hot-cfg.EPS, one_hot+cfg.EPS)
	return label_smoothing

def CE_SPLloss_gen(loss, labels, lambda_fore, lambda_back, eps=1e-10):
	labels_ind = labels>0.
	loss_class1 = torch.where(labels_ind, loss, torch.ones_like(loss)*1000.)
	loss_class0 = torch.where(labels_ind, torch.ones_like(loss)*1000., loss)

	ClipV_weights_tensor_cpu1 = torch.where(loss_class1<=lambda_fore, torch.ones_like(loss), torch.zeros_like(loss)).cpu().data.numpy()
	ClipV_weights_tensor_cpu0 = torch.where(loss_class0<=lambda_back, torch.ones_like(loss), torch.zeros_like(loss)).cpu().data.numpy()	
	ClipV_weights_tensor1 = torch.from_numpy(ClipV_weights_tensor_cpu1).cuda()
	ClipV_weights_tensor0 = torch.from_numpy(ClipV_weights_tensor_cpu0).cuda()

	if torch.sum(ClipV_weights_tensor1) == 0:
		loss_class1 = torch.sum(loss*ClipV_weights_tensor1)
	else:
		loss_class1 = torch.sum(loss*ClipV_weights_tensor1)/torch.sum(ClipV_weights_tensor1)

	if torch.sum(ClipV_weights_tensor0) == 0:
		loss_class0 = torch.sum(loss*ClipV_weights_tensor0)
	else:
		loss_class0 = torch.sum(loss*ClipV_weights_tensor0)/torch.sum(ClipV_weights_tensor0)

	return loss_class1, loss_class0, ClipV_weights_tensor1, ClipV_weights_tensor0

def JA_SPLloss_gen(loss, lambda_JA, eps=1e-10):
	ClipV_weights_JA_cpu = torch.where(loss<=lambda_JA, torch.ones_like(loss), torch.zeros_like(loss)).cpu().data.numpy()
	ClipV_weights_JA = torch.from_numpy(ClipV_weights_JA_cpu).cuda()
	if torch.sum(ClipV_weights_JA).item() == 0:
		SPL_loss = torch.sum(loss*ClipV_weights_JA)
	else:
		SPL_loss = torch.sum(loss*ClipV_weights_JA)/torch.sum(ClipV_weights_JA)

	return SPL_loss, ClipV_weights_JA

def CE_Lambda_gen(loss, labels, percent_fore, percent_back, eps=1e-10):
	loss = loss.cpu().data.numpy()
	labels = labels.cpu().data.numpy()
	len_class1 = np.sum(labels)
	len_class0 = np.sum(np.ones_like(labels)) - np.sum(labels)
	labels_ind = labels>0.5

	loss_class1 = np.where(labels_ind, loss, np.ones_like(loss)*10e9)
	loss_class0 = np.where(labels_ind, np.ones_like(loss)*10e9, loss)
	loss_class1_sorted = np.sort(loss_class1, axis=None) 
	loss_class0_sorted = np.sort(loss_class0, axis=None) 

	index1 = np.int32(len_class1 * percent_fore)
	index0 = np.int32(len_class0 * percent_back)
	lambda_class1 = loss_class1_sorted[index1-1]
	lambda_class0 = loss_class0_sorted[index0-1]
	return lambda_class1, lambda_class0

def JA_Lambda_gen(loss, percent, eps=1e-10):
	loss = loss.cpu().data.numpy()
	len_loss = np.sum(np.ones_like(loss)) 
	loss_sorted = np.sort(loss, axis=None) 
	index = np.int32(len_loss * percent)
	lambda_JA = loss_sorted[index-1]
	return lambda_JA

def model_snapshot(model, new_file=None, old_file=None):
    if os.path.exists(old_file) is True:
        os.remove(old_file) 
    torch.save(model, new_file)
    print('      %s has been saved\n'%new_file)

def train_net():
	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train', cfg.DATA_AUG)
#	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train_cross'+cfg.DATA_CROSS, cfg.DATA_AUG)
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TRAIN_BATCHES, 
				shuffle=cfg.TRAIN_SHUFFLE, 
				num_workers=cfg.DATA_WORKERS,
				drop_last=True)

	test_dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test')
#	test_dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test_cross'+cfg.DATA_CROSS)
	test_dataloader = DataLoader(test_dataset, 
				batch_size=cfg.TEST_BATCHES, 
				shuffle=False, 
				num_workers=cfg.DATA_WORKERS)

	net = generate_net(cfg)

	print('Use %d GPU'%cfg.TRAIN_GPUS)
	device = torch.device(0)
	if cfg.TRAIN_GPUS > 1:
		net = nn.DataParallel(net)
		patch_replication_callback(net)
	net.to(device)	

	if cfg.TRAIN_CKPT:
		pretrained_dict = torch.load(cfg.TRAIN_CKPT)
		net_dict = net.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape==net_dict[k].shape)}
		net_dict.update(pretrained_dict)
		net.load_state_dict(net_dict)
		# net.load_state_dict(torch.load(cfg.TRAIN_CKPT),False)
    
#	criterion = nn.CrossEntropyLoss(ignore_index=255)
	criterion = nn.BCEWithLogitsLoss()
#	optimizer = optim.SGD(net.parameters(), lr = cfg.TRAIN_LR, momentum=cfg.TRAIN_MOMENTUM)
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.TRAIN_LR)  
	scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # decay LR by a factor of 0.5 every 30 epochs

	itr = cfg.TRAIN_MINEPOCH * len(dataloader)
	max_itr = cfg.TRAIN_EPOCHS*len(dataloader)
	best_jacc = 0.
	best_epoch = 0
	for epoch in range(cfg.TRAIN_MINEPOCH, cfg.TRAIN_EPOCHS):
		running_loss = 0.0
		running_dice_loss = 0.0
		net.train()
		scheduler.step()
		#now_lr = scheduler.get_lr()
		for i_batch, sample_batched in enumerate(dataloader):
			now_lr = adjust_lr(optimizer, itr, max_itr)
			inputs_batched, labels_batched = sample_batched['image'], sample_batched['segmentation']
			optimizer.zero_grad()
			inputs_batched = inputs_batched.cuda()
			labels_batched = labels_batched.long().cuda()
			outputs = net(inputs_batched)
			loss = 0
			dice_loss = 0
			for output in outputs:
				output = output.cuda()
				loss += criterion(output, make_one_hot(labels_batched))
				soft_predicts_batched = nn.Softmax(dim=1)(output)
				dice_loss += Jaccard_loss_cal(labels_batched, soft_predicts_batched, eps=1e-7)
			loss /= len(outputs)
			dice_loss /= len(outputs)

			(loss+dice_loss).backward()
			optimizer.step()

			running_loss += loss.item()
			running_dice_loss += dice_loss.item()
			
			itr += 1

		i_batch = i_batch + 1
		print('epoch:%d/%d\tmean loss:%g\tmean dice loss:%g \n' %  (epoch, cfg.TRAIN_EPOCHS, running_loss/i_batch, running_dice_loss/i_batch))
		
		#### start testing now
		if epoch % 10 == 0:
			IoUP = test_one_epoch(test_dataset, test_dataloader, net, epoch)
		if IoUP > best_jacc:
			model_snapshot(net.state_dict(), new_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_%s_epoch%d_jac%.3f.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,epoch,IoUP)),
                old_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_%s_epoch%d_jac%.3f.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,best_epoch,best_jacc)))
			best_jacc = IoUP
			best_epoch = epoch

def adjust_lr(optimizer, itr, max_itr):
	now_lr = cfg.TRAIN_LR * (1 - (itr/(max_itr+1)) ** cfg.TRAIN_POWER)
	for param_group in optimizer.param_groups:
		param_group['lr'] = now_lr
	return now_lr

def test_one_itr(predicts_batched, labels_batched):
	IoUarray = 0.
	samplenum = 0.
	result = torch.argmax(predicts_batched, dim=1).cpu().numpy().astype(np.uint8)
	label = labels_batched.cpu().numpy()
	[batch, channel, height, width] = predicts_batched.size()
	del predicts_batched			
	del labels_batched			
	for i in range(batch):
		p = result[i,:,:]					
		l = label[i,:,:]
		predict = np.int32(p)
		gt = np.int32(l)
		TP = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)
		P = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)
		T = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)  

		P = np.sum((predict==1)).astype(np.float64)
		T = np.sum((gt==1)).astype(np.float64)
		TP = np.sum((gt==1)*(predict==1)).astype(np.float64)

		IoU = TP/(T+P-TP+10e-5)
		IoUarray += IoU
		samplenum += 1

	return IoUarray, samplenum

def test_one_epoch(dataset, DATAloader, net, epoch):
	#### start testing now
	Acc_array = 0.
	Prec_array = 0.
	Spe_array = 0.
	Rec_array = 0.
	IoU_array = 0.
	Dice_array = 0.
	HD_array = 0.
	sample_num = 0.
	result_list = []
	CEloss_list = []
	JAloss_list = []
	Label_list = []
	net.eval()
	with torch.no_grad():
		for i_batch, sample_batched in enumerate(DATAloader):
			name_batched = sample_batched['name']
			row_batched = sample_batched['row']
			col_batched = sample_batched['col']

			[batch, channel, height, width] = sample_batched['image'].size()
			multi_avg = torch.zeros((batch, cfg.MODEL_NUM_CLASSES, height, width), dtype=torch.float32).cuda()
			labels_batched = sample_batched['segmentation'].cpu().numpy()
			for rate in cfg.TEST_MULTISCALE:
				inputs_batched = sample_batched['image_%f'%rate]
				inputs_batched = inputs_batched.cuda()
				predicts = net(inputs_batched)[-1].cuda()
				predicts_batched = predicts.clone()
				del predicts
				if cfg.TEST_FLIP:
					inputs_batched_flip = torch.flip(inputs_batched,[3]) 
					predicts_flip = torch.flip(net(inputs_batched_flip),[3]).cuda()
					predicts_batched_flip = predicts_flip.clone()
					del predicts_flip
					predicts_batched = (predicts_batched + predicts_batched_flip) / 2.0
			
				predicts_batched = F.interpolate(predicts_batched, size=None, scale_factor=1/rate, mode='bilinear', align_corners=True)
				multi_avg = multi_avg + predicts_batched
				del predicts_batched			
			multi_avg = multi_avg / len(cfg.TEST_MULTISCALE)
			result = torch.argmax(multi_avg, dim=1).cpu().numpy().astype(np.uint8)

			for i in range(batch):
				row = row_batched[i]
				col = col_batched[i]
				p = result[i,:,:]					
				l = labels_batched[i,:,:]
				#p = cv2.resize(p, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
				#l = cv2.resize(l, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
				predict = np.int32(p)
				gt = np.int32(l)
				cal = gt<255
				mask = (predict==gt) * cal 
				TP = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)
				TN = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)
				P = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)
				T = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)  

				P = np.sum((predict==1)).astype(np.float64)
				T = np.sum((gt==1)).astype(np.float64)
				TP = np.sum((gt==1)*(predict==1)).astype(np.float64)
				TN = np.sum((gt==0)*(predict==0)).astype(np.float64)

				Acc = (TP+TN)/(T+P-TP+TN)
				Prec = TP/(P+10e-6)
				Spe = TN/(P-TP+TN)
				Rec = TP/T
				DICE = 2*TP/(T+P)
				IoU = TP/(T+P-TP)
			#	HD = max(directed_hausdorff(predict, gt)[0], directed_hausdorff(predict, gt)[0])
				beta = 2
				HD = Rec*Prec*(1+beta**2)/(Rec+beta**2*Prec+1e-10)
				Acc_array += Acc
				Prec_array += Prec
				Spe_array += Spe
				Rec_array += Rec
				Dice_array += DICE
				IoU_array += IoU
				HD_array += HD
				sample_num += 1
				#p = cv2.resize(p, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
				result_list.append({'predict':np.uint8(p*255), 'label':np.uint8(l*255), 'IoU':IoU, 'name':name_batched[i]})

		Acc_score = Acc_array*100/sample_num
		Prec_score = Prec_array*100/sample_num
		Spe_score = Spe_array*100/sample_num
		Rec_score = Rec_array*100/sample_num
		Dice_score = Dice_array*100/sample_num
		IoUP = IoU_array*100/sample_num
		HD_score = HD_array*100/sample_num
		print('%10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%\n'%('Acc',Acc_score,'Sen',Rec_score,'Spe',Spe_score,'Prec',Prec_score,'Dice',Dice_score,'Jac',IoUP,'F2',HD_score))
		if epoch % 50 == 0:
			dataset.save_result_train(result_list, cfg.MODEL_NAME)

		return IoUP

if __name__ == '__main__':
	train_net()
