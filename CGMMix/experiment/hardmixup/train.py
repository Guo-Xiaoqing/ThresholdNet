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
    if true.shape[1] == 2:
        true_1_hot = true.float()
        probas = logits[:,1,:,:].to(1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = logits[:,1,:,:].to(1)
    true_1_hot = true_1_hot.type(logits.type())[:,1,:,:].to(1)
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dim=(1,2))
    cardinality = torch.sum(probas + true_1_hot, dim=(1,2))
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1. - jacc_loss)

def SoftCrossEntropy(inputs, target, reduction='average', eps=1e-6):
    log_likelihood = -torch.log(inputs+eps)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.mean(torch.sum(torch.mul(log_likelihood, target), dim=(1)))
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss

def Lambda_gen(loss, labels, percent_fore, percent_back, eps=1e-10):
	len_class1 = torch.sum(labels)
	len_class0 = torch.sum(torch.ones_like(labels)) - torch.sum(labels)
	labels_ind = labels>0.

	loss_class1 = torch.where(labels_ind, loss, torch.ones_like(loss)*1000.)
	loss_class0 = torch.where(labels_ind, torch.ones_like(loss)*1000., loss)
	loss_class1_seq = loss_class1.reshape(-1)
	loss_class0_seq = loss_class0.reshape(-1)
	loss_class1_sorted, _ = loss_class1_seq.sort(dim=0, descending=False)
	loss_class0_sorted, _ = loss_class0_seq.sort(dim=0, descending=False)

	index1 = np.int32(len_class1.cpu().numpy() * percent_fore)
	index0 = np.int32(len_class0.cpu().numpy() * percent_back)
	lambda_class1 = loss_class1_sorted[index1-1]
	lambda_class0 = loss_class0_sorted[index0-1]
#	print(index1, index0)
	return loss_class1, loss_class0, lambda_class1, lambda_class0

def model_snapshot(model, new_file=None, old_file=None):
    if os.path.exists(old_file) is True:
        os.remove(old_file) 
    torch.save(model, new_file)
    print('      %s has been saved'%new_file)

def make_one_hot(labels):
	target = torch.eye(cfg.MODEL_NUM_CLASSES)[labels]
	gt_1_hot = target.permute(0, 3, 1, 2).float().to(1)
	return gt_1_hot

def input_mixup(input1, input2, lambda1):
	lambda2 = torch.from_numpy(np.array(1.))-lambda1
	mixup_input = input1.mul(lambda1)+ input2.mul(lambda2)
	return mixup_input

def label_mixup(labels1, labels2, lambda1):
	lambda2 = torch.from_numpy(np.array(1.))-lambda1
	labels1_1_hot = make_one_hot(labels1)
	labels2_1_hot = make_one_hot(labels2)
	mixup_label = labels1_1_hot.mul(lambda1)+ labels2_1_hot.mul(lambda2)

	mixup_label_margin = cfg.Mixup_label_margin
	mixup_label_ones = torch.ones_like(labels1)
	mixup_label_zeros = torch.zeros_like(labels1)
	mixup_label[:,1,:,:] = torch.where(mixup_label[:,1,:,:] >= mixup_label_margin, mixup_label_ones, mixup_label_zeros)
	mixup_label_1 = mixup_label[:,1,:,:].long().to(1) 
	return mixup_label_1
#	return mixup_label

def train_one_epoch(inputs_batched, labels_batched, net, optimizer, epoch):
	criterion = nn.CrossEntropyLoss(ignore_index=255)
	optimizer.zero_grad()
	_, predicts_batched = net(inputs_batched)
	predicts_batched = predicts_batched.to(1)            
					
	soft_predicts_batched = nn.Softmax(dim=1)(predicts_batched)
	jaccard_loss = Jaccard_loss_cal(labels_batched, soft_predicts_batched, eps=1e-7)
	if labels_batched.shape[1] == 2:
		CE_loss = SoftCrossEntropy(soft_predicts_batched, labels_batched)
	else:
		CE_loss = criterion(predicts_batched, labels_batched)
			
	seg_loss = CE_loss
	seg_loss.backward()
	optimizer.step()
	return CE_loss, jaccard_loss
	
def train_net():
#	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train_cross'+cfg.DATA_CROSS, cfg.DATA_AUG)
	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train', cfg.DATA_AUG)
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TRAIN_BATCHES, 
				shuffle=cfg.TRAIN_SHUFFLE, 
				num_workers=cfg.DATA_WORKERS,
				drop_last=True)

	dataset_mixup = generate_dataset(cfg.DATA_NAME, cfg, 'train', cfg.DATA_AUG)
	dataloader_mixup = DataLoader(dataset_mixup, 
				batch_size=cfg.TRAIN_BATCHES, 
				shuffle=cfg.TRAIN_SHUFFLE, 
				num_workers=cfg.DATA_WORKERS,
				drop_last=True)

#	test_dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test_cross'+cfg.DATA_CROSS)
	test_dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test')
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
    
	optimizer = optim.SGD(
		params = [
			{'params': get_params(net.module,key='1x'), 'lr': cfg.TRAIN_LR},
			{'params': get_params(net.module,key='10x'), 'lr': 10*cfg.TRAIN_LR}
		],
		momentum=cfg.TRAIN_MOMENTUM
	)
  
	itr = cfg.TRAIN_MINEPOCH * len(dataloader)
	max_itr = cfg.TRAIN_EPOCHS*len(dataloader)
	best_jacc = 0.
	best_epoch = 0
	for epoch in range(cfg.TRAIN_MINEPOCH, cfg.TRAIN_EPOCHS):
		running_loss = 0.0
		seg_jac_running_loss = 0.0
		SPL_loss1 = 0.0
		SPL_loss2 = 0.0
		epoch_samples1 = 0.0
		epoch_samples2 = 0.0
		Lambda1 = 0.0
		Lambda0 = 0.0
		mixup_running_loss = 0.0
		mixup_seg_jac_running_loss = 0.0
		mixup_SPL_loss1 = 0.0
		mixup_SPL_loss2 = 0.0
		mixup_epoch_samples1 = 0.0
		mixup_epoch_samples2 = 0.0
		mixup_Lambda1 = 0.0
		mixup_Lambda0 = 0.0
		dataset_list = []
		net.train()        
        #########################################################
        ########### give lambda && decay coefficient ############
        #########################################################        
		for i_batch, (sample_batched, mixup_batched) in enumerate(zip(dataloader, dataloader_mixup)):
			now_lr = adjust_lr(optimizer, itr, max_itr)
			inputs_batched, labels_batched = sample_batched['image'], sample_batched['segmentation']
			mixup_inputs_batched, mixup_labels_batched = mixup_batched['image'], mixup_batched['segmentation']
			labels_batched = labels_batched.long().to(1)
			mixup_labels_batched = mixup_labels_batched.long().to(1) 

			alpha = cfg.Mixup_Alpha
			random_lambda = np.random.beta(alpha, alpha)
			mixuped_input = input_mixup(inputs_batched, mixup_inputs_batched, random_lambda)
			mixuped_label = label_mixup(labels_batched, mixup_labels_batched, random_lambda)

			CE, JA = train_one_epoch(inputs_batched, labels_batched, net, optimizer, epoch)

			running_loss += CE.item()
			seg_jac_running_loss += JA.item()
			
			itr += 1

			CE, JA = train_one_epoch(mixuped_input, mixuped_label, net, optimizer, epoch)

			mixup_running_loss += CE.item()
			mixup_seg_jac_running_loss += JA.item()

		i_batch = i_batch + 1
		print('\nepoch:%d/%d\tCE loss:%g\tJA loss:%g' % (epoch, cfg.TRAIN_EPOCHS, running_loss/i_batch, seg_jac_running_loss/i_batch))

		print('mixup: \tCE loss:%g\tJA loss:%g' % (mixup_running_loss/i_batch, mixup_seg_jac_running_loss/i_batch))

		#if (epoch) % 50 == 0:
		#	dataset.save_result_train_weight(dataset_list, cfg.MODEL_NAME)

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
		net.eval()
		with torch.no_grad():
			if epoch % 4 == 0:
				for i_batch, sample_batched in enumerate(test_dataloader):
					name_batched = sample_batched['name']
					row_batched = sample_batched['row']
					col_batched = sample_batched['col']

					[batch, channel, height, width] = sample_batched['image'].size()
					multi_avg = torch.zeros((batch, cfg.MODEL_NUM_CLASSES, height, width), dtype=torch.float32).to(1)
					for rate in cfg.TEST_MULTISCALE:
						inputs_batched = sample_batched['image_%f'%rate]
						_, predicts = net(inputs_batched)
						predicts = predicts.to(1)
						predicts_batched = predicts.clone()
						del predicts
			
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
						HD = max(directed_hausdorff(predict, gt)[0], directed_hausdorff(predict, gt)[0])
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
				if Dice_score > best_jacc:
					model_snapshot(net.state_dict(), new_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_%s_epoch%d_dice%.3f.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,epoch,Dice_score)),
                    	old_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_%s_epoch%d_dice%.3f.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,best_epoch,best_jacc)))
					best_jacc = Dice_score
					best_epoch = epoch

			if epoch % 50 == 0:
				dataset.save_result_train(result_list, cfg.MODEL_NAME)

#	save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_epoch%d_all.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,cfg.TRAIN_EPOCHS))		
#	torch.save(net.state_dict(),save_path)
	#if cfg.TRAIN_TBLOG:
	#	tblogger.close()
#	print('%s has been saved'%save_path)

def adjust_lr(optimizer, itr, max_itr):
	now_lr = cfg.TRAIN_LR * (1 - (itr/(max_itr+1)) ** cfg.TRAIN_POWER)
#	now_lr = np.max([cfg.TRAIN_LR*0.1, now_lr])
	optimizer.param_groups[0]['lr'] = now_lr
	optimizer.param_groups[1]['lr'] = 10*now_lr
	return now_lr

def get_params(model, key):
	for m in model.named_modules():
		if key == '1x':
			if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
				for p in m[1].parameters():
					yield p
		elif key == '10x':
			if 'backbone' not in m[0] and isinstance(m[1], nn.Conv2d):
				for p in m[1].parameters():
					yield p

if __name__ == '__main__':
	train_net()