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

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
    #    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
    #    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
    #    return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2)
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2)
        return self.TVLoss_weight*(h_tv[:,:,:,1:]+w_tv[:,:,1:,:]).sum(dim=1)

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def Overlap_loss(true, logits, eps=1e-7):
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
        probas = logits[:,1,:,:].to(1)
    true_1_hot = true_1_hot.type(logits.type())[:,1,:,:].to(1)
    dims = (0,) + tuple(range(2, true.ndimension()))
#    intersection = torch.sum(probas * true_1_hot, dims)
#    cardinality = torch.sum(probas + true_1_hot, dims)
    intersection = torch.sum(probas * true_1_hot, dim=(1,2))
    cardinality = torch.sum(probas + true_1_hot, dim=(1,2))
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    dice_loss = (2*intersection / (cardinality)).mean()
    return (1 - jacc_loss)

def SoftCrossEntropy(inputs, target, reduction='average'):
    log_likelihood = -torch.log(inputs)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.mean(torch.mul(log_likelihood, target))
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss

def MSE(inputs, target, reduction='average'):
	MSE = torch.nn.MSELoss(reduce=False, size_average=False)
	loss = MSE(inputs, target).sum(dim=1)
	if reduction == 'average':
		loss = torch.mean(loss)
	else:
		loss = torch.sum(loss)
	return loss

def model_snapshot(model, new_file=None, old_file=None):
    if os.path.exists(old_file) is True:
        os.remove(old_file) 
    torch.save(model, new_file)
    print('%s has been saved'%new_file)

def make_one_hot(labels):
	target = torch.eye(cfg.MODEL_NUM_CLASSES)[labels]
	gt_1_hot = target.permute(0, 3, 1, 2).float().to(1)
	return gt_1_hot

def train_one_batch(inputs_batched, labels_batched, net, seg_optimizer, thr_optimizer, list=None, phase='original'):
	criterion = nn.CrossEntropyLoss(ignore_index=255)
	grad_criterion = TVLoss()
	cos = nn.CosineSimilarity(dim=1, eps=1e-6)
	###############################################
    #### train segmenatation branch && backbone####
	###############################################
	_, predicts_batched, threshold_batched = net(inputs_batched)
	predicts_batched = predicts_batched.to(1)  
	threshold_batched = threshold_batched.to(1) 
	Softmax_predicts_batched = nn.Softmax(dim=1)(predicts_batched)
	mask = nn.Sigmoid()(50*(Softmax_predicts_batched-threshold_batched))
	loss = criterion(predicts_batched, labels_batched)
	seg_jac_loss = Overlap_loss(labels_batched, mask, eps=1e-7)
	seg_loss = (loss + seg_jac_loss)

	seg_optimizer.zero_grad()
	seg_loss.backward()
	seg_optimizer.step()

	################################
	#### train threshold branch ####
	################################
	_, predicts_batched, threshold_batched = net(inputs_batched)
	predicts_batched = predicts_batched.to(1)  
	threshold_batched = threshold_batched.to(1) 
	Softmax_predicts_batched = nn.Softmax(dim=1)(predicts_batched)
	mask = nn.Sigmoid()(50*(Softmax_predicts_batched-threshold_batched))
	dice_loss = Overlap_loss(labels_batched, mask, eps=1e-7)

	margin = cfg.Threshold_margin
	gt_1_hot = make_one_hot(labels_batched)
	threshold_gt = torch.where(gt_1_hot == 1, Softmax_predicts_batched-margin, Softmax_predicts_batched+margin)
#	threshold_gt = torch.clamp(threshold_gt, 0.0, 1.0)
	sup_loss = SoftCrossEntropy(threshold_batched, threshold_gt)
	input_edge_aware = torch.exp(-grad_criterion(inputs_batched)).to(1)
	grad_loss = grad_criterion(threshold_batched).mul(input_edge_aware).mean()
	thres_loss = (dice_loss+sup_loss+grad_loss)
				
	thr_optimizer.zero_grad()
	thres_loss.backward()
	thr_optimizer.step()

	return loss, seg_jac_loss, dice_loss, grad_loss, sup_loss


def train_net():
#	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train_cross'+cfg.DATA_CROSS, cfg.DATA_AUG)
	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train'+cfg.DATA_CROSS, cfg.DATA_AUG)
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TRAIN_BATCHES, 
				shuffle=cfg.TRAIN_SHUFFLE, 
				num_workers=cfg.DATA_WORKERS,
				drop_last=True)

#	test_dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test_cross'+cfg.DATA_CROSS)
	test_dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test'+cfg.DATA_CROSS)
	test_dataloader = DataLoader(test_dataset, 
				batch_size=cfg.TEST_BATCHES, 
				shuffle=False, 
				num_workers=cfg.DATA_WORKERS)
	
	net = generate_net(cfg)
	#if cfg.TRAIN_TBLOG:
	#	from tensorboardX import SummaryWriter
		# Set the Tensorboard logger
		#tblogger = SummaryWriter(cfg.LOG_DIR)

	print('Use %d GPU'%cfg.TRAIN_GPUS)
	device = torch.device(0)
	if cfg.TRAIN_GPUS > 1:
		net = nn.DataParallel(net)
		patch_replication_callback(net)
	net.to(device)		

	if cfg.TRAIN_CKPT:
		pretrained_dict = torch.load(cfg.TRAIN_CKPT)
		net_dict = net.state_dict()
	#	for i, p in enumerate(net_dict):
 	#	   print(i, p)
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape==net_dict[k].shape)}
		net_dict.update(pretrained_dict)
		net.load_state_dict(net_dict)
		# net.load_state_dict(torch.load(cfg.TRAIN_CKPT),False)
#	for i, para in enumerate(net.named_parameters()):
#		(name, param) = para
#		print(i, name)	

	threshold_dict = []
	segment_dict = []
	backbone_dict = []
	for i, para in enumerate(net.parameters()):
		if i <= 47 and i >= 38:
			threshold_dict.append(para)
		elif i < 38:
			segment_dict.append(para)
		else:
			backbone_dict.append(para)
#	print(i)

	thr_optimizer = optim.SGD(threshold_dict, lr=10*cfg.TRAIN_LR, momentum=cfg.TRAIN_MOMENTUM)
	seg_optimizer = optim.SGD(
		params = [
            {'params': backbone_dict, 'lr': cfg.TRAIN_LR},
            {'params': segment_dict, 'lr': 10*cfg.TRAIN_LR}
        ],
		momentum=cfg.TRAIN_MOMENTUM)
	
	'''optimizer = optim.SGD(
		params = [
			{'params': get_params(net.module,key='1x'), 'lr': cfg.TRAIN_LR},
			{'params': get_params(net.module,key='10x'), 'lr': 10*cfg.TRAIN_LR}
		],
		momentum=cfg.TRAIN_MOMENTUM
	)'''
	#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN_LR_MST, gamma=cfg.TRAIN_LR_GAMMA, last_epoch=-1)
	itr = cfg.TRAIN_MINEPOCH * len(dataloader)
	max_itr = cfg.TRAIN_EPOCHS_lr * len(dataloader)
	#tblogger = SummaryWriter(cfg.LOG_DIR)
	#net.train()
	best_jacc = 0.
	best_epoch = 0
	for epoch in range(cfg.TRAIN_MINEPOCH, cfg.TRAIN_EPOCHS):			
		running_loss = 0.0
		seg_jac_running_loss = 0.0
		dice_running_loss = 0.0
		grad_running_loss = 0.0
		average_running_loss = 0.0
		mixup_running_loss = 0.0
		mixup_seg_jac_running_loss = 0.0
		mixup_dice_running_loss = 0.0
		mixup_grad_running_loss = 0.0
		mixup_average_running_loss = 0.0
		dataset_list = []
		net.train()
		#scheduler.step()
		#now_lr = scheduler.get_lr()
		for i_batch, sample_batched in enumerate(dataloader):
			now_lr = adjust_lr(seg_optimizer, itr, max_itr)
			name_batched = sample_batched['name']
			inputs_batched1, labels_batched_cpu1 = sample_batched['image'], sample_batched['segmentation']
			labels_batched1 = labels_batched_cpu1.long().to(1) 

			loss, seg_jac_loss, dice_loss, grad_loss, sup_loss = train_one_batch(inputs_batched1, labels_batched1, net, seg_optimizer, thr_optimizer)
			
			running_loss += loss.item()
			seg_jac_running_loss += seg_jac_loss.item()
			dice_running_loss += dice_loss.item()
			grad_running_loss += grad_loss.item()
			average_running_loss += sup_loss.item()

		i_batch = i_batch + 1
		print('epoch:%d/%d\tSegCE loss:%g \tSegJaccard loss:%g \tThrJaccard loss:%g \tThrGrad loss:%g \tThrSup loss:%g' %  (epoch, cfg.TRAIN_EPOCHS, 
					running_loss/i_batch, seg_jac_running_loss/i_batch, dice_running_loss/i_batch, grad_running_loss/i_batch, average_running_loss/i_batch))

		#### start testing now
		if (epoch) % 2 == 0:
			Dice_score, IoUP = test_one_epoch(test_dataset, test_dataloader, net, epoch)
			if Dice_score > best_jacc:
				model_snapshot(net.state_dict(), new_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_%s_epoch%d_dice%.3f.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,epoch,Dice_score)),
                	old_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_%s_epoch%d_dice%.3f.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,best_epoch,best_jacc)))
				best_jacc = Dice_score
				best_epoch = epoch

	#save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_epoch%d_all.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,cfg.TRAIN_EPOCHS))		
	#torch.save(net.state_dict(),save_path)
	#if cfg.TRAIN_TBLOG:
	#	tblogger.close()
	#print('%s has been saved'%save_path)

def adjust_lr(optimizer, itr, max_itr):
	now_lr = cfg.TRAIN_LR * (1 - (itr/(max_itr+1)) ** cfg.TRAIN_POWER)
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
			multi_avg = torch.zeros((batch, cfg.MODEL_NUM_CLASSES, height, width), dtype=torch.float32).to(1)
			labels_batched = sample_batched['segmentation'].cpu().numpy()
			for rate in cfg.TEST_MULTISCALE:
				inputs_batched = sample_batched['image_%f'%rate]
				_, predicts, threshold = net(inputs_batched)
				predicts = predicts.to(1) 
				threshold = threshold.to(1)
				predicts_batched = predicts.clone()
				threshold_batched = threshold.clone()
				del predicts
				del threshold
				if cfg.TEST_FLIP:
					inputs_batched_flip = torch.flip(inputs_batched,[3]) 
					_, predicts_flip, threshold_flip = net(inputs_batched_flip)
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
			multi_avg = nn.Softmax(dim=1)(multi_avg)
			multi_avg = multi_avg - threshold_batched
			result = torch.argmax(multi_avg, dim=1).cpu().numpy().astype(np.uint8)
			threshold = threshold_batched.cpu().numpy()
			del threshold_batched

			for i in range(batch):
				row = row_batched[i]
				col = col_batched[i]
				p = result[i,:,:]					
				l = labels_batched[i,:,:]
				thres = threshold[i,1,:,:]
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
				Prec = TP/(P+1e-10)
				Spe = TN/(P-TP+TN)
				Rec = TP/T
				DICE = 2*TP/(T+P)
				IoU = TP/(T+P-TP)
			#	HD = max(directed_hausdorff(predict, gt)[0], directed_hausdorff(predict, gt)[0])
			#	HD = 2*Prec*Rec/(Rec+Prec+1e-10)
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
				result_list.append({'predict':np.uint8(p*255), 'threshold':np.uint8(thres*255), 'label':np.uint8(l*255), 'IoU':IoU, 'name':name_batched[i]})

		Acc_score = Acc_array*100/sample_num
		Prec_score = Prec_array*100/sample_num
		Spe_score = Spe_array*100/sample_num
		Rec_score = Rec_array*100/sample_num
		Dice_score = Dice_array*100/sample_num
		IoUP = IoU_array*100/sample_num
		HD_score = HD_array*100/sample_num
		print('%10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%\n'%('Acc',Acc_score,'Sen',Rec_score,'Spe',Spe_score,'Prec',Prec_score,'Dice',Dice_score,'Jac',IoUP,'F2',HD_score))
		if epoch % 50 == 0:
			dataset.save_result_train_thres(result_list, cfg.MODEL_NAME)

		return Dice_score, IoUP

if __name__ == '__main__':
	train_net()