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

def input_mixup(input1, input2, lambda1):
	lambda2 = torch.from_numpy(np.array(1.))-lambda1
	mixup_input = input1.mul(lambda1)+ input2.mul(lambda2)
	return mixup_input

def label_mixup(labels1, labels2, gtc1, gtc2, lambda1):
	gtc1 = torch.unsqueeze(gtc1, 1)
	gtc2 = torch.unsqueeze(gtc2, 1)
	lambda2 = torch.from_numpy(np.array(1.))-lambda1
	labels1_1_hot = make_one_hot(labels1)
	labels2_1_hot = make_one_hot(labels2)
	
	mixup_label = labels1_1_hot.mul(gtc1).mul(lambda1)+ labels2_1_hot.mul(gtc2).mul(lambda2)
	mixup_label_margin = cfg.Mixup_label_margin
	mixup_label[:,1,:,:] = torch.where(mixup_label[:,1,:,:] >= mixup_label_margin, torch.ones_like(labels1), torch.zeros_like(labels1))
	mixup_label = torch.from_numpy(mixup_label[:,1,:,:].cpu().data.numpy())
	return mixup_label

def feature_cosine_similarity(feature, gt):
	### inputs are feature maps and segmentation ground truth, 
	### we should first calculate the mean features in foreground area,
	### then compute the cosine similarity map.
	feature = feature.to(1)
	gt = gt.to(1)
	fore_feature_center = torch.sum(feature.mul(gt), dim=(0,2,3)).div(torch.sum(gt))
	fore_feature_mean = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(fore_feature_center, 0), 2), 3)
	fore_feature_mean_len = torch.norm(fore_feature_mean, dim=1) 
	fore_feature_len = torch.norm(feature, dim=1) 
	nominator = feature.mul(fore_feature_mean)
	denominator = torch.unsqueeze(fore_feature_len.mul(fore_feature_mean_len), 1)
	cosine_similarity = torch.sum(nominator, dim=(1), keepdim=True)/denominator
	return cosine_similarity

def ManifoldMixup_loss(feature1, feature2, mixup_feature, gt1, gt2, mixup_gt, gtc1, gtc2, random_lambda):
	### feature1, gt1, gtc1 are features, segmentation ground truth and segmentation confidence map of the first image;
	### feature2, gt2, gtc2 are features, segmentation ground truth and segmentation confidence map of the second image;
	### mixup_feature, mixup_gt are features, segmentation ground truth of the mixup image
	### random_lambda is the parameter in mixup
	mixup_cosine_similarity = feature_cosine_similarity(mixup_feature, mixup_gt)
	
	mixup_label_margin = cfg.Mixup_label_margin
	cosine_similarity1 = feature_cosine_similarity(feature1, gt1)
	cosine_similarity2 = feature_cosine_similarity(feature2, gt2)

	Con_simi_gt1 = torch.where(random_lambda*gt1*gtc1 > mixup_label_margin, torch.ones_like(gt1), torch.zeros_like(gt1))
	Con_simi_gt2 = torch.where((1.-random_lambda)*gt2*gtc2 > mixup_label_margin, torch.ones_like(gt1), torch.zeros_like(gt1))
	Con_simi_gtinterf = Con_simi_gt1.mul(Con_simi_gt2)
	Con_simi_gtinterb = torch.where(Con_simi_gt1 + Con_simi_gt2 < 0.1, torch.ones_like(gt1), torch.zeros_like(gt1))
	Con_simi_gt1 = Con_simi_gt1 - Con_simi_gtinterf
	Con_simi_gt2 = Con_simi_gt2 - Con_simi_gtinterf
	Con_simi_gtinter = Con_simi_gtinterf + Con_simi_gtinterb
	Con_simi_gt1 = cosine_similarity1.mul(Con_simi_gt1)
	Con_simi_gt2 = cosine_similarity2.mul(Con_simi_gt2)
	Con_simi_gtinter = input_mixup(cosine_similarity1, cosine_similarity2, random_lambda).mul(Con_simi_gtinter)
	Con_simi_gt = Con_simi_gt1 + Con_simi_gt2 + Con_simi_gtinter
	Con_simi_gt = torch.from_numpy(Con_simi_gt.cpu().data.numpy()).to(1)

#	intersection = torch.sum(mixup_cosine_similarity * Con_simi_gt, dim=(1,2,3))
#	cardinality = torch.sum(mixup_cosine_similarity + Con_simi_gt, dim=(1,2,3))
#	union = cardinality - intersection
#	MM_loss = 1. - (intersection / (union + 1e-6)).mean()
#	MM_loss = 1. - (2*intersection / cardinality).mean()

#	MM_loss = nn.BCELoss()(mixup_cosine_similarity, Con_simi_gt)

	MSE = torch.nn.MSELoss(reduce=False, size_average=False)
	MM_loss = MSE(mixup_cosine_similarity, Con_simi_gt)
	MM_loss = torch.mean(MM_loss, dim=(1,2,3)).sum()
	mixup_cosine_similarity = torch.from_numpy(mixup_cosine_similarity.cpu().data.numpy()).to(1)
	return mixup_cosine_similarity, Con_simi_gt, MM_loss

def train_one_batch(inputs_batched, labels_batched, net, seg_optimizer, thr_optimizer, list=None, phase='original'):
	criterion = nn.CrossEntropyLoss(ignore_index=255)
	grad_criterion = TVLoss()
	cos = nn.CosineSimilarity(dim=1, eps=1e-6)
	###############################################
    #### train segmenatation branch && backbone####
	###############################################
	feature_batched, predicts_batched = net(inputs_batched)
	feature_batched = feature_batched.to(1)  
	predicts_batched = predicts_batched.to(1)  
	feature = torch.from_numpy(feature_batched.cpu().data.numpy()).to(1)
	Softmax_predicts_batched = nn.Softmax(dim=1)(predicts_batched)
	loss = criterion(predicts_batched, labels_batched)
	seg_jac_loss = Overlap_loss(labels_batched, Softmax_predicts_batched, eps=1e-7)

	sample_cosSim = Softmax_predicts_batched.mul(make_one_hot(labels_batched)).sum(dim=1)
	sample_cosSim_cpu = sample_cosSim.cpu().data.numpy()

	if phase == 'mixup':
		######### ManifoldMixuploss (MFMC loss)#########
		_, _, MM_consistent_loss = ManifoldMixup_loss(list['deep_feature1'], list['deep_feature2'], feature_batched, 
						list['feature_label1'], list['feature_label2'], list['feature_mixup_label'], 
						list['cosSim1'], list['cosSim2'], list['random_lambda'])
		MM_consistent_loss = cfg.weight_ManifoldMixuploss * MM_consistent_loss
		
		######### Similoss (MCMC loss)#########
		MSE = torch.nn.MSELoss(reduce=False, size_average=False)
		Simi_consistent_loss = MSE(sample_cosSim, list['cosSimilarity'])
		Simi_consistent_loss = torch.mean(Simi_consistent_loss, dim=(1,2)).sum()
		Simi_consistent_loss = cfg.weight_Similoss * Simi_consistent_loss
	#	Simi_consistent_loss = cfg.weight_Similoss * MSE(sample_cosSim, list['cosSimilarity']).mean()

		seg_loss = (loss + MM_consistent_loss + Simi_consistent_loss)
	else:
		seg_loss = (loss)

	seg_optimizer.zero_grad()
	seg_loss.backward()
	seg_optimizer.step()

	if phase == 'mixup':
    	#### train segmenatation branch ####
		######### MixupFeatureUpdate #########
		mixup_deep_feature = input_mixup(list['deep_feature1'].to(1), list['deep_feature2'].to(1), list['random_lambda'])
		_, mixup_feature_predicts = net(inputs_batched, feature_cat=mixup_deep_feature, phase=phase)
		mixup_deep_feature = torch.from_numpy(mixup_deep_feature.cpu().data.numpy()).to(1)
		mixup_feature_predicts = mixup_feature_predicts.to(1)  
		Softmax_predicts_batched = nn.Softmax(dim=1)(mixup_feature_predicts)
		mixup_feature_loss = criterion(mixup_feature_predicts, labels_batched)
		mixup_feature_jac_loss = Overlap_loss(labels_batched, Softmax_predicts_batched, eps=1e-7)
		mixup_feature_seg_loss = cfg.weight_MixupFeatureUpdate * (mixup_feature_loss)

		seg_optimizer.zero_grad()
		mixup_feature_seg_loss.backward()
		seg_optimizer.step()

	if phase == 'mixup':
		return loss, seg_jac_loss, MM_consistent_loss, Simi_consistent_loss, mixup_feature_seg_loss
	else:
		return loss, seg_jac_loss, sample_cosSim_cpu, feature


def train_net():
#	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train_cross'+cfg.DATA_CROSS, cfg.DATA_AUG)
	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train'+cfg.DATA_CROSS, cfg.DATA_AUG)
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TRAIN_BATCHES, 
				shuffle=cfg.TRAIN_SHUFFLE, 
				num_workers=cfg.DATA_WORKERS,
				drop_last=True)

#	dataset_mixup = generate_dataset(cfg.DATA_NAME, cfg, 'train_cross'+cfg.DATA_CROSS, cfg.DATA_AUG)
	dataset_mixup = generate_dataset(cfg.DATA_NAME, cfg, 'train'+cfg.DATA_CROSS, cfg.DATA_AUG)
	dataloader_mixup = DataLoader(dataset_mixup, 
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
		for i_batch, (sample_batched, mixup_batched) in enumerate(zip(dataloader, dataloader_mixup)):
			now_lr = adjust_lr(seg_optimizer, itr, max_itr)
			name_batched = sample_batched['name']
			inputs_batched1, labels_batched_cpu1 = sample_batched['image'], sample_batched['segmentation']
			inputs_batched2, labels_batched_cpu2 = mixup_batched['image'], mixup_batched['segmentation']
			labels_batched1 = labels_batched_cpu1.long().to(1) 
			labels_batched2 = labels_batched_cpu2.long().to(1) 

			loss, seg_jac_loss, cosSim1, deep_feature1 = train_one_batch(inputs_batched1, labels_batched1, net, seg_optimizer, thr_optimizer)
					
			running_loss += loss.item()
			seg_jac_running_loss += seg_jac_loss.item()

			##############################
			#### obtain mixup samples ####
			##############################
			if epoch >= cfg.Mixup_start_epoch:
				margin = cfg.Threshold_margin
				imgresize = torch.nn.UpsamplingBilinear2d(size=(int(cfg.DATA_RESCALE/4),int(cfg.DATA_RESCALE/4)))
				deep_feature2, predicts_batched2 = net(inputs_batched2)
				predicts_batched2 = predicts_batched2.to(1)  
				Softmax_predicts_batched = nn.Softmax(dim=1)(predicts_batched2)
				cosSim2 = Softmax_predicts_batched.mul(make_one_hot(labels_batched2)).sum(dim=1)
				cosSim2 = torch.from_numpy(cosSim2.cpu().data.numpy()).to(1)
				cosSim1 = torch.from_numpy(cosSim1).to(1)
				feature_label1 = imgresize(torch.unsqueeze(labels_batched_cpu1.to(1), 1)) 
				feature_label2 = imgresize(torch.unsqueeze(labels_batched_cpu2.to(1), 1))

				alpha = cfg.Alpha
				random_lambda = np.random.beta(alpha, alpha)
				mixup_input = input_mixup(inputs_batched1, inputs_batched2, random_lambda)
				mixup_label = label_mixup(labels_batched1, labels_batched2, cosSim1, cosSim2, random_lambda)
				feature_mixup_label = imgresize(torch.unsqueeze(mixup_label, 1))
				mixup_label = mixup_label.long().to(1) 
				cosSimilarity = input_mixup(cosSim1, cosSim2, random_lambda)
				cosSim1 = imgresize(torch.unsqueeze(cosSim1, 1))
				cosSim2 = imgresize(torch.unsqueeze(cosSim2, 1))
				### here cosSim all means the confidence map of segmentation branch

				mixup_list = {'deep_feature1':deep_feature1, 'deep_feature2':deep_feature2, 'feature_label1':feature_label1, 'feature_label2':feature_label1,
							'feature_mixup_label':feature_mixup_label, 'cosSim1':cosSim1, 'cosSim2':cosSim2, 'random_lambda':random_lambda, 'cosSimilarity':cosSimilarity}
				loss, seg_jac_loss, MM_consistent_loss, Simi_consistent_loss, mixup_feature_seg_loss = train_one_batch(mixup_input, mixup_label, net, seg_optimizer, thr_optimizer, list=mixup_list, phase='mixup')

				mixup_running_loss += loss.item()
				mixup_seg_jac_running_loss += seg_jac_loss.item()
				mixup_dice_running_loss += MM_consistent_loss.item()
				mixup_grad_running_loss += 10*Simi_consistent_loss.item()
				mixup_average_running_loss += 10*mixup_feature_seg_loss.item()	

			itr += 1

			'''if (epoch) % 50 == 0:
				[batch, channel, height, width] = mixup_input.size()
				for i in range(batch):
					mixup1 = inputs_batched[i,:,:,:].cpu().numpy()
					mixup2 = mixup_inputs_batched[i,:,:,:].cpu().numpy()
					mixup_ = mixup_input[i,:,:,:].cpu().numpy()
					mixup1_l = labels_batched[i,:,:].cpu().numpy()
					mixup2_l = mixup_labels_batched[i,:,:].cpu().numpy()
					mixup_l = mixup_label[i,1,:,:].cpu().numpy()
					cosSimi1 = Sample_cos[i,0,:,:].cpu().numpy()
					cosSimi2 = Mixup_cos[i,0,:,:].cpu().numpy()
					dataset_list.append({'inputs_batched':np.uint8(mixup1*255), 'mixup_inputs_batched':np.uint8(mixup2*255), 
			                    'mixup_input':np.uint8(mixup_*255), 'name':name_batched[i], 'mixup_label':np.uint8(mixup_l*255), 
								'inputs_labels':np.uint8(mixup1_l*255), 'mixup_labels_batched':np.uint8(mixup2_l*255),
								'cosSimi1':np.uint8(cosSimi1*255), 'cosSimi2':np.uint8(cosSimi2*255)})
		if (epoch) % 50 == 0:
			dataset.save_result_train_mixup(dataset_list, cfg.MODEL_NAME)'''
		i_batch = i_batch + 1
		print('epoch:%d/%d\tSegCE loss:%g \tSegJaccard loss:%g' %  (epoch, cfg.TRAIN_EPOCHS, 
					running_loss/i_batch, seg_jac_running_loss/i_batch))
		if epoch >= cfg.Mixup_start_epoch:
			print('Mixup:\tSegCE loss:%g \tSegJaccard loss:%g \tMFMC loss:%g \tMCMC loss:%g \tMMfeature loss:%g \n' %  (mixup_running_loss/i_batch, 
						mixup_seg_jac_running_loss/i_batch, mixup_dice_running_loss/i_batch, mixup_grad_running_loss/i_batch, mixup_average_running_loss/i_batch))

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
				_, predicts = net(inputs_batched)
				predicts = predicts.to(1) 
				predicts_batched = predicts.clone()
				del predicts
			
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
				result_list.append({'predict':np.uint8(p*255), 'threshold':np.uint8(p*255), 'label':np.uint8(l*255), 'IoU':IoU, 'name':name_batched[i]})

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