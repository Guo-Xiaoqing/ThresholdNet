# ----------------------------------------
# Written by Yude Wang
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

def jaccard_loss(true, logits, eps=1e-7):
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
        probas = F.softmax(logits, dim=1)[:,1,:,:].to(1)
    true_1_hot = true_1_hot.type(logits.type())[:,1,:,:].to(1)
    dims = (0,) + tuple(range(2, true.ndimension()))
#    intersection = torch.sum(probas * true_1_hot, dims)
#    cardinality = torch.sum(probas + true_1_hot, dims)
    intersection = torch.sum(probas * true_1_hot, dim=(1,2))
    cardinality = torch.sum(probas + true_1_hot, dim=(1,2))
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

def model_snapshot(model, new_file=None, old_file=None):
    if os.path.exists(old_file) is True:
        os.remove(old_file) 
    torch.save(model, new_file)
    print('%s has been saved'%new_file)


def train_net():
#	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train', cfg.DATA_AUG)
	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train_cross'+cfg.DATA_CROSS, cfg.DATA_AUG)
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TRAIN_BATCHES, 
				shuffle=cfg.TRAIN_SHUFFLE, 
				num_workers=cfg.DATA_WORKERS,
				drop_last=True)

#	test_dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test')
	test_dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test_cross'+cfg.DATA_CROSS)
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
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape==net_dict[k].shape)}
		net_dict.update(pretrained_dict)
		net.load_state_dict(net_dict)
		# net.load_state_dict(torch.load(cfg.TRAIN_CKPT),False)
	
	criterion = nn.CrossEntropyLoss(ignore_index=255)
	optimizer = optim.SGD(
		params = [
			{'params': get_params(net.module,key='1x'), 'lr': cfg.TRAIN_LR},
			{'params': get_params(net.module,key='10x'), 'lr': 10*cfg.TRAIN_LR}
		],
		momentum=cfg.TRAIN_MOMENTUM
	)
	'''optimizer = optim.Adam(
		params = [
			{'params': get_params(net.module,key='1x'), 'lr': cfg.TRAIN_LR},
			{'params': get_params(net.module,key='10x'), 'lr': 10*cfg.TRAIN_LR}
		]
	)	'''
	#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN_LR_MST, gamma=cfg.TRAIN_LR_GAMMA, last_epoch=-1)
	itr = cfg.TRAIN_MINEPOCH * len(dataloader)
	max_itr = cfg.TRAIN_EPOCHS*len(dataloader)
	best_jacc = 0.
	best_epoch = 0
	#tblogger = SummaryWriter(cfg.LOG_DIR)
	#net.train()
	for epoch in range(cfg.TRAIN_MINEPOCH, cfg.TRAIN_EPOCHS):
		running_loss = 0.0
		running_dice_loss = 0.0
		net.train()
		#scheduler.step()
		#now_lr = scheduler.get_lr()
		for i_batch, sample_batched in enumerate(dataloader):
			now_lr = adjust_lr(optimizer, itr, max_itr)
			inputs_batched, labels_batched = sample_batched['image'], sample_batched['segmentation']
			optimizer.zero_grad()
			labels_batched = labels_batched.long().to(1)
			#0foreground_pix = (torch.sum(labels_batched!=0).float()+1)/(cfg.DATA_RESCALE**2*cfg.TRAIN_BATCHES)
			_, predicts_batched = net(inputs_batched)
			predicts_batched = predicts_batched.to(1) 
			loss = criterion(predicts_batched, labels_batched)
			dice_loss = jaccard_loss(labels_batched, predicts_batched, eps=1e-7)

			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			running_dice_loss += dice_loss.item()
			
		#	print('epoch:%d/%d\tbatch:%d/%d\titr:%d\tlr:%g\tloss:%g ' % 
			#	(epoch, cfg.TRAIN_EPOCHS, i_batch, dataset.__len__()//cfg.TRAIN_BATCHES,
			#	itr+1, now_lr, loss.item()))
			'''if cfg.TRAIN_TBLOG and itr%100 == 0:
				#inputs = np.array((inputs_batched[0]*128+128).numpy().transpose((1,2,0)),dtype=np.uint8)
				#inputs = inputs_batched.numpy()[0]
				inputs = inputs_batched.numpy()[0]/2.0 + 0.5
				labels = labels_batched[0].cpu().numpy()
				labels_color = dataset.label2colormap(labels).transpose((2,0,1))
				predicts = torch.argmax(predicts_batched[0],dim=0).cpu().numpy()
				predicts_color = dataset.label2colormap(predicts).transpose((2,0,1))
				pix_acc = np.sum(labels==predicts)/(cfg.DATA_RESCALE**2)

				tblogger.add_scalar('loss', running_loss, itr)
				tblogger.add_scalar('lr', now_lr, itr)
				tblogger.add_scalar('pixel acc', pix_acc, itr)
				tblogger.add_image('Input', inputs, itr)
				tblogger.add_image('Label', labels_color, itr)
				tblogger.add_image('Output', predicts_color, itr)'''
			
			#if itr % 5000 == 0:
			#	save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_itr%d.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,itr))
			#	torch.save(net.state_dict(), save_path)
			#	print('%s has been saved'%save_path)

			itr += 1

		print('epoch:%d/%d\tmean loss:%g\tmean dice loss:%g \n' %  (epoch, cfg.TRAIN_EPOCHS, running_loss/i_batch, running_dice_loss/i_batch))

		#### start testing now
		#IoU_array = np.zeros(IoU.shape)
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
			if epoch % 10 == 0:
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

			#if epoch % 50 == 0:
			#	dataset.save_result_train(result_list, cfg.MODEL_NAME)



#	save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_epoch%d_all.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,cfg.TRAIN_EPOCHS))		
#	torch.save(net.state_dict(),save_path)
	#if cfg.TRAIN_TBLOG:
	#	tblogger.close()
	print('%s has been saved'%save_path)

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

if __name__ == '__main__':
	train_net()


