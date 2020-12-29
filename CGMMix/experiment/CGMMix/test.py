# ----------------------------------------
# Written by Xiaoqing Guo
# ----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2
import os

from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from net.sync_batchnorm.replicate import patch_replication_callback

from torch.utils.data import DataLoader

def make_one_hot(labels):
	target = torch.eye(cfg.MODEL_NUM_CLASSES)[labels]
	gt_1_hot = target.permute(0, 3, 1, 2).float().to(0)
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
	feature = feature.to(0)
	gt = gt.to(0)
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
	Con_simi_gt = torch.from_numpy(Con_simi_gt.cpu().data.numpy()).to(0)
	return cosine_similarity1, cosine_similarity2, mixup_cosine_similarity, Con_simi_gt

def test_net():
	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train')
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TEST_BATCHES, 
				shuffle=False, 
				num_workers=cfg.DATA_WORKERS)

	dataset_mixup = generate_dataset(cfg.DATA_NAME, cfg, 'train')
	dataloader_mixup = DataLoader(dataset_mixup, 
				batch_size=cfg.TEST_BATCHES, 
				shuffle=True, 
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
		for i_batch, (sample_batched, mixup_batched) in enumerate(zip(dataloader, dataloader_mixup)):
			name_batched = sample_batched['name']
			row_batched = sample_batched['row']
			col_batched = sample_batched['col']

			[batch, channel, height, width] = sample_batched['image'].size()
			multi_avg = torch.zeros((batch, cfg.MODEL_NUM_CLASSES, height, width), dtype=torch.float32).to(0)

			inputs_batched1, labels_batched_cpu1 = sample_batched['image'], sample_batched['segmentation']
			inputs_batched2, labels_batched_cpu2 = mixup_batched['image'], mixup_batched['segmentation']
			labels_batched1 = labels_batched_cpu1.long().to(0)
			labels_batched2 = labels_batched_cpu2.long().to(0)
			feature1, predicts1 = net(inputs_batched1)
			feature2, predicts2 = net(inputs_batched2)
			predicts_batched1 = predicts1.clone()
			predicts_batched2 = predicts2.clone()
			deep_feature1 = feature1.clone()
			deep_feature2 = feature2.clone()
			del predicts1
			del predicts2
			del feature1
			del feature2

			alpha = cfg.Alpha
			random_lambda = np.random.beta(alpha, alpha)
			random_lambda = 0.5
			Softmax_predicts_batched = nn.Softmax(dim=1)(predicts_batched1)
			cosSim1 = Softmax_predicts_batched.mul(make_one_hot(labels_batched1)).sum(dim=1)
			Softmax_predicts_batched = nn.Softmax(dim=1)(predicts_batched2)
			cosSim2 = Softmax_predicts_batched.mul(make_one_hot(labels_batched2)).sum(dim=1)
			## Confidence1, Confidence2, Confidence_gt of mixup image
			cosSim1 = torch.from_numpy(cosSim1.cpu().data.numpy()).to(0)
			cosSim2 = torch.from_numpy(cosSim2.cpu().data.numpy()).to(0)
			cosSimilarity_gt = input_mixup(cosSim1, cosSim2, random_lambda)

			## Segmentation_gt of mixup image
			mixup_input = input_mixup(inputs_batched1, inputs_batched2, random_lambda)
			mixup_label = label_mixup(labels_batched1, labels_batched2, cosSim1, cosSim2, random_lambda)
			mixup_label_long = mixup_label.long().to(0)
			mixup_feature, mixup_predicts = net(mixup_input)
			mixup_deep_feature = mixup_feature.clone()
			mixup_predicts_batched = mixup_predicts.clone()
			del mixup_feature
			del mixup_predicts
			## Prediction of mixup image
			result = torch.argmax(mixup_predicts_batched, dim=1).cpu().numpy().astype(np.uint8)
			## Confidence of mixup image
			Softmax_predicts_batched = nn.Softmax(dim=1)(mixup_predicts_batched)
			cosSimilarity = Softmax_predicts_batched.mul(make_one_hot(mixup_label_long)).sum(dim=1)

			imgresize = torch.nn.UpsamplingBilinear2d(size=(int(cfg.DATA_RESCALE/4),int(cfg.DATA_RESCALE/4)))
			feature_label1 = imgresize(torch.unsqueeze(labels_batched_cpu1.to(0), 1)) 
			feature_label2 = imgresize(torch.unsqueeze(labels_batched_cpu2.to(0), 1))
			feature_mixuplabel = imgresize(torch.unsqueeze(mixup_label.to(0), 1))
			cosSim1_small = imgresize(torch.unsqueeze(cosSim1, 1))
			cosSim2_small = imgresize(torch.unsqueeze(cosSim2, 1))
			## CosineSimilarityMap && CosineSimilarityMap_gt of mixup image
			cosine_similarity1, cosine_similarity2, mixup_cosine_similarity, Con_simi_gt = ManifoldMixup_loss(deep_feature1, deep_feature2, mixup_deep_feature, feature_label1, feature_label2, feature_mixuplabel, cosSim1_small, cosSim2_small, random_lambda)
			
			#inputs_batched1, inputs_batched2, mixup_input, Softmax_predicts_batched, mixup_label
			#cosine_similarity1, cosine_similarity2, mixup_cosine_similarity, Con_simi_gt,
			#cosSim1, cosSim2, cosSimilarity, cosSimilarity_gt, 
			for i in range(batch):
				in1 = inputs_batched1[i,:,:,:].cpu().data.numpy()					
				in2 = inputs_batched2[i,:,:,:].cpu().data.numpy()					
				mixin = mixup_input[i,:,:,:].cpu().data.numpy()					
				mixpred = Softmax_predicts_batched[i,1,:,:].cpu().data.numpy()						
				mixlabel = mixup_label[i,:,:].cpu().data.numpy()

				cos1 = np.uint8(cosine_similarity1[i,0,:,:].cpu().data.numpy()*255)
				cos1 = cv2.applyColorMap(cos1, cv2.COLORMAP_JET)
				cos2 = np.uint8(cosine_similarity2[i,0,:,:].cpu().data.numpy()*255)
				cos2 = cv2.applyColorMap(cos2, cv2.COLORMAP_JET)
				mixcos = np.uint8(mixup_cosine_similarity[i,0,:,:].cpu().data.numpy()*255)
				mixcos = cv2.applyColorMap(mixcos, cv2.COLORMAP_JET)
				mixcosGT = np.uint8(Con_simi_gt[i,0,:,:].cpu().data.numpy()*255)
				mixcosGT = cv2.applyColorMap(mixcosGT, cv2.COLORMAP_JET)

				conf1 = np.uint8(cosSim1[i,:,:].cpu().data.numpy()*255)
				conf1 = cv2.applyColorMap(conf1, cv2.COLORMAP_JET)
				conf2 = np.uint8(cosSim2[i,:,:].cpu().data.numpy()*255)
				conf2 = cv2.applyColorMap(conf2, cv2.COLORMAP_JET)
				mixconf = np.uint8(cosSimilarity[i,:,:].cpu().data.numpy()*255)
				mixconf = cv2.applyColorMap(mixconf, cv2.COLORMAP_JET)
				mixconfGT = np.uint8(cosSimilarity_gt[i,:,:].cpu().data.numpy()*255)	
				mixconfGT = cv2.applyColorMap(mixconfGT, cv2.COLORMAP_JET)
			#	print(in1.shape, mixin.shape, mixpred.shape, mixlabel.shape, cos1.shape, mixcos.shape, mixcosGT.shape, conf1.shape, mixconf.shape, mixconfGT.shape)	


				result_list.append({'in1':np.uint8(in1*255), 'in2':np.uint8(in2*255), 'mixin':np.uint8(mixin*255), 'mixpred':np.uint8(mixpred*255), 'mixlabel':np.uint8(mixlabel*255), 
									'cos1':cos1, 'cos2':cos2, 'mixcos':mixcos, 'mixcosGT':mixcosGT, 
									'conf1':conf1, 'conf2':conf2, 'mixconf':mixconf, 'mixconfGT':mixconfGT, 'name':name_batched[i]})

		root_dir = os.path.join(cfg.ROOT_DIR,'data')
		rst_dir = os.path.join(root_dir,'results','CVC','Train')
		if not os.path.exists(rst_dir):
			os.makedirs(rst_dir)
		for sample in result_list:
			name = sample['name'].split()[0]
			folder_path = os.path.join(rst_dir, '%s'%name)
			if not os.path.exists(folder_path):
				os.makedirs(folder_path)
			in1 = sample['in1'].transpose((1,2,0))
			in2 = sample['in2'].transpose((1,2,0))
			mixin = sample['mixin'].transpose((1,2,0))
			mixpred = np.stack([sample['mixpred'], sample['mixpred'], sample['mixpred']]).transpose((1,2,0))
			mixlabel = np.stack([sample['mixlabel'], sample['mixlabel'], sample['mixlabel']]).transpose((1,2,0))

		#	cos1 = np.stack([sample['cos1'], sample['cos1'], sample['cos1']]).transpose((1,2,0))
		#	cos2 = np.stack([sample['cos2'], sample['cos2'], sample['cos2']]).transpose((1,2,0))
		#	mixcos = np.stack([sample['mixcos'], sample['mixcos'], sample['mixcos']]).transpose((1,2,0))
		#	mixcosGT = np.stack([sample['mixcosGT'], sample['mixcosGT'], sample['mixcosGT']]).transpose((1,2,0))
			cos1 = sample['cos1']
			cos2 = sample['cos2']	
			mixcos = sample['mixcos']	
			mixcosGT = sample['mixcosGT']	

		#	conf1 = np.stack([sample['conf1'], sample['conf1'], sample['conf1']]).transpose((1,2,0))
		#	conf2 = np.stack([sample['conf2'], sample['conf2'], sample['conf2']]).transpose((1,2,0))
		#	mixconf = np.stack([sample['mixconf'], sample['mixconf'], sample['mixconf']]).transpose((1,2,0))
		#	mixconfGT = np.stack([sample['mixconfGT'], sample['mixconfGT'], sample['mixconfGT']]).transpose((1,2,0))
			conf1 = sample['conf1']	
			conf2 = sample['conf2']		
			mixconf = sample['mixconf']		
			mixconfGT = sample['mixconfGT']		

			in1 = in1[...,::-1]
			in2 = in2[...,::-1]
			mixin = mixin[...,::-1]

        #    img_RGB=np.zeros([inputs_batched.shape[0]*3, inputs_batched.shape[1]*3, inputs_batched.shape[2]])
        #    img_RGB[:,:,0] = img[:,:,2]
        #    img_RGB[:,:,1] = img[:,:,1]
        #    img_RGB[:,:,2] = img[:,:,0]
			cv2.imwrite(os.path.join(folder_path, 'in1.png'), in1)
			cv2.imwrite(os.path.join(folder_path, 'in2.png'), in2)
			cv2.imwrite(os.path.join(folder_path, 'mixin.png'), mixin)
			cv2.imwrite(os.path.join(folder_path, 'mixpred.png'), mixpred)
			cv2.imwrite(os.path.join(folder_path, 'mixlabel.png'), mixlabel)

			cv2.imwrite(os.path.join(folder_path, 'cos1.png'), cos1)
			cv2.imwrite(os.path.join(folder_path, 'cos2.png'), cos2)
			cv2.imwrite(os.path.join(folder_path, 'mixcos.png'), mixcos)
			cv2.imwrite(os.path.join(folder_path, 'mixcosGT.png'), mixcosGT)

			cv2.imwrite(os.path.join(folder_path, 'conf1.png'), conf1)
			cv2.imwrite(os.path.join(folder_path, 'conf2.png'), conf2)
			cv2.imwrite(os.path.join(folder_path, 'mixconf.png'), mixconf)
			cv2.imwrite(os.path.join(folder_path, 'mixconfGT.png'), mixconfGT)
            # print('[%d/%d] %s saved'%(i,len(result_list),file_path))

if __name__ == '__main__':
	test_net()


