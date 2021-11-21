import torch
import pandas as pd 
import torch.nn.functional as F
import numpy as np

from results.trial_3.model_file import RNN_VAE
from data_proc.data_process import Peptide_dataset 
from torch.utils.data import DataLoader
import sys
from shutil import copyfile

from torch import nn


class model_usage():

	def __init__(self,model_file_folder,model_file):
		sys.path.insert(0,model_file_folder)
		import configure_file as Params
		from model_file import RNN_VAE 
		self.Params=Params
		self.model=RNN_VAE(**self.Params.RNN_VAE_params)
		self.model.load_state_dict(torch.load(model_file,map_location=Params.device))
		self.seq_prefix='YPEDILDKHLQRVIL'

	def sample_encoding(self,mean,std,batch):
		encodings=[]
		for i in range(len(mean)):
			val=np.expand_dims(np.random.normal(mean[i], std[i],batch),axis=1)
			if i ==0:
				encodings=val
			else: 
				encodings=np.concatenate((encodings,val),axis=1)
		return encodings


	def get_encoding(self,df,preprocess=False):
		DP=self.Params.dataset_params
		np.random.seed(2021)
		# print (df)
		dataset=Peptide_dataset(df,is_train=True,preprocess=preprocess,plot=False)
		TP=self.Params.training_params
		dataloader = DataLoader(dataset, batch_size=TP['train_batch_size'], shuffle=False)

		model=self.model
		with torch.no_grad():
			dataset_iter=iter(dataloader)
			encodings=[]
			for i in range(len(dataloader)):
				seq_token, gt_props = next(dataset_iter)
				(z_mu, z_logvar), (z, pred_props), dec_logits = model(seq_token)
				z=z.cpu().numpy()
				encodings.extend(z)
		return encodings 		

	def __get_training_encodings__(self):
		DP=self.Params.dataset_params
		np.random.seed(2021)
		df=pd.read_csv(DP['data_path'])
		df=df.sample(frac=1).reset_index(drop=True)
		max_seq_len=DP['max_seq_len']
		train_eval_ratio=DP['train_eval_ratio']
		N_seq=len(df)
		N_train=int(N_seq*train_eval_ratio)
		df_train=df[:N_train].sample(frac=1).reset_index(drop=True)
		train_dataset=Peptide_dataset(df_train,is_train=True,preprocess=True,plot=False)
		TP=self.Params.training_params
		train_dataloader = DataLoader(train_dataset, batch_size=TP['train_batch_size'], shuffle=TP['shuffle'])

		df_test=df[N_train:].sample(frac=1).reset_index(drop=True)
		test_dataset=Peptide_dataset(df_test,is_train=False,preprocess=True,plot=False)
		test_dataloader = DataLoader(test_dataset, batch_size=TP['test_batch_size'], shuffle=TP['shuffle'])
		model=self.model
		with torch.no_grad():
			dataset_iter=iter(train_dataloader)
			encodings=[]
			for i in range(len(train_dataloader)):
				seq_token, gt_props = next(dataset_iter)
				(z_mu, z_logvar), (z, pred_props), dec_logits = model(seq_token)
				z=z.cpu().numpy()
				encodings.extend(z)
		return encodings 


	def get_init_mean_std(self):	
		encodings=self.__get_training_encodings__()
		mean=np.mean(encodings,axis=0)
		std=np.std(encodings,axis=0)
		return mean,std 

	def get_encoding_PCs(self,n_components):
		from sklearn.decomposition import PCA
		pca = PCA(n_components=n_components)
		encodings=self.__get_training_encodings__()
		pca.fit(encodings)
		return pca.components_


	def __convert_num_to_str__(self,recon_seqs,allow_pad): 
		seqs=[]
		good_idx=[]
		for idx,num_repr in enumerate(recon_seqs): 
			# if idx==0:
			# 	print (num_repr)

			if num_repr[0]!=self.Params.token['start']:
				# if idx==0:
				# 	print ('im here 1')
				continue 
			if self.Params.token['start'] in num_repr[1:]: 
				# if idx==0:
				# 	print ('im here 2')
				continue 
			if self.Params.token['eos'] not in num_repr[1:]:
				# if idx==0:
				# 	print ('im here 3')
				continue 
			if self.Params.token['unk'] in num_repr: 
				# if idx==0:
				# 	print ('im here 4')
				continue 
			# print ('num_repr',num_repr)
			eos_idx=(num_repr == self.Params.token['eos']).nonzero()[0][0]
			# print (eos_idx)
			if self.Params.token['pad'] in num_repr[:eos_idx+1]:
				# if idx==0:
				# 	print ('im here 5')
				continue 
			### after eos must all be padding (currently omit)###
			# if eos_idx!=len(num_repr)-1: ## if eos is not the final token  
			# 	right_repr=num_repr[eos_idx+1:]
			# 	# all_pad=np.sum(left_repr==self.Params.token['pad'])==len(left_repr)
			# 	all_pad=np.sum(right_repr==self.Params.token['pad'])==len(right_repr)
			# 	# print (all_pad)
			# 	if all_pad==False:
			# 		continue
					   
			# print (num_repr)
			num_repr=num_repr[1:eos_idx]
			# if idx==0:
			# 	print (num_repr)
			str_repr=''.join([self.Params.token_reverse[dim] for dim in num_repr])
			seqs.append(str_repr)
			good_idx.append(idx)
			# print (seqs)

			# if idx==0:
			# 	print (str_repr)
			# 	print (seqs)
			# sys.exit(0)
		# print (seqs[0])
		# sys.exit(0)
		return seqs,good_idx


	def decode(self,encodings,obtain_embeding=False,allow_pad=False): 
		MP=self.Params.RNN_VAE_params
		with torch.no_grad():
			encodings_torch=torch.tensor(encodings).to(self.Params.device)
			encodings_torch=encodings_torch.to(torch.float32)
			recon_seqs,embeddings=self.model.sampling_embedding(encodings_torch)
			# print (embeddings.size())
			# sys.exit(0)
			recon_seqs=recon_seqs.cpu().numpy()
			seqs,good_idx=self.__convert_num_to_str__(recon_seqs,allow_pad)
			good_encodings=[encodings[idx] for idx in good_idx]
			if obtain_embeding==True:
				embeddings=embeddings.cpu().numpy()
				good_embedings=[embeddings[idx] for idx in good_idx]
				return seqs,good_encodings,good_embedings
		return seqs,good_encodings

	def get_props(self,embeddings,classification=True):
		with torch.no_grad():
			embeddings_torch=torch.tensor(embeddings).to(self.Params.device)
			embeddings_torch=embeddings_torch.to(torch.float32)
			pred_props=self.model.props_predictor.forward_embedding_conv(embeddings_torch)
			if classification==True:
				Sigmoid=nn.Sigmoid()
				pred_props=(Sigmoid(pred_props)>=0.5)
			return pred_props 



class neural_optimization():
	def __init__(self,model_info_folder,prop_model_param_path,DoF):
		'''
			params: 
				prop_model_path: path for the model file 
				prop_model_param_path: path for the param dict file
				DoF: degree of freedom 	
		'''
		# sys.path.insert(0,prop_model_path)
		sys.path.insert(0,model_info_folder)
		from model_file import RNN_VAE 
		import configure_file as Params
		self.model = RNN_VAE(params.RNN_VAE_params).load_state_dict(torch.load(prop_model_param_path))
		self.device = Params.device
		self.DoF=DoF
		self.control_points=torch.rand(N_samples,self.DoF).to(self.device)

		### optimization layer 
		self.control_layer = nn.Linear(DoF,Params.RNN_VAE_params['z_dim'])

	def forward(self,N_samples):
		# DoF_values=torch.rand(N_samples,self.DoF).to(self.device)
		# DoF_values=(DoF_values-0.5)*2
		# init_encoding=
		encoding=self.control_layer(self.control_points)
		with torch,no_grad():
			seqs,embedding=self.model.sampling_embedding(encodings)
			props=self.model.props_predictor.forward_embedding_conv(embeddings)
		return props,encodings,seqs







if __name__=='__main__':
	
	#### get encoding #### 
	# model_file_folder = 'results/trial_8'
	# model_params = 'results/trial_8/models/model_44700.ckpt'
	# MU=model_usage(model_file_folder,model_params)
	# mean,std = MU.get_init_mean_std()
	# std=np.mean(std)*np.ones_like(std)
	# encodings = MU.sample_encoding(mean,std,64)
	# seqs,good_encodings=MU.decode(encodings)
	# PCs=MU.get_encoding_PCs(2)
	#### ends ####

	#### neural optimization ####
	# NO_params={
	# 'model_info_folder':,
	# 'prop_model_param_path':,
	# 'DoF':
	# }


	# print (PCs)
	# print (np.shape(PCs))
	# print (seqs)



	df=pd.read_csv('data/CCP_NCCP_data.csv')
	train_eval_ratio=0.8
	N_seq=len(df)
	N_train=int(N_seq*train_eval_ratio)
	# print (df)
	df_train=df[:N_train]
	# print (df_train)
	df_train_pos=df_train[df_train['is_CCP']==1]
	# print (df_train_pos)
	# sys.exit(0)
	config_folder='results/trial_3_cont'
	mode_file='results/trial_3_cont/models/model_66.ckpt'
	MU=model_usage(config_folder,mode_file)
	pos_encodings=MU.get_encoding(df_train_pos)
	seqs,good_encodings,good_embedings=MU.decode(pos_encodings,obtain_embeding=True,allow_pad=True)
	
	# print (df_train_pos['aa'][0])
	# print ('----------------------------------')
	# print (seqs)
	# sys.exit(0)
	# from sklearn.decomposition import PCA
	# from matplotlib import pyplot as plt 
	# pca = PCA(n_components=2).fit(pos_encodings)
	# reduced_encoding=np.matmul(pos_encodings,pca.components_.T)
	# plt.plot(reduced_encoding[:,0],reduced_encoding[:,1],'o')
	# plt.show()
	# print (np.shape(reduced_encoding))

	from sklearn.mixture import GaussianMixture
	gmm = GaussianMixture(n_components=5, random_state=0,covariance_type='full').fit(pos_encodings)
	# print (gmm.weights_)
	# soft=gmm.predict_proba(pos_encodings)
	N_desired=10

	result=[]
	ccp_seq=df[df['is_CCP']==1]
	while len(result)<N_desired:
		N_sample=100
		encodings=gmm.sample(N_sample)[0]
		seqs,good_encodings,good_embedings=MU.decode(encodings,obtain_embeding=True,allow_pad=True)
		# print (len(seqs))
		# print (len(good_embedings))
		# print (np.shape(good_embedings))
		# print (len(good_embedings))
		print (len(result))
		if len(good_embedings)!=0:
			props=MU.get_props(good_embedings)
			seqs=[seqs[idx] for idx,prop in enumerate(props) if prop==1]
			seqs=[seq for seq in seqs if seq not in result and seq not in ccp_seq]
			result.extend(seqs)
	print (result[:N_desired])

