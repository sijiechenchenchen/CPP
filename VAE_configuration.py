import torch
import numpy as np

device=torch.device('cpu')
sample_frac=0.2
log_results=True 
Result_root='results/trial_3_cont/'

### resume training setup ###
resume_train=True
resume_train_path='results/trial_3/models/model_990.ckpt'




### Dataset settings ###

token={'unk':0,'pad':1,'start':2,'eos':3,'A':4,'R':5,'N':6,'D':7,'C':8,'E':9,'Q':10,'G':11,'H':12,'I':13,'L':14,'K':15,
'M':16,'F':17,'P':18,'S':19,'T':20,'W':21,'Y':22,'V':23}

token_reverse={0:'unk',1:'pad',2:'start',3:'eos',4:'A',5:'R',6:'N',7:'D',8:'C',9:'E',10:'Q',11:'G',12:'H',13:'I',14:'L',15:'K',
16:'M',17:'F',18:'P',19:'S',20:'T',21:'W',22:'Y',23:'V'}


target_props=['is_CCP']



outlier_param={
'bind_E_r':{'remove_what':'high','low_q':0.1,'high_q':0.9},
'interface_dSASA_r':{'remove_what':'low','low_q':0.1,'high_q':0.9},
'total_E_r':{'remove_what':'high','low_q':0.1,'high_q':0.9},
'hbond_E_fraction_r':{'remove_what':'both','low_q':0.1,'high_q':0.9},
'interface_separated_energy_r':{'remove_what':'high','low_q':0.1,'high_q':0.9},
'separated_total_energy_r':{'remove_what':'high','low_q':0.1,'high_q':0.9},
'interface_hbonds_r':{'remove_what':'none','low_q':0.1,'high_q':0.9},
'z_hydrophobicity':{'remove_what':'none','low_q':0.1,'high_q':0.9}
}


trans_param={
'bind_E_r':lambda x:np.log(x+100),
'interface_dSASA_r':lambda x: np.log(x+1),
'total_E_r':lambda x: np.log(x+1000),
'hbond_E_fraction_r':lambda x: x*100+50,
'interface_separated_energy_r':lambda x: np.log(x+200),
'separated_total_energy_r':lambda x: np.log(x+800),
'interface_hbonds_r':None,
'z_hydrophobicity':{}
}


preprocess_param={
'use_log_only':	True,
'outlier':outlier_param,
'transformation':trans_param
}



dataset_params={
'data_path':'data/CCP_NCCP_data.csv',
'max_seq_len':63,
'train_eval_ratio':0.8,
'token':token,
'target_props':target_props
}


### training setups ###
training_params={
'model_save_folder':Result_root+'models',
'lr':10**-4,
'epochs':100000,
'train_batch_size':64,
'test_batch_size':64,
'shuffle':True}


loss_weights={
'kl':1,
'recon':1,
'prop':1	
}


### Model setups ###
GRUEncoder_params={'h_dim':80,'biGRU':True,'layers':1,'p_dropout':0.0}
GRUDecoder_params={'p_word_dropout':0.3,'p_out_dropout':0.3,'skip_connetions':True}
PropsRegressor_params={'h_dim':[128,64,32],'output_dim':1}#len(target_props)}


RNN_VAE_params={'n_vocab':len(token),'emb_dim':100,'z_dim':100,'max_seq_len':dataset_params['max_seq_len'],'PAD_IDX':token['pad'],
'device':device,'GRUEncoder_params':GRUEncoder_params,'GRUDecoder_params':GRUDecoder_params,
'PropsRegressor_params':PropsRegressor_params}

PropsRegressor_B_params={'network':'conv','z_dim':RNN_VAE_params['z_dim'],'h_dim':[128,64,32],'output_dim':PropsRegressor_params['output_dim']}

