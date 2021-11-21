import torch
import pandas as pd 
import torch.nn.functional as F
import numpy as np

from model.model import RNN_VAE
from data_proc.data_process import Peptide_dataset 
from torch.utils.data import DataLoader
import VAE_configuration as Params
from shutil import copyfile
# from torchsummary import summary
# from torchviz import make_dot
from matplotlib import pyplot as plt 




def recon_dec(gt_seq, logits):
    PAD_IDX=Params.dataset_params['token']['pad']
    recon_loss = F.cross_entropy(  # this is log_softmax + nll
        logits.view(-1, logits.size(2)), gt_seq.view(-1), reduction='mean',
        ignore_index=PAD_IDX  # padding doesnt contribute to recon loss & gradient
    )
    return recon_loss


def props_loss(gt_props,pred_props): 
    MSE_loss = torch.mean(torch.sqrt(torch.sum(torch.pow(gt_props-pred_props,2),axis=1)))#F.mse_loss(gt_props,pred_props)
    MAPE_loss =torch.mean(torch.div(torch.abs(gt_props-pred_props),torch.abs(pred_props)))#F.mse_loss(gt_props,pred_props)
    # print ('MAPE_loss',MAPE_loss)
    # sys.exit(0)
    # MAPE_loss= torch.mean(torch.sqrt(torch.sum(torch.pow(gt_props-pred_props,2),axis=1)))#F.mse_loss(gt_props,pred_props)

    return MSE_loss,MAPE_loss


def kl_gaussianprior(mu, logvar):
    """ analytically compute kl divergence with unit gaussian. """
    return torch.mean(0.5 * torch.sum((logvar.exp() + mu ** 2 - 1 - logvar), 1))



def running_avg(avg_val,cur_val,it):
    it=it+1  ## due to the model is 1 based 
    return avg_val*(it-1)/it+cur_val/it


def calculate_loss(model,dataset_iter):
    [seq_token, gt_props] = next(dataset_iter)   
    # gt_props=gt_props
    gt_props=gt_props[:,0:1]

    (z_mu, z_logvar), (z, pred_props), dec_logits = model(seq_token)
    recon_loss = recon_dec(seq_token, dec_logits)
    kl_loss = kl_gaussianprior(z_mu, z_logvar)
    # kl_loss = torch.zeros_like(recon_loss)
    prop_loss,prop_MAPE=props_loss(gt_props,pred_props)
    total_loss=recon_loss*100+kl_loss*10+prop_loss
    cur_losses=[total_loss,recon_loss,kl_loss,prop_loss,prop_MAPE]
    return cur_losses, pred_props, gt_props

def step_model(model,dataset):
    avg_losses=[0,0,0,0,0]
    with torch.no_grad():
        all_pred_props=[]
        all_gt_props=[]
        dataset_iter=iter(dataset)
        for it in range(len(dataset)): #tqdm(range(len(train_dataloader)),disable=None):
            cur_losses, pred_props, gt_props=calculate_loss(model,dataset_iter)
            pred_props=pred_props.detach().cpu().numpy()
            gt_props=gt_props.detach().cpu().numpy()
            all_pred_props.extend(pred_props)
            all_gt_props.extend(gt_props)
            avg_losses=[running_avg(avg_losses[idx],cur_losses[idx],it).item() for idx in range(len(avg_losses))]

            # print (np.shape(all_pred_props))
            # print (np.shape(all_gt_props))
            # sys.exit(0)
            # np.save('all_pred_props_test.npy',all_pred_props)
            # np.save('all_gt_props_test.npy',all_gt_props)
    return avg_losses,all_pred_props,all_gt_props




def eval(model_path,dataset_type='test'):
    device=Params.device
    TP=Params.training_params
    MP=Params.RNN_VAE_params
    DP=Params.dataset_params
    print('Training base vae ...')
    model=RNN_VAE(**MP)
    model.load_state_dict(torch.load(model_path))

    np.random.seed(2021)

    df=pd.read_csv(DP['data_path'])    
    df=df.sample(frac=1).reset_index(drop=True)


    max_seq_len=DP['max_seq_len']
    train_eval_ratio=DP['train_eval_ratio']
    
    # df=Peptide_dataset(df,preprocess=True,plot=False,normalization_params=None,previous_N_data=0).data

    N_seq=len(df)
    N_train=int(N_seq*train_eval_ratio)
    
    df_train=df[:N_train].sample(frac=1).reset_index(drop=True)
    train_dataset=Peptide_dataset(df_train,is_train=True,preprocess=True,plot=False)

    df_test=df[N_train:].sample(frac=1).reset_index(drop=True)
    test_dataset=Peptide_dataset(df_test,is_train=False,preprocess=True,plot=False)

    if dataset_type=='test':
        dataloader = DataLoader(test_dataset, batch_size=TP['test_batch_size'], shuffle=False)
    else: 
        dataloader = DataLoader(train_dataset, batch_size=TP['test_batch_size'], shuffle=False)

    # plt.show()

    test_losses,all_pred_props,all_gt_props=step_model(model,dataloader)
    
    print ('Test results:',test_losses)
    torch.cuda.empty_cache()
    return test_losses,all_pred_props,all_gt_props


if __name__=='__main__':

    model_path='results/trial_8/models/model_74200.ckpt'
    test_losses,all_pred_props,all_gt_props=eval(model_path,'test')
    # all_gt_props=[all_gt_prop[0] for all_gt_prop in all_gt_props]
    # print (all_gt_props[:64])
    # print (all_gt_props[64:64*2])
    # sys.exit(0)
    # plt.plot(all_gt_props,'o')
    # plt.show()   
    np.save('all_pred_props_test_conv.npy',all_pred_props)
    np.save('all_gt_props_test_conv.npy',all_gt_props)
