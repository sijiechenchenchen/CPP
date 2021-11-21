import torch
import pandas as pd 
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from model.model import RNN_VAE
from data_proc.data_process import Peptide_dataset 
from torch.utils.data import DataLoader
import VAE_configuration as Params
from shutil import copyfile
# from torchsummary import summary
# from torchviz import make_dot
from matplotlib import pyplot as plt 


if Params.log_results:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(Params.Result_root+'runs')



Sigmoid = nn.Sigmoid()
BCE_func = nn.BCELoss()





def recon_dec(gt_seq, logits):
    PAD_IDX=Params.dataset_params['token']['pad']
    recon_loss = F.cross_entropy(  # this is log_softmax + nll
        logits.view(-1, logits.size(2)), gt_seq.view(-1), reduction='mean',
        ignore_index=PAD_IDX  # padding doesnt contribute to recon loss & gradient
    )
    return recon_loss

def props_loss(gt_props,pred_props): 
    # print (pred_props)
    # print (gt_props)
    gt_props=gt_props.float()
    pred_logit=Sigmoid(pred_props)
    BCE_loss = BCE_func(pred_logit, gt_props)
    acc= torch.sum((pred_logit>=0.5)==gt_props)/len(gt_props)
    pred_cls=pred_logit>=0.5
    acc=torch.sum(pred_cls==gt_props)/len(gt_props)

    pos_idx=gt_props==1
    pos_gt=gt_props[pos_idx]
    pos_pred=pred_cls[pos_idx]
    TP=torch.sum(pos_pred==pos_gt)/len(pos_gt)

    neg_idx=gt_props==0
    neg_gt=gt_props[neg_idx]
    neg_pred=pred_cls[neg_idx]
    TN=torch.sum(neg_pred==neg_gt)/len(neg_gt)
    # print ('acc:',acc,'TP:',TP,'TN:',TN)
    # print ('\n\n\n')
    
    # print (output)
    # print (acc)
    # print (output.size())
    # sys.exit(0)
    # MSE_loss = torch.mean(torch.sqrt(torch.sum(torch.pow(gt_props-pred_props,2),axis=1)))#F.mse_loss(gt_props,pred_props)
    # MAPE_loss =torch.mean(torch.div(torch.abs(gt_props-pred_props),torch.abs(pred_props)))#F.mse_loss(gt_props,pred_props)
    

    # print ('MAPE_loss',MAPE_loss)
    # sys.exit(0)
    # MAPE_loss= torch.mean(torch.sqrt(torch.sum(torch.pow(gt_props-pred_props,2),axis=1)))#F.mse_loss(gt_props,pred_props)

    return BCE_loss,acc


def kl_gaussianprior(mu, logvar):
    """ analytically compute kl divergence with unit gaussian. """
    return torch.mean(0.5 * torch.sum((logvar.exp() + mu ** 2 - 1 - logvar), 1))



def running_avg(avg_val,cur_val,it):
    it=it+1  ## due to the model is 1 based 
    return avg_val*(it-1)/it+cur_val/it


def calculate_loss(model,dataset):

    seq_token, gt_props = next(dataset)
    # print (seq_token[0])
    # sys.exit(0)
    # gt_props=gt_props
    gt_props=gt_props[:,0:1]
    # print (gt_props[:,0:1].size())

    (z_mu, z_logvar), (z, pred_props), dec_logits = model(seq_token)
    recon_loss = recon_dec(seq_token, dec_logits)
    kl_loss = kl_gaussianprior(z_mu, z_logvar)
    # kl_loss = torch.zeros_like(recon_loss)
    prop_loss,prop_MAPE=props_loss(gt_props,pred_props)
    total_loss=recon_loss*Params.loss_weights['recon']+kl_loss*Params.loss_weights['kl']+prop_loss*Params.loss_weights['prop']
    # print (Params.loss_weights['prop'])
    # print (total_loss,recon_loss,kl_loss,prop_loss)
    # sys.exit(0)
    cur_losses=[total_loss,recon_loss,kl_loss,prop_loss,prop_MAPE]
    return cur_losses, pred_props, gt_props

def step_model(model,dataset,optimizer,require_grad=True):
    avg_losses=[0,0,0,0,0]
    dataset_iter=iter(dataset)
    if require_grad:
        for it in range(len(dataset)): #tqdm(range(len(train_dataloader)),disable=None):
            cur_losses, pred_props, gt_props=calculate_loss(model,dataset_iter)
            model.zero_grad()
            optimizer.zero_grad()
            cur_losses[0].backward()
            optimizer.step()
            with torch.no_grad():
                avg_losses=[running_avg(avg_losses[idx],cur_losses[idx],it).item() for idx in range(len(avg_losses))]
    else:
        with torch.no_grad():
            all_pred_props=[]
            all_gt_props=[]

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
    return avg_losses




def train(training_params,model_params,dataset_params,device):
    TP=training_params
    MP=model_params
    DP=dataset_params


    print('Training base vae ...')


    model=RNN_VAE(**MP)
    if Params.resume_train == False: 
        optimizer = torch.optim.Adam(model.parameters(), lr=TP['lr'])
    else: 
        print ('resume training')
        model.load_state_dict(torch.load(Params.resume_train_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=TP['lr'])


    np.random.seed(2021)
    df=pd.read_csv(DP['data_path'])
    # df=df.sample(frac=1).reset_index(drop=True)
    max_seq_len=DP['max_seq_len']
    train_eval_ratio=DP['train_eval_ratio']
    
    # df=Peptide_dataset(df,preprocess=True,plot=False,normalization_params=None,previous_N_data=0).data

    N_seq=len(df)
    N_train=int(N_seq*train_eval_ratio)
    
    df_train=df[:N_train].sample(frac=1).reset_index(drop=True)

    # print (len(df_train[df_train['is_CCP']==1])/len(df_train))
    # sys.exit(0)
 
    train_dataset=Peptide_dataset(df_train,is_train=True,preprocess=False,plot=False)
    df_test=df[N_train:].sample(frac=1).reset_index(drop=True)
    # print (len(df_test[df_test['is_CCP']==1])/len(df_test))
    # sys.exit(0)

    test_dataset=Peptide_dataset(df_test,is_train=False,preprocess=False,plot=False)
    cur_test_loss=1000

    for epoch in range(TP['epochs']):#tqdm(range(epochs),disable=None): 
        train_dataloader = DataLoader(train_dataset, batch_size=TP['train_batch_size'], shuffle=TP['shuffle'])
        train_losses=step_model(model,train_dataloader,optimizer,require_grad=True)
        print ('Train epoch:'+str(epoch),train_losses)
        # train_losses=step_model(model,train_dataloader,optimizer,require_grad=False)
        test_dataloader = DataLoader(test_dataset, batch_size=TP['test_batch_size'], shuffle=False)
        test_losses=step_model(model,test_dataloader,optimizer,require_grad=False)
        print ('Test epoch:'+str(epoch),test_losses)

        if cur_test_loss>=test_losses[0]:
            cur_test_loss=test_losses[0]
            torch.save(model.state_dict(), TP['model_save_folder']+'/model_{}.ckpt'.format(epoch))
        if epoch % 100 ==0:
            torch.save(model.state_dict(), TP['model_save_folder']+'/model_{}.ckpt'.format(epoch))

        if epoch % 1000 ==0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_losses
            }, TP['model_save_folder']+'/train_model_{}.ckpt')

        if Params.log_results:
            train_setup={'total_loss':train_losses[0],'recon_loss':train_losses[1],
            'kl_loss':train_losses[2],'prop_loss':train_losses[3],'prop_MAPE':train_losses[4]}
            test_setup={'total_loss':test_losses[0],'recon_loss':test_losses[1],
            'kl_loss':test_losses[2],'prop_loss':test_losses[3],'prop_MAPE':test_losses[4]}
            [writer.add_scalar('Train/'+key, value, epoch) for key,value in train_setup.items()]
            [writer.add_scalar('Test/'+key, value, epoch) for key,value in test_setup.items()]

    torch.cuda.empty_cache()



if __name__=='__main__':
    copyfile('VAE_configuration.py', Params.Result_root+'configure_file.py')
    copyfile('model/model.py', Params.Result_root+'model_file.py')
    device=Params.device
    training_params=Params.training_params
    RNN_VAE_params=Params.RNN_VAE_params
    dataset_params=Params.dataset_params

    train(training_params,RNN_VAE_params,dataset_params,device)