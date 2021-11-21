import torch
import pandas as pd 
import torch.nn.functional as F
import numpy as np

from model.model import RNN_VAE
from model.model import PropsRegressor_B
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


def calculate_loss(model,prop_pred_model,dataset):

    seq_token, gt_props = next(dataset)
    # gt_props=gt_props
    gt_props=gt_props[:,0:1]
    # print (gt_props[:,0:1].size())
    embeddings = model.get_embedding(seq_token)
    pred_props = prop_pred_model.forward(embeddings)
 
    prop_loss,prop_MAPE=props_loss(gt_props,pred_props)
    total_loss=prop_loss*10
    cur_losses=[prop_loss,prop_MAPE]
    return cur_losses, pred_props, gt_props

def step_model(model,prop_pred_model,dataset,optimizer,require_grad=True):
    avg_losses=[0,0]
    dataset_iter=iter(dataset)
    if require_grad:
        for it in range(len(dataset)): #tqdm(range(len(train_dataloader)),disable=None):
            cur_losses, pred_props, gt_props=calculate_loss(model,prop_pred_model,dataset_iter)
            prop_pred_model.zero_grad()
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
                cur_losses, pred_props, gt_props=calculate_loss(model,prop_pred_model,dataset_iter)
                pred_props=pred_props.detach().cpu().numpy()
                gt_props=gt_props.detach().cpu().numpy()
                all_pred_props.extend(pred_props)
                all_gt_props.extend(gt_props)
                avg_losses=[running_avg(avg_losses[idx],cur_losses[idx],it).item() for idx in range(len(avg_losses))]

    return avg_losses




def train(training_params,model_params,dataset_params,device):
    TP=training_params
    MP=model_params
    DP=dataset_params


    print('Training base vae ...')
    model=RNN_VAE(**MP)
    model.load_state_dict(torch.load('results/trial_8/models/model_7.ckpt',map_location=Params.device))
    


    prop_pred_model=PropsRegressor_B(**Params.PropsRegressor_B_params).to(Params.device) 

    optimizer = torch.optim.Adam(prop_pred_model.parameters(), lr=TP['lr'])


    np.random.seed(2021)
    df=pd.read_csv(DP['data_path'])
    
    # df=df.sample(frac=1).reset_index(drop=True)
    df=df.sample(frac=Params.sample_frac).reset_index(drop=True)



    max_seq_len=DP['max_seq_len']
    train_eval_ratio=DP['train_eval_ratio']
    
    # df=Peptide_dataset(df,preprocess=True,plot=False,normalization_params=None,previous_N_data=0).data

    N_seq=len(df)
    N_train=int(N_seq*train_eval_ratio)
    
    df_train=df[:N_train].sample(frac=1).reset_index(drop=True)
    train_dataset=Peptide_dataset(df_train,is_train=True,preprocess=True,plot=False)


    df_test=df[N_train:].sample(frac=1).reset_index(drop=True)
    test_dataset=Peptide_dataset(df_test,is_train=False,preprocess=True,plot=False)


    cur_test_loss=1000

    for epoch in range(TP['epochs']):#tqdm(range(epochs),disable=None): 
        train_dataloader = DataLoader(train_dataset, batch_size=TP['train_batch_size'], shuffle=TP['shuffle'])
        train_losses=step_model(model,prop_pred_model,train_dataloader,optimizer,require_grad=True)
        print ('Train epoch:'+str(epoch),train_losses)
        test_dataloader = DataLoader(test_dataset, batch_size=TP['test_batch_size'], shuffle=False)
        test_losses=step_model(model,prop_pred_model,test_dataloader,optimizer,require_grad=False)
        print ('Test epoch:'+str(epoch),test_losses)

        if cur_test_loss>=test_losses[0]:
            cur_test_loss=test_losses[0]
            torch.save(prop_pred_model.state_dict(), TP['model_save_folder']+'/prop_model_{}.ckpt'.format(epoch))
        if epoch % 100 ==0:
            torch.save(prop_pred_model.state_dict(), TP['model_save_folder']+'/prop_model_{}.ckpt'.format(epoch))


        if Params.log_results:
            train_setup={'prop_loss':train_losses[0],'prop_MAPE':train_losses[1]}
            test_setup={'prop_loss':test_losses[0],'prop_MAPE':test_losses[1]}
            [writer.add_scalar('Train/'+key, value, epoch) for key,value in train_setup.items()]
            [writer.add_scalar('Test/'+key, value, epoch) for key,value in test_setup.items()]

    torch.cuda.empty_cache()



if __name__=='__main__':
    copyfile('VAE_configuration.py', Params.Result_root+'prop_configure.py')
    copyfile('model/model.py', Params.Result_root+'prop_model.py')
    device=Params.device
    training_params=Params.training_params
    RNN_VAE_params=Params.RNN_VAE_params
    dataset_params=Params.dataset_params

    train(training_params,RNN_VAE_params,dataset_params,device)