import numpy as np 
import VAE_configuration as Params 
from scipy.stats import pearsonr as corr
import matplotlib.pyplot as plt 
from eval_model import eval 

# pred_test=np.load('all_pred_props_test.npy')
# gt_test=np.load('all_gt_props_test.npy')

# print (gt_test)
# print (np.shape(pred_test))
# plt.plot(gt_test,'o')
# plt.show()

# sys.exit(0)

# SHAPE=np.shape(pred)

# figs,axs=plt.subplots(2,4)


model_path='results/trial_8/models/model_74200.ckpt'
test_losses,pred_test,gt_test=eval(model_path,'test')
pred_test=np.array(pred_test)
# print (pred_test)
gt_test=np.array(gt_test)


figs,axs=plt.subplots()


for idx, prop in enumerate(Params.target_props[0:1]):

	# prop_pred_train=pred_train[:,idx]
	# gt_pred_train=gt_train[:,idx]

	# R_train=corr(prop_pred_train,gt_pred_train)[0]
	# ax=axs[idx//4][idx%4]
	ax=axs
	# ax.plot(prop_pred_train,gt_pred_train,'o')
	# ax.set_xlabel('prediction')
	# ax.set_ylabel('ground truth')
	# ax.set_title(prop)

	prop_pred_test=pred_test[:,idx]
	gt_pred_test=gt_test[:,idx]
	MAPE=np.mean(np.divide(prop_pred_test-gt_pred_test,1.5))
	print (MAPE)
	# print (prop_pred_test-gt_pred_test)
	# print (MAPE)
	R_test=round(corr(prop_pred_test,gt_pred_test)[0],3)
	ax.plot(prop_pred_test,gt_pred_test,'o')
	ax.set_xlabel('prediction')
	ax.set_ylabel('ground truth')


	# min_train=np.min(gt_pred_train)
	# max_train=np.max(gt_pred_train)
	# ax.plot(np.arange(min_train,max_train),np.arange(min_train,max_train))

	ax.legend(['train R:{}'.format(R_test)])
	# ax.legend(['train','test','ideal line'])
plt.show()
	# print ('R',R)






