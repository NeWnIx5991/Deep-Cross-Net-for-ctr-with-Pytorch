from DCN_utils import utils
from DCN_Model import DCN_Model
import torch
import torch.utils.data as Data

X_train,Y_train,X_test,Y_test,feature_num = utils()

device = 'cpu'
use_cuda = True

if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda'

cross_iter_epoch = deep_iter_epoch = 5
emb_feature_num_list = [6*int(pow(num,0.25)) for num in feature_num]
#continuous_feature_num = X_train.shape[1] - sum(feature_num)
all_feature_num = sum(emb_feature_num_list)
deep_layer_size = 50
deep_layer_num_list = [all_feature_num] + [deep_layer_size] * deep_iter_epoch

fm = DCN_Model(feature_num,cross_iter_epoch,deep_iter_epoch,emb_feature_num_list,
               all_feature_num,deep_layer_num_list,device).to(device)

train_torch_data = Data.TensorDataset(torch.from_numpy(X_train),torch.from_numpy(Y_train))
test_torch_data = Data.TensorDataset(torch.from_numpy(X_test),torch.from_numpy(Y_test))

fm.fit(train_torch_data,test_torch_data,120,0.01)
