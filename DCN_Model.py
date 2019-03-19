import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class DCN_Model(nn.Module):

    def __init__(self,
                 feature_num,
                 cross_iter_epoch,
                 deep_iter_epoch,
                 emb_feature_num_list,

                 all_feature_num,
                 deep_layer_num_list,
                 device
                 ):

        super(DCN_Model,self).__init__()

        self.feature_num = feature_num
        self.cross_iter_epoch = cross_iter_epoch
        self.deep_iter_epoch = deep_iter_epoch
        self.emb_feature_num_list = emb_feature_num_list
        self.all_feature_num = all_feature_num
        self.deep_layer_num_list = deep_layer_num_list
        self.device = device
        self.batch_size = 128

        self.input_emb = nn.ModuleList(
            [nn.Embedding(num,self.emb_feature_num_list[index]) for index,num in enumerate(feature_num)]
        )

        self.cross_weight_list = []
        self.cross_bias_list = []
        self.cross_batchnorm_list = []
        for index in range(self.cross_iter_epoch):
            self.cross_weight_list.append(nn.Parameter(nn.init.normal_(torch.empty(self.batch_size,self.all_feature_num))))
            self.cross_bias_list.append(nn.Parameter(nn.init.normal_(torch.empty(self.batch_size,self.all_feature_num))))
            self.cross_batchnorm_list.append(nn.BatchNorm1d(self.all_feature_num,affine=False))

        self.cross_weight_list = nn.ParameterList(self.cross_weight_list)
        self.cross_bias_list = nn.ParameterList(self.cross_bias_list)
        self.batchnorm_list = nn.ModuleList(self.cross_batchnorm_list)

        self.deep_linear_list = nn.ModuleList(
            [nn.Linear(self.deep_layer_num_list[index], self.deep_layer_num_list[index + 1]) for index in range(self.deep_iter_epoch)]
        )
        self.deep_batchnorm_list = nn.ModuleList(
            [nn.BatchNorm1d(self.deep_layer_num_list[index+1],affine=False) for index in range(self.deep_iter_epoch)]
        )

        self.linear_out = nn.Linear(self.all_feature_num + self.deep_layer_num_list[-1] , 1)

        self.output = nn.Sigmoid()



    def forward(self, X):

        first = last = 0

        temp = []
        #temp.append(X[:,:self.continuous_feature_num])
        for index , input_emb in enumerate(self.input_emb):
            last += self.feature_num[index]
            temp.append(torch.mean(input_emb(X[:,first:last].long()),dim=1))
            first = last


        x0 = torch.cat(temp,1)
        #print(x0)

        '''
        cross part
        '''
        cross_iter = x0
        #print(x0)
        for index in range(self.cross_iter_epoch):
            #cross_iter = torch.chain_matmul(x0.t(),cross_iter,self.cross_weight_list[index][:X.size(0)].t()) + cross_iter.t() + self.cross_bias_list[index][:X.size(0)].t()
            cross_iter = torch.mm(torch.mm(x0.t(), cross_iter),self.cross_weight_list[index][:X.size(0)].t()) + cross_iter.t() + self.cross_bias_list[index][:X.size(0)].t()
            cross_iter = cross_iter.t()
            cross_iter = self.batchnorm_list[index](cross_iter)

        '''
        deep part
        '''
        deep_iter = x0
        for index, deep_linear in enumerate(self.deep_linear_list):
            deep_iter = deep_linear(deep_iter)
            deep_iter = self.deep_batchnorm_list[index](deep_iter)
            deep_iter = F.relu(deep_iter)

        #print(deep_iter)
        '''
        output
        '''
        cross_deep_cat = torch.cat((cross_iter,deep_iter), 1)
        y_pred = self.linear_out(cross_deep_cat)
        #print(y_pred)
        y_pred = self.output(y_pred)

        return y_pred

    def fit(self,trainset,testset,n_epochs,lr):

        train_loader = DataLoader(dataset=trainset,batch_size= self.batch_size,shuffle=True)

        net = self.train()
        #net = torch.nn.DataParallel(net, [0, 1, 2, 3])
        optim = torch.optim.SGD(self.parameters(),lr, momentum=0.9)
        loss_func = F.binary_cross_entropy

        for epoch in range(n_epochs):

            loss_sum = 0

            for index , (X_train,Y_train) in enumerate(train_loader):

                x_train = X_train.to(self.device).float()
                y_train = Y_train.to(self.device).float()

                optim.zero_grad()
                y_pred = net(x_train).view(-1)
                loss = loss_func(y_pred,y_train)
                loss.backward()
                optim.step()

                loss_sum += loss

                if index == len(train_loader) - 1:
                    t = ((y_pred > 0.5).float() == y_train).sum().float() / (y_train.size(0))
                    print('epoch : %d ,Loss :%.8f , accuracy : %.2f%%' % (epoch,loss_sum,t.cpu().numpy() * 100))

        self.test(testset, net)

    def test(self,testloader,net):
        net.eval()
        test_loader = DataLoader(dataset=testloader, shuffle=True, batch_size=100)

        num_correct, num_sample = 0, 0

        with torch.no_grad():
            for index, (x_test, y_test) in enumerate(test_loader):
                x = x_test.to(self.device).float()
                y = y_test.to(self.device).float()

                y_pred = net(x).squeeze()

                num_correct += ((y_pred > 0.5).float() == y).sum().float()
                num_sample += y.size(-1)

            accuracy = num_correct / num_sample
            print('=' * 50)
            print('test accuracy : %.2f%%' % (accuracy.cpu().numpy() * 100))