import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.data import Dataset,DataLoader, random_split
from pathlib import Path
import torch
import torch.nn as nn
# from Function.detach_dataset import *

#set Dataset
class Dataset(Dataset):
    def __init__(self,root_pos,root_neg):
        self.sample = []
        pos_dir = Path(root_pos)
        neg_dir = Path(root_neg)
        for i in sorted(pos_dir.rglob('*.npy')):
            if (str(i.parts[-1][9:16]) != 'trail1_'):
                array = np.load(str(i))
                tensor = torch.from_numpy(array.T)
                tensor = torch.tensor(tensor,dtype=torch.float)
                condititon = [tensor,torch.tensor(1)]
                self.sample.append(condititon)
                condititon = []
        for i in sorted(neg_dir.rglob('*.npy')):
            if (str(i.parts[-1][9:16]) != 'trail1_'):
                array = np.load(str(i))
                tensor = torch.from_numpy(array.T)
                tensor = torch.tensor(tensor,dtype=torch.float)
                condititon = [tensor,torch.tensor(0)]
                self.sample.append(condititon)
                condititon = []
                
            
    def __getitem__(self,idx):
        data = self.sample[idx][0]
        label = self.sample[idx][1]
        return data,label
    
    def __len__(self):
        return len(self.sample)

batch_size = 10
p = 0
q = 0
dataset = Dataset('data_numpy/pos','data_numpy/neg')
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True,drop_last=True)


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.01),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 实例化网络
ann = MyCNN()

n_epochs = 200
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(ann.parameters(), lr=1e-4)
acc_train = []
acc_test = []
plt.figure(0,figsize=(10,5))
for n in range(n_epochs):
    for data, label in tqdm(iter(train_dataloader),colour='yellow'):
        optim.zero_grad()
        data = torch.reshape(data,(batch_size,1,17,85))
        output = ann(data)
        output2 = torch.reshape(output,(batch_size,2))
        loss = loss_fn(output2,label)
        loss.backward()
        optim.step()
        
    with torch.no_grad():
        print(f"the result of number of {n}:")
        train_dataloadertest = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True,drop_last=True)
        for data,label in iter(train_dataloadertest):
            data = torch.reshape(data,(len(train_dataset),1,17,85))
            output = ann(data)
            # output = output.mean(1)
            output = torch.reshape(output,(len(train_dataset),2))
            output = output.argmax(1)
            n0 = 0
            n1 = 0
            for i in output:
                if i == 0:
                    n0 += 1 
                if i == 1 :
                    n1 += 1
            print(f'0的数量:{n0}，1的数量:{n1}') 
            print(f'训练准确率:{(output==label).sum()/len(output)}')
            acc_train.append((output==label).sum()/len(output))
    with torch.no_grad():
        z = 0
        o = 0
        for data,label in iter(test_dataloader):
            if label == 0:
                z += 1
            if label == 1:
                o += 1
        print(f'测试标签中共有{z}个0，{o}个1')
        n_0 = 0
        n_1 = 0
        result_list = []
        num_result = 0
        for data,label in tqdm(test_dataloader,colour='yellow'):
            data = torch.reshape(data,(1,1,17,85))
            output = ann(data)
            # output = output.mean(1)
            output = output.reshape(1,2)
            output = output.argmax(1)
            result = (output.item()==label.item())
            if output.item() == 0:
                n_0  += 1
            if output.item() == 1:
                n_1 += 1
            result_list.append(result)
            if result == False:
                num_result += 1
        print(f'预测错误:{num_result}个')
        acc = sum(result_list)/len(result_list)
        print(f'预测准确率:{acc}')
        print(f'预测的0的数量:{n_0}，预测的1的数量:{n_1}') 
        acc_test.append(acc)
torch.save(ann,'/home/ruixing/workspace/chbtar/chb/models/ann_model.pth')
plt.plot(range(n_epochs),acc_train)
plt.plot(range(n_epochs),acc_test)
plt.show()