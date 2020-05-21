import os
from Embedding import Embedding
import torch
import torch.nn as nn
import datetime
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class model(nn.Module):
    def __init__(self, input_n, hidden_n, output_n, feature_n):
        super(model, self).__init__()
        self.input_n = input_n
        self.hidden_n = hidden_n
        self.output_n = output_n
        self.feature_n = feature_n

        # modeling
        self.linear_1 = nn.Linear(self.input_n * self.feature_n, self.hidden_n)
        self.linear_2 = nn.Linear(self.hidden_n, self.hidden_n)
        self.linear_3 = nn.Linear(self.hidden_n, self.hidden_n)
        self.linear_4 = nn.Linear(self.hidden_n, self.hidden_n)
        self.linear_5 = nn.Linear(self.hidden_n, self.hidden_n)
        self.linear_6 = nn.Linear(self.hidden_n, self.output_n)

        self.dropout = nn.Dropout() # p -> probability to be zero

    def forward(self, x):
        x = x.view(-1, self.input_n * self.feature_n)
        x = torch.relu(self.linear_1(x))
        x = self.dropout(x)
        x = torch.relu(self.linear_2(x))
        x = self.dropout(x)
        x = torch.relu(self.linear_3(x))
        x = self.dropout(x)
        x = torch.relu(self.linear_4(x))
        x = self.dropout(x)
        x = torch.relu(self.linear_5(x))
        x = self.dropout(x)
        x = self.linear_6(x)

        return x

def load_embedding(enterprise):
    root = os.path.join('data', 'STOCK')
    model = Embedding('Drop_Features.csv')
    
    # model.fit()
    pretrained_embedding = model.load_file() # time, stock, field
    root_stock = os.path.join(root, enterprise + '.csv')
    
    stocks = pd.read_csv(root_stock)
    stocks['Date'] = [int(datetime.datetime.strptime(d, '%Y. %m. %d').strftime('%Y%m%d')) for d in stocks.Date]
    means = stocks.iloc[0]
    # stocks['Open'] = [d/means['Open'] for d in stocks['Open']]
    # stocks['Close'] = [d/means['Close'] for d in stocks['Close']]
    # stocks['Volume'] = [d / means['Volume'] for d in stocks['Volume']]
    stads = []
    for i in range(2011, 2020):
        stads.extend([int('{0}0331'.format(i)),int('{0}0630'.format(i)),int('{0}0930'.format(i)),int('{0}1231'.format(i))])
    idx = 0
    cnts = [0]*len(stads)
    for d in stocks['Date']:
        if d > stads[idx]:
            idx += 1
        cnts[idx]+=1
    fs = None
    enterprise_list = {'AMZN':0, 'GOOGL':1, 'FB':2, 'TENCENT':3, 'BRK':4}
    for cnt, e in zip(cnts, pretrained_embedding):
        e = np.expand_dims(e[enterprise_list[enterprise]], 1) # google : 1
        if fs is None:
            fs = np.repeat(e, cnt, axis=1).reshape(10, -1)   # k = 10
        else:
            fs = np.concatenate((fs, np.repeat(e, cnt, axis=1).reshape(10, -1)),axis=1)
    final_data = np.concatenate((stocks[['Date','Open', 'Close']].values.T, fs),axis=0)
    return final_data.T

inputLength = 20
hidden_dim = 128
targetOffset = 20
windowOffset = 5
output_dim = targetOffset-1
l2 = 0.0001
learningRate = 0.001
lastTradeTime = 20200000
testStartTime = 20180000
# FirstDay = 20110000
# LastDay = 20200000


enterprise = 'GOOGL'

window = plt.figure ()
windowLoss = plt.figure ()
# window.subplots_adjust ( 0.03,0,0.99,1,0.1,0 )
graph = window.add_subplot ( 1,1,1 )
graphLoss = windowLoss.add_subplot ( 1,1,1 )
# graph.legend(['raw data', 'prediction'])
# graphLoss.legend(['train loss', 'test loss'])

def loadCompany (enterprise):
    rawData = load_embedding(enterprise)    # times, features (2263, 14)
    featureNum = np.shape(rawData)[1] - 2 # open and time
    # 0: Time, 1:Open, 2: Close, 3: Volume, 4~ 13: embedded features
    nData = rawData.__len__() - inputLength - targetOffset + 1 #
    nInput = int((nData-1) // windowOffset) + 1
    # nInput = len(range(0, rawData.__len__()-inputLength - targetOffset + 1, windowOffset))
    # nInput = rawData.__len__ ( ) - inputLength - targetOffset + 1
    inputs = np.empty ( [ nInput,inputLength, featureNum],np.float32 )
    labels = np.empty ( [ nInput,1 ],np.float32 )
    inputs [:,:, :] = [ rawData [ dataIndex:dataIndex + inputLength, 2:] for dataIndex in range ( 0, nData, windowOffset)]
    labels [ :,0 ] = [rawData [ dataIndex + inputLength + targetOffset - 1,1] for dataIndex in range(0, nData, windowOffset)]

    # divide with end-close
    if featureNum == 12:
        print('Close, Volume, Financial (10)')
    elif featureNum == 11:
        print('Close, Financial (10)')
    closeOfWindow = np.array([inputs[idx,-1,0] for idx in range(nInput)]) # close
    labels[:, 0] = np.array([labels[idx,0]/closeOfWindow[idx]  for idx in range(nInput)])
    inputs[:,:,0] = np.array([inputs[idx,:,0]/closeOfWindow[idx]  for idx in range(nInput)])    # close

    inputs = torch.from_numpy ( inputs )
    labels = torch.from_numpy ( labels )
    rawData = rawData [ inputLength - 1: ]
    nTrainData = 0
    while True:
        if rawData [ nTrainData*windowOffset,0 ] > testStartTime:break
        nTrainData+=1
    print(nTrainData)
    graph.plot ( labels.detach().cpu().numpy(),color="black",linewidth=0.5 ,label='raw data')
    return rawData,inputs,labels,nTrainData, featureNum
class Investor ( ):
    def __init__ ( self,gpuIndex ):
        self.device = torch.device ( "cuda:" + str ( gpuIndex ) )
        self.rawData,self.inputs,self.labels,self.nTrainData, self.featureNum = loadCompany (enterprise)
        self.nData = self.rawData.__len__ ( )
        self.nTestData = (self.nData - inputLength - targetOffset) // windowOffset + 1
        self.inputs = self.inputs.to ( self.device )
        self.labels = self.labels.to ( self.device )
        self.acc = torch.where((self.labels >= self.inputs[:,-1,0:1]),torch.ones_like(self.labels),torch.zeros_like(self.labels))
        # self.acc = self.acc.detach().cpu().numpy()
        self.model = model(inputLength, hidden_dim, output_dim, self.featureNum).to(self.device)
        self.optimizer = torch.optim.Adam ( self.model.parameters ( ),learningRate,weight_decay=l2 )
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=100)
    def backward ( self,outputs,labels ):
        loss = torch.mean ( torch.abs ( outputs - labels ) )
        self.optimizer.zero_grad ( )
        loss.backward ( )
        self.optimizer.step ( )
        # self.scheduler.step(loss)
        return loss.cpu ( ).detach ( ).numpy( )
    def run ( self ):
        nRun = 0
        fast_loss = [0.0, 0.0]
        mean_loss = [0.0, 0.0]
        while True:
            self.model.train()
            outputs = self.model(self.inputs)
            outputs_cpu = outputs[:,-1].cpu ( ).detach ( ).numpy( )
            loss = self.backward ( outputs [ :self.nTrainData],self.labels [ :self.nTrainData, ] )
            acc = torch.where(outputs[:,-1:] >= self.inputs[:,-1,:1], torch.ones_like(self.labels), torch.zeros_like(self.labels))
            train_acc = torch.where(acc == self.acc, torch.ones_like(acc), torch.zeros_like(acc))[self.nTrainData:].mean()

            if ( nRun & 31 ) == 0:
                test_loss = torch.mean(torch.abs(outputs - self.labels)[self.nTrainData:]).detach().cpu().numpy()

                self.model.eval()
                outputs = self.model(self.inputs)
                outputs_cpu = outputs[:, -1].cpu().detach().numpy()
                loss = self.backward(outputs[:self.nTrainData], self.labels[:self.nTrainData, ])
                acc = torch.where(outputs[:, -1:] >= self.inputs[:, -1, :1], torch.ones_like(self.labels),torch.zeros_like(self.labels))
                acc = torch.where(acc == self.acc, torch.ones_like(acc), torch.zeros_like(acc))[self.nTrainData:]
                print ( "%5d \t %f \t %f \t %f \t %f" % (nRun+1, loss, test_loss, train_acc.mean(), acc.mean()))
                graph.plot ( outputs_cpu,color="red",linewidth=0.5 , label='prediction')
                if nRun // 31 !=0:
                    graphLoss.scatter((nRun)//31+1, loss, color='blue', s=0.5, label='train loss')
                    graphLoss.scatter((nRun)//31 + 1, test_loss, color='red', s=0.5, label='test loss')
                plt.pause ( 0.00000000000001 )
                graph.lines.pop ( -1 )
            nRun+=1
Investor (0 ).run ( )





