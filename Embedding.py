import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

standardized = True

class MatrixFactorization(nn.Module):
    def __init__(self, stock_n, field_n, k):
        super(MatrixFactorization, self).__init__()
        self.stock_n = stock_n
        self.k = k
        self.field_n = field_n

        # embedding stock intrinsic property
        self.stock_intr = nn.Embedding(self.stock_n, self.k)
        self.field_corr = nn.Embedding(self.field_n, self.k)


    def forward(self, stock, field):
        '''

        :param stock: indices for stock
        :param field: indices for field
        :return: expected financial statement
        '''
        return (self.stock_intr(stock) * self.field_corr(field)).sum(1)

    def get_weight(self):
        return self.stock_intr.weight, self.field_corr.weight

class Embedding():
    def __init__(self, file_csv, k=10, stock_n=5, epoch_n=15000, standardized=True):
        self.root_fin = os.path.join('data', 'FINAN')
        self.root = os.path.join(self.root_fin, file_csv)
        self.root_file = os.path.join(self.root_fin, 'embedding', 'Stock_intrinsic.npy')

        self.data_raw = pd.read_csv(self.root)
        self.data = None
        self.times = self.data_raw['Time'].unique()
        self.epoch_n = epoch_n
        self.stock_n = stock_n
        self.field_n = len(self.data_raw.columns) - 2  # delete 'Enterprise', 'Time'
        self.k = k
        self.standardized = standardized
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess(self):
        fs = None
        standards = None
        for t in self.times:
            tmp = self.data_raw[self.data_raw['Time'] == t].drop(columns=['Enterprise', 'Time']).values
            tmp = np.nan_to_num(tmp).reshape(1, self.stock_n, self.field_n)  # nan to zero
            if fs is None:
                fs = tmp
                standards = np.array([tmp[0, :, 0]])  # 0th column : Total Revenue
            else:
                fs = np.concatenate((fs, tmp), axis=0)
                standards = np.concatenate((standards, np.array([tmp[0, :, 0]])), axis=0)

        if self.standardized:
            # standardize using 'Total Revenue'
            fs /= np.array([np.repeat(standards[idx], self.field_n).reshape(self.stock_n, -1) for idx in range(len(standards))])
        self.data = fs
    def fit(self):
        self.preprocess()
        stocks = np.array([[idx] * self.field_n for idx in range(self.stock_n)]).reshape(-1)
        fields = np.array([idx for idx in range(self.field_n)] * self.stock_n).reshape(-1)
        stocks = torch.LongTensor(stocks).to(self.device)
        fields = torch.LongTensor(fields).to(self.device)

        fs = torch.FloatTensor(self.data).to(self.device)
        stock_intrinsics = torch.empty([len(self.times), self.stock_n, self.k])
        for idx, data in enumerate(fs):
            mf = MatrixFactorization(self.stock_n, self.field_n, self.k).to(self.device)
            optimizer = torch.optim.Adam(mf.parameters(), lr=0.001, weight_decay=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=10)
            for epoch in range(self.epoch_n):
                expected = mf(stocks, fields).view(5, -1)
                losses = ((data - expected) ** 2)
                print('{0:5d} : {1:.5f}'.format(epoch + 1, losses.mean()))

                loss = losses.sum()
                optimizer.zero_grad()  # gradient reset
                loss.backward()  # backprop
                optimizer.step()  # parameter reweighting
                scheduler.step(loss)

                if losses.mean() < 0.0001:  # terminate condition
                    break
            stock_intr, field_corr = mf.get_weight()
            stock_intrinsics[idx] = stock_intr
        if not self.check_exist():
            np.save(self.root_file[:-4], stock_intrinsics.detach().cpu().numpy())
            print('File is saved...')
    def check_exist(self):
        if os.path.isfile(self.root_file):
            print('File already exist...')
            return True
        return False

    def load_file(self):
        return np.load(self.root_file)

