import pandas as pd
import os
from constants import STANDARD_COLUMN
import numpy as np
import math

class data_preprocessing():
    def __init__(self):
        self.root = os.path.join('ROUETER')
        self.raw_list = ['10_19_finan.csv','00_09_finan.csv','90_99_finan.csv']
        self.files = ['10_19_.csv','00_09_.csv','90_99_.csv']
        self.nice_companies = ['from10NiceCompanies.csv','from00NiceCompanies.csv',None]
        # self.column_standards = pd.read_csv(os.path.join(self.root, 'column_ref.csv'))['Field'].unique()

    def remove_repetation(self):
        '''
            remove duplication of each field
        :return: repetation-removed data
        '''
        for idx_file, file in enumerate(self.raw_list):
            quarter_list = []
            missing_list = []
            field_list = []
            df = pd.read_csv(os.path.join(self.root, file))
            for idx in range(3, len(df)):  # FISICAL YEAR starts from 4 row
                t = df.loc[[idx], ['Time']].values[0][0]
                f = df.loc[[idx], ['Field']].values[0][0]
                if f not in field_list: # new field
                    quarter_list = []
                    field_list.append(f)
                if t not in quarter_list:
                    quarter_list.append(t)
                else:                   # repetation
                    # print(idx)        # ERROR CAN BE OCCURED !!!!!!!!!!!!!
                                        # ERROR (a float) -> check file
                    if sum([math.isnan(d) for i, d in enumerate(df.loc[[idx], :].values[0]) if i >= 2]):
                        missing_list.append(idx)
                    else:       # rows are not NaN?
                        print(idx)
                        break
            df = df.drop(missing_list)
            df.to_csv(os.path.join(self.root, self.files[idx_file]), index=False)

    def pick_quarter(self):
        '''
            remove non-quarter based companies (e.g., semi-annual)
        '''
        for file in self.files:
            r = os.path.join(self.root, file)
            df = pd.read_csv(r)
            drop_list = [c for c in df.columns if df.loc[[2],[c]].values[0][0] != 'Quarterly'][2:]
            df = df.drop(columns=drop_list)
            df.to_csv(r, index=False)

    def pick_field(self):
        '''
            pick specific fields
        '''
        for file in self.files:
            r = os.path.join(self.root, file)
            df = pd.read_csv(r)
            drop_i = []
            for idx in range(len(df)):
                if df.loc[[idx],['Field']].values[0][0] not in self.column_standards:
                    drop_i.append(idx)
            df = df.drop(drop_i)
            df.to_csv(r, index=False)

    def pick_k_companies(self, K):
        '''
            pick K companies
        :param K: the number of companies to be selected (int)
        '''
        for file in self.files:
            r = os.path.join(self.root, file)
            df = pd.read_csv(r)
            companies = df.drop(columns=['Time','Field']).columns
            assert K <= len(companies)
            drop_c = companies[K:]
            df = df.drop(columns=drop_c)
            df.to_csv(r, index=False)
    def pick_nice_companies(self):
        '''
            pick faithful companies
        '''

        for file, company in zip(self.files, self.nice_companies):
            r = os.path.join(self.root, file)
            if company is None:
                print('Nice companies does not exist')
                break
            df = pd.read_csv(r)
            nice = pd.read_csv(os.path.join(self.root, company)).columns
            drop_c = [c for c in df.columns if c not in nice]
            df = df.drop(columns=drop_c)
            df.to_csv(r, index=False)

# d = data_preprocessing()
# # d.remove_repetation()
# d.pick_quarter()
# d.pick_field()
# # d.pick_k_companies(40)
# d.pick_nice_companies()