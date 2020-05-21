import pandas as pd
import os
from constants import STANDARD_COLUMN
import numpy as np
import math, datetime
import pandas_datareader.data as finance
import matplotlib.pyplot as plt


class data_collection():
    def __init__(self, symbols=None):
        self.ISIN_to_Symbol, self.Symbol_to_ISIN = self.look_up_dict()
        if symbols is None:
            self.symbols = [Symbol for Symbol in self.ISIN_to_Symbol.values() if Symbol != '2222:SE']
        else:
            self.symbols = symbols
        # check for fault
        marketcaps = finance.get_quote_yahoo(self.symbols)['marketCap']
        prices = finance.get_quote_yahoo(self.symbols)['price']
        self.numstocks = dict([(f, marketcaps[f] / prices[f]) for f in list(marketcaps.index)])

    def look_up_dict(self):
        df = pd.read_csv('ISIN_to_Symbol.csv')
        ISIN_to_Symbol = dict(zip(df['ISIN'].values, df['Symbol'].values))
        Symbol_to_ISIN = dict(zip(df['Symbol'].values, df['ISIN'].values))

        return ISIN_to_Symbol, Symbol_to_ISIN

    def str_to_date(self, str):
        year = int(str[:4])
        month = int(str[4:6])
        date = int(str[6:])
        return datetime.datetime(year, month, date)

    def stock_prices(self):
        for symbol in self.symbols:
            df = finance.DataReader(symbol,'yahoo','1990-01-01', '2020-05-15')
            df['Date'] = [date.strftime('%Y%m%d') for date in df.index]
            df =df.drop(columns=['Close'])
            df['MarketCap'] = [price*self.numstocks[symbol] for price in df['Adj Close'].values.reshape(-1)]
            df = df.rename(columns={'Adj Close':'Close'})
            df = df[['Date','Open','High','Low','Close','Volume','MarketCap']]
            print(self.Symbol_to_ISIN[symbol]+'.csv')
            df.to_csv(os.path.join('STOCK', self.Symbol_to_ISIN[symbol]+'.csv'),index=False)

print(finance.get_quote_yahoo('NTDMF')['marketCap'])
symbols = None #['MSFT', 'AAPL','HK:0700','SE:2222']
c = data_collection()
c.stock_prices()

class data_preprocessing():
    def __init__(self):
        self.root = os.path.join('ROUETER')
        self.raw_list = ['10_19_finan.csv','00_09_finan.csv','90_99_finan.csv']
        self.files = ['10_19_rm.csv','00_09_rm.csv','90_99_rm.csv']
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

    def only_quarter(self):
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
    def pick_field_by_standards(self):
        '''
            pick field by standards
        '''
        for file in self.files:
            r = os.path.join(self.root, file)
            df = pd.read_csv(r)
            drop_i = [idx for idx in range(len(df)) if df.loc[[idx],['Field']].values[0][0] not in STANDARD_COLUMN]
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

    def pick_company(self, name):
        '''
            pick certain company
        :param name: the name of company to be selected (str)
        '''
        for file in self.files:
            r = os.path.join(self.root, file)
            df = pd.read_csv(r)
            df = df.loc[:,['Field','Time',name]]
            df.to_csv(file[:-4]+'_WMT.csv', index=False)

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
    def get_files(self):

        print(self.files)
        print('completely finished!')
#
#
# processor = data_preprocessing()
# # processor.remove_repetation()
# processor.pick_company('US9311421039') # Bank of America : US0605051046 / Coca cola : US1912161007 / 'AT & T : US00206R1023 / walmart : US9311421039
# # processor.only_quarter()
# # d.pick_field()
# # processor.pick_field_by_standards()
# # # d.pick_k_companies(40)
# # processor.pick_nice_companies()
#
# # processor.get_files()

# class data_selection():
#     def __init__(self, symbol):
#         self.train_start = train_start
#         self.train_end = train_end
#         self.fiscal_num = fiscal_num
#
#         df = pd.read_csv('ISIN_to_Symbol.csv')
#         self.INIS_to_Symbol = dict(zip(df['INIS'].values, df['Symbol'].values))
#         self.Symbol_to_INIS = dict(zip(df['Symbol'].values, df['INIS'].values))
#
#         self.inis_lists = [stock[:-3] for stock in os.listdir('FINANCE') if stock[-3:] == '.csv']
#         self.selected_inis = self.Symbol_to_INIS[symbol]
#         self.others_inis = [ self.Symbol_to_INIS[inis] for inis in self.inis_lists if inis != self.selected_inis ]
#
#     def get_data(self, train_start, train_end, fiscal_num=4):
