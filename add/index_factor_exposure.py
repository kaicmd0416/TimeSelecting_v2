import pandas as pd
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import global_tools_func.global_tools as gt
import global_setting.global_dic as glv
from data.data_prepare import data_prepare
import numpy as np
from scipy.io import loadmat

class index_factor_exposure:
    def __init__(self,available_date):
        self.available_date=available_date
    def index_dic_processing2(self):
        dic_index = {'上证50': 'sz50', '沪深300': 'hs300', '中证500': 'zz500', '中证1000': 'zz1000',
                     '中证2000': 'zz2000', '中证A500': 'zzA500','国证2000':'gz2000'}
        return dic_index
    def jy_factor_exposure_update(self):  # available_date这里是YYYYMMDD格式
        if self.available_date <= '20200102':
            inputpath_factor = 'D:\OneDrive\Data_Original\data_jy\output_old\FactorRet'
        else:
            inputpath_factor = 'D:\OneDrive\Data_Original\data_jy\output_new\FactorRet'
        inputpath_factor = os.path.join(inputpath_factor, 'LNMODELACTIVE-' + str(self.available_date) + '.mat')
        try:
            annots = loadmat(inputpath_factor)['lnmodel_active_daily']['factorexposure'][0][0]
            barra_name, industry_name = gt.factor_name(inputpath_factor)
            status = 1
        except:
            status = 0
        if status == 1:
            df_factor_exposure = pd.DataFrame(annots, columns=barra_name + industry_name)
            df_factor_exposure.drop(columns=['country'], inplace=True)
        else:
            df_factor_exposure = pd.DataFrame()
        return df_factor_exposure
    def jy_factor_index_exposure_update(self, index_type):
        dic_index = self.index_dic_processing2()
        file_name = dic_index[index_type]
        inputpath_stockuniverse ='D:\OneDrive\Data_Original\data_other'
        if self.available_date <= '20200102':
            inputpath_factor = 'D:\OneDrive\Data_Original\data_jy\output_old\FactorRet'
            inputpath_stockuniverse = os.path.join(inputpath_stockuniverse, 'StockUniverse_old.csv')
        else:
            inputpath_factor = 'D:\OneDrive\Data_Original\data_jy\output_new\FactorRet'
            inputpath_stockuniverse = os.path.join(inputpath_stockuniverse, 'StockUniverse_new.csv')
        inputpath_factor = os.path.join(inputpath_factor, 'LNMODELACTIVE-' + str(self.available_date) + '.mat')
        inputpath_indexcomponent = 'D:\OneDrive\Data_prepared_test\data_index\index_component'
        inputpath_indexcomponent = os.path.join(inputpath_indexcomponent, file_name)
        df_stockuniverse = gt.readcsv(inputpath_stockuniverse)
        df_stockuniverse = df_stockuniverse[df_stockuniverse.columns.tolist()[:-2]]
        df_stockuniverse.rename(columns={'S_INFO_WINDCODE': 'code'}, inplace=True)
        stock_code = df_stockuniverse['code'].tolist()
        try:
            df_factor_exposure = self.jy_factor_exposure_update()
            status = 1
        except:
            status = 0
        if status == 1:
            
            df_factor_exposure['code'] = stock_code
            inputpath_indexcomponent = gt.file_withdraw(inputpath_indexcomponent, self.available_date)
            df_component = gt.readcsv(inputpath_indexcomponent)
            df_component.columns = ['code', 'weight', 'status']
            df_component = df_component[df_component['status'] == 1]
            index_code_list = df_component['code'].tolist()
            slice_df_stock_universe = df_stockuniverse[df_stockuniverse['code'].isin(index_code_list)]
            slice_df_stock_universe.reset_index(inplace=True)
            slice_df_stock_universe = slice_df_stock_universe.merge(df_component, on='code', how='left')
            index_code_list_index = slice_df_stock_universe['index'].tolist()
            df_factor_exposure=df_factor_exposure[['code','earningsyield']]
            df_factor_exposure.dropna(inplace=True)
            index_list=df_factor_exposure.index.tolist()
            df_final=df_factor_exposure
            df_final.reset_index(inplace=True)
            df_final = df_final[df_final['index'].isin(index_code_list_index)]
            slice_df_stock_universe = slice_df_stock_universe[slice_df_stock_universe['index'].isin(index_list)]
            slice_df_stock_universe['weight']=slice_df_stock_universe['weight']/slice_df_stock_universe['weight'].sum()
            weight = slice_df_stock_universe['weight'].astype(float).tolist()
            df_final.drop(columns='index', inplace=True)
            df_final.set_index('code', inplace=True)
            index_factor_exposure = list(
                np.array(np.dot(np.mat(df_final.values).T, np.mat(weight).T)).flatten())
            index_factor_exposure = [index_factor_exposure]
            df_final = pd.DataFrame(np.array(index_factor_exposure), columns=['earningsyield'])
            available_date2 = gt.strdate_transfer(self.available_date)
            df_final['valuation_date'] = available_date2
            df_final = df_final[['valuation_date'] + ['earningsyield']]
        else:
            df_final = pd.DataFrame()
        return df_final
    def get_earnings_yield_exposure(self):
        """
        Calculate earnings yield exposure between HS300 and ZZ2000
        """
        df = self.dp.raw_index_earningsyield()
        return df
    
    def get_index_return(self):
        """
        Get cumulative returns for HS300 and ZZ2000
        """
        df = self.dp.target_index()
        return df

    def merge_factor_exposure(self):
        """
        Merge earnings yield exposure with index returns
        """
        df_earnings = self.get_earnings_yield_exposure()
        df_return = self.get_index_return()
        
        df_final = df_return.merge(df_earnings, on='valuation_date', how='left')
        df_final.dropna(inplace=True)
        
        return df_final

if __name__ == '__main__':
    ife = index_factor_exposure('20171229')
    df = ife.jy_factor_index_exposure_update('国证2000')
    print(df) 