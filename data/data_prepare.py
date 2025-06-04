import os
import sys
import pandas as pd
path = os.getenv('GLOBAL_TOOLSFUNC')
sys.path.append(path)
import global_tools as gt
import global_setting.global_dic as glv
from matplotlib import pyplot as plt
global source,config_path
source=glv.get('source')
config_path=glv.get('config_path')
class data_prepare:
    def df1_transformer(self, df1, type):
        # 获取唯一的估值日期
        valuation_date_list = df1['valuation_date'].unique().tolist()

        # 将数据透视化：organization成为列，value作为值
        if type == 'indexOther':
            df1_pivot = df1.pivot(index='valuation_date', columns='organization', values='value')
        elif type == 'macroData':
            df1_pivot = df1.pivot(index='valuation_date', columns='name', values='value')
        elif type == 'intData':
            df1_pivot = df1.pivot(index='valuation_date', columns='code', values='value')
        elif type=='indexData':
            df1_pivot = df1.pivot(index='valuation_date', columns='code', values='value')
        elif type=='stockData':
            df1_pivot = df1.pivot(index='valuation_date', columns='code', values='value')
        else:
            raise ValueError

        # 重置索引，使valuation_date成为列
        df1_pivot = df1_pivot.reset_index()

        # 确保valuation_date是第一列
        cols = df1_pivot.columns.tolist()
        cols.remove('valuation_date')
        df1_pivot = df1_pivot[['valuation_date'] + cols]

        return df1_pivot
    def index_name_mapping(self,index_name):
        if index_name=='沪深300':
            return '000300.SH'
        elif index_name=='中证2000':
            return '932000.CSI'
        elif index_name=='中证500':
            return '000905.SH' 
        elif index_name=='中证1000':
            return '000852.SH'
        elif index_name=='中证800':
            return '932000.CSI'
        elif index_name=='中证A500':
            return '000510.CSI'
        elif index_name=='上证50':
            return '000016.SH'
        elif index_name=='国证2000':
            return '399303.SZ'
        elif index_name=='金融等权':
            return '000076.SH'
        else:
            raise ValueError('index_name must be 沪深300 or 中证2000 or 中证500 or 中证1000 or 中证800 or 中证A500 or 上证50 or 国证2000')
    #shibor
    def raw_shibor(self,period):
        inputpath = glv.get('raw_Shibor')
        if source=='sql':
            inputpath=str(inputpath)+" WHERE organization = 'Shibor' AND type = 'close'"
        df1 = gt.data_getting(inputpath,config_path)
        if source=='sql':
            df1 = self.df1_transformer(df1, 'macroData')
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        signal_name='Shibor_'+period
        if signal_name not in df1.columns:
            raise ValueError('period must be 2W or 9M')
        df1=df1[['valuation_date',signal_name]]
        return df1
    #国债
    def raw_bond(self,period):
        inputpath = glv.get('raw_Bond')
        if source == 'sql':
            inputpath = str(inputpath) + " WHERE organization = 'ChinaGovernmentBonds' AND type = 'close'"
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = self.df1_transformer(df1, 'macroData')
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        signal_name='CGB_'+period
        if signal_name not in df1.columns:
            raise ValueError('period must be 3Y or 10Y')
        df1=df1[['valuation_date',signal_name]]
        return df1
    #国开债
    def raw_ZZGK(self,period):
        inputpath = glv.get('raw_ZZGK')
        if source == 'sql':
            inputpath = str(inputpath) + " WHERE organization = 'ChinaDevelopmentBankBonds' AND type = 'close'"
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = self.df1_transformer(df1, 'macroData')
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        signal_name='CDBB_'+period
        if signal_name not in df1.columns:
            raise ValueError('period must be 3M or9M or 1Y or 5Y or 10Y')
        df1=df1[['valuation_date',signal_name]]
        return df1
    #中债中短
    def raw_ZZZD(self,period):
        inputpath= glv.get('raw_ZZZD')
        if source == 'sql':
            inputpath = str(inputpath) + " WHERE organization = 'ChinaMediumTermNotes' AND type = 'close'"
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = self.df1_transformer(df1, 'macroData')
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        signal_name='CMTN_'+period
        if signal_name not in df1.columns:
            raise ValueError('period must be 9M or 3M or 5Y ')
        df1=df1[['valuation_date',signal_name]]
        return df1
    #M1 and M2
    def raw_M1M2(self,signal_name):
        inputpath = glv.get('raw_M1M2')
        if source == 'sql':
            inputpath = str(inputpath) + " WHERE organization = 'M1M2' AND type = 'close'"
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = self.df1_transformer(df1, 'macroData')
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))   
        if signal_name not in df1.columns:
            raise ValueError('type must be M1 or M2')
        df1=df1[['valuation_date',signal_name]]
        return df1
    #美国方面
    #美元指数
    def raw_usdx(self):
        inputpath = glv.get('raw_USDX')
        if source == 'sql':
            inputpath = str(inputpath) + " WHERE organization = 'USDollar' AND type = 'close'"
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = self.df1_transformer(df1, 'macroData')
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        return df1
    def raw_DJUS(self):
        inputpath_D=glv.get('raw_DJUS')
        df1_DJUS=gt.readcsv(inputpath_D)
        df1_DJUS.columns=['valuation_date','DJUS','DJUS_close']
        df1_DJUS.fillna(0,inplace=True)
        df1_DJUS=df1_DJUS[['valuation_date','DJUS']]
        return df1_DJUS
    def raw_NDAQ(self):
        inputpath_N = glv.get('raw_NDAQ')
        df1_NDAQ = gt.readcsv(inputpath_N)
        df1_NDAQ.columns = ['valuation_date', 'NDAQ', 'NDAQ_close']
        df1_NDAQ.fillna(0,inplace=True)
        df1_NDAQ=df1_NDAQ[['valuation_date','NDAQ']]
        return df1_NDAQ
    #风险因子方面：
    def raw_index_earningsyield(self):
        inputpath=glv.get('raw_indexFactor')
        if source == 'sql':
            inputpath = str(inputpath) + " WHERE type = 'earningsyield'"
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = self.df1_transformer(df1, 'indexOther')
        df1['valuation_date']=pd.to_datetime(df1['valuation_date'])
        df1['valuation_date']=df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1.rename(columns={'gz2000':'gz2000Earningsyield','hs300':'hs300Earningsyield','zz1000':'zz1000Earningsyield'},inplace=True)
        df1=df1[['valuation_date','hs300Earningsyield','gz2000Earningsyield','zz1000Earningsyield']]
        df1_final=df1.dropna()
        df1_final['difference_earningsyield']=df1_final['hs300Earningsyield']-df1_final['gz2000Earningsyield']-df1_final['zz1000Earningsyield']
        df1_final=df1_final[['valuation_date','difference_earningsyield']]
        return df1_final
    def raw_index_growth(self):
        inputpath=glv.get('raw_indexFactor')
        if source == 'sql':
            inputpath = str(inputpath) + " WHERE type = 'growth'"
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = self.df1_transformer(df1, 'indexOther')
        df1['valuation_date']=pd.to_datetime(df1['valuation_date'])
        df1['valuation_date']=df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1.rename(
            columns={'gz2000': 'gz2000Growth', 'hs300': 'hs300Growth', 'zz1000': 'zz1000Growth'},
            inplace=True)
        df1=df1[['valuation_date','hs300Growth','gz2000Growth','zz1000Growth']]
        df1_final=df1.dropna()
        df1_final['difference_Growth']=df1_final['hs300Growth']-df1_final['gz2000Growth']-df1_final['zz1000Growth']
        df1_final=df1_final[['valuation_date','difference_Growth']]
        return df1_final
    def raw_CPI_withdraw(self):
        inputpath = glv.get('raw_CPI')
        if source == 'sql':
            inputpath = str(inputpath) + " WHERE organization = 'CPI' AND type = 'close'"
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = self.df1_transformer(df1, 'macroData')
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1.columns=['valuation_date','CPI']
        return df1
    def raw_PPI_withdraw(self):
        inputpath = glv.get('raw_PPI')
        if source == 'sql':
            inputpath = str(inputpath) + " WHERE organization = 'PPI' AND type = 'close'"
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = self.df1_transformer(df1, 'macroData')
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1.columns=['valuation_date','PPI']
        return df1
    def raw_PMI_withdraw(self):
        inputpath = glv.get('raw_PMI')
        if source == 'sql':
            inputpath = str(inputpath) + " WHERE organization = 'PMI' AND type = 'close'"
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = self.df1_transformer(df1, 'macroData')
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1.columns=['valuation_date','PMI']
        return df1
    #资金因子方面:
    def raw_LHBProportion_withdraw(self):
        inputpath=glv.get('raw_LHBProportion')
        df1 = gt.data_getting(inputpath, config_path)
        df1['valuation_date']=pd.to_datetime(df1['valuation_date'])
        df1['valuation_date']=df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1=df1[['valuation_date','LHBProportion']]
        return df1
    def raw_NetLeverageBuying_withdraw(self):
        inputpath=glv.get('raw_NLBPDifference')
        if source == 'sql':
            inputpath = str(inputpath) + " WHERE type = 'NetLeverageBuying' And organization='NetLeverageAMTProportion_difference'"
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = self.df1_transformer(df1, 'indexOther')
        df1['valuation_date']=pd.to_datetime(df1['valuation_date'])
        df1['valuation_date']=df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1=df1[['valuation_date','NetLeverageAMTProportion_difference']]
        df1.columns=['valuation_date','NLBP_difference']
        df1['NLBP_difference']=df1['NLBP_difference'].shift(1)
        return df1
    def raw_LargeOrder_withdraw(self):
        inputpath=glv.get('raw_LargeOrder')
        if source == 'sql':
            inputpath = str(
                inputpath) + " WHERE type = 'LargeOrderInflow'"
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = self.df1_transformer(df1, 'indexOther')
        df1['valuation_date']=pd.to_datetime(df1['valuation_date'])
        df1['valuation_date']=df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        return df1
    #股票方面:
    def raw_stockClose_withdraw(self):
        inputpath = glv.get('raw_StockClose')
        df1=gt.readcsv(inputpath)
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        return df1
    #技术指标方面:
    #指数方面
    def raw_index_return(self,index_name):
        index_code=self.index_name_mapping(index_name)
        inputpath = glv.get('raw_indexReturn')
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1=df1[['valuation_date','code','pct_chg']]
            df1=df1[df1['code']==index_code]
            df1=df1[['valuation_date','pct_chg']]
        else:
            df1=df1[['valuation_date',index_code]]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1.columns=['valuation_date',index_name]
        return df1
    def raw_index_volume(self,index_name):
        index_code=self.index_name_mapping(index_name)
        inputpath = glv.get('raw_indexVolume')
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1=df1[['valuation_date','code','volume']]
            df1=df1[df1['code']==index_code]
            df1=df1[['valuation_date','volume']]
        else:
            df1=df1[['valuation_date',index_code]]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1.columns=['valuation_date',index_name]
        return df1
    def raw_index_turnover(self,index_name):
        index_code=self.index_name_mapping(index_name)
        inputpath = glv.get('raw_indexTurnOver')
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = df1[['valuation_date', 'code', 'turn_over']]
            df1 = df1[df1['code'] == index_code]
            df1 = df1[['valuation_date', 'turn_over']]
        else:
            df1 = df1[['valuation_date', index_code]]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1.columns=['valuation_date',index_name]
        return df1
    def raw_index_amt(self,index_name):
        index_code=self.index_name_mapping(index_name)
        inputpath = glv.get('raw_indexAMT')
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = df1[['valuation_date', 'code', 'amt']]
            df1 = df1[df1['code'] == index_code]
            df1 = df1[['valuation_date', 'amt']]
        else:
            df1 = df1[['valuation_date', index_code]]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1.columns=['valuation_date',index_name]
        return df1
    def raw_index_close(self,index_name):
        index_code=self.index_name_mapping(index_name)
        inputpath = glv.get('raw_indexClose')
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = df1[['valuation_date', 'code', 'close']]
            df1 = df1[df1['code'] == index_code]
            df1 = df1[['valuation_date', 'close']]
        else:
            df1 = df1[['valuation_date', index_code]]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1.columns=['valuation_date',index_name]
        return df1
    def raw_index_high(self,index_name):
        index_code=self.index_name_mapping(index_name)
        inputpath = glv.get('raw_indexHigh')
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = df1[['valuation_date', 'code', 'high']]
            df1 = df1[df1['code'] == index_code]
            df1 = df1[['valuation_date', 'high']]
        else:
            df1 = df1[['valuation_date', index_code]]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1.columns=['valuation_date',index_name]
        return df1
    def raw_index_low(self,index_name):
        index_code=self.index_name_mapping(index_name)
        inputpath = glv.get('raw_indexLow')
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = df1[['valuation_date', 'code', 'low']]
            df1 = df1[df1['code'] == index_code]
            df1 = df1[['valuation_date', 'low']]
        else:
            df1 = df1[['valuation_date', index_code]]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1.columns=['valuation_date',index_name]
        return df1
    def raw_index_open(self,index_name):
        index_code=self.index_name_mapping(index_name)
        inputpath = glv.get('raw_indexOpen')
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = df1[['valuation_date', 'code', 'open']]
            df1 = df1[df1['code'] == index_code]
            df1 = df1[['valuation_date', 'open']]
        else:
            df1 = df1[['valuation_date', index_code]]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1.columns=['valuation_date',index_name]
        return df1
    def index_return_withdraw(self,index_list):
        df1_return=gt.timeSeries_index_return_withdraw()
        df1_return=df1_return[['valuation_date']+index_list]
        df1_return[index_list]=df1_return[index_list].astype(float)
        return df1_return
    def BankMomentum_withdraw(self):
        df_finance=self.raw_index_return('金融等权')
        df_gz2000=self.raw_index_return('国证2000')
        df1=df_finance.merge(df_gz2000,on='valuation_date',how='left')
        df1.set_index('valuation_date', inplace=True)
        df1 = (1 + df1).cumprod()
        df1['difference'] = df1['金融等权'] - df1['国证2000']
        df1.reset_index(inplace=True)
        df1 = df1[['valuation_date', 'difference']]
        return df1
    #target_index
    def target_index(self):
        df1_return=self.index_return_withdraw(['沪深300','中证2000'])
        df1_return=df1_return[['valuation_date','沪深300','中证2000']]
        df1_return.set_index('valuation_date',inplace=True)
        df1_return=df1_return.astype(float)
        df1_return=(1+df1_return).cumprod()
        df1_return['target_index']=df1_return['沪深300']/df1_return['中证2000']
        df1_return.reset_index(inplace=True)
        df1_return=df1_return[['valuation_date','target_index','沪深300','中证2000']]
        return df1_return
    def target_index_candle(self):
        df_close_hs300=self.raw_index_close('沪深300')
        df_close_zz2000=self.raw_index_close('国证2000')
        df_high_hs300=self.raw_index_high('沪深300')
        df_high_zz2000=self.raw_index_high('国证2000')
        df_low_hs300=self.raw_index_low('沪深300')
        df_low_zz2000=self.raw_index_low('国证2000')
        df1_close=df_close_hs300.merge(df_close_zz2000,on='valuation_date',how='left')
        df1_high=df_high_hs300.merge(df_high_zz2000,on='valuation_date',how='left')
        df1_low=df_low_hs300.merge(df_low_zz2000,on='valuation_date',how='left')
        df1_close.columns=['valuation_date', '000300.SH_close', '399303.SZ_close']
        df1_high.columns=['valuation_date','000300.SH_high','399303.SZ_high']
        df1_low.columns = ['valuation_date', '000300.SH_low', '399303.SZ_low']
        df1_hl=df1_high.merge(df1_low,on='valuation_date',how='left')
        df1_final=df1_close.merge(df1_hl,on='valuation_date',how='left')
        df1_final['close']=df1_final['000300.SH_close']-df1_final['399303.SZ_close']
        df1_final['high']=df1_final['000300.SH_low']-df1_final['399303.SZ_high']
        df1_final['low']=df1_final['000300.SH_high']-df1_final['399303.SZ_low']
        df1_final=df1_final[['valuation_date','high','close','low']]
        return df1_final
    def target_index_candle2(self):
        df_close_hs300 = self.raw_index_close('沪深300')
        df_close_zz2000 = self.raw_index_close('中证2000')
        df_high_hs300 = self.raw_index_high('沪深300')
        df_high_zz2000 = self.raw_index_high('中证2000')
        df_low_hs300 = self.raw_index_low('沪深300')
        df_low_zz2000 = self.raw_index_low('中证2000')
        df_open_hs300=self.raw_index_open('沪深300')
        df_open_zz2000=self.raw_index_open('中证2000')
        df1_open=df_open_hs300.merge(df_open_zz2000,on='valuation_date',how='left')
        df1_close = df_close_hs300.merge(df_close_zz2000, on='valuation_date', how='left')
        df1_high = df_high_hs300.merge(df_high_zz2000, on='valuation_date', how='left')
        df1_low = df_low_hs300.merge(df_low_zz2000, on='valuation_date', how='left')
        df1_open.columns = ['valuation_date', '000300.SH_open', '932000.CSI_open']
        df1_close.columns=['valuation_date', '000300.SH_close', '932000.CSI_close']
        df1_high.columns=['valuation_date','000300.SH_high','932000.CSI_high']
        df1_low.columns = ['valuation_date', '000300.SH_low', '932000.CSI_low']
        df1_hl=df1_high.merge(df1_low,on='valuation_date',how='left')
        df1_final=df1_close.merge(df1_hl,on='valuation_date',how='left')
        df1_final=df1_final.merge(df1_open,on='valuation_date',how='left')
        df1_final['open']=df1_final['000300.SH_open']-df1_final['932000.CSI_open']
        df1_final['close']=df1_final['000300.SH_close']-df1_final['932000.CSI_close']
        df1_final['high']=df1_final['000300.SH_high']-df1_final['932000.CSI_low']
        df1_final['low']=df1_final['000300.SH_low']-df1_final['932000.CSI_high']
        df1_final=df1_final[['valuation_date','open','high','close','low']]
        return df1_final
    def future_difference_withdraw(self):
        inputpath=glv.get('raw_futureDifference')
        if source == 'sql':
            inputpath = str(
                inputpath) + " WHERE type = 'FutureDifference' AND organization='indexFuture_difference'"
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1 = self.df1_transformer(df1, 'indexOther')
        df1.rename(columns={'indexFuture_difference':'difference_future'},inplace=True)
        df1=df1[['valuation_date','difference_future']]
        return df1
    def raw_rrscoreDifference(self):
        inputpath=glv.get('raw_rrscoreDifference')
        if source == 'sql':
            inputpath = str(
                inputpath) + " WHERE type = 'rrIndexScore'"
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            print(df1)
            df1 = self.df1_transformer(df1, 'indexOther')
        df1['rrscoreDifference']=df1['hs300']-df1['gz2000']
        df1=df1[['valuation_date','rrscoreDifference']]
        return df1
    def raw_vix_withdraw(self):
        inputpath=glv.get('raw_vix')
        if source == 'sql':
            inputpath = str(
                inputpath) + " WHERE vix_type = 'TimeWeighted'"
        df1 = gt.data_getting(inputpath, config_path)
        if source == 'sql':
            df1.rename(columns={'ch_vix':'value'},inplace=True)
            df1 = self.df1_transformer(df1, 'indexOther')
        df1=df1[['valuation_date','hs300','zz1000']]
        df1.fillna(method='ffill',inplace=True)
        return df1
if __name__ == "__main__":
    dp = data_prepare()
    df1=dp.raw_vix_withdraw()
    print(df1)

