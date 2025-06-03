import pandas as pd
from matplotlib import pyplot as plt

import global_tools_func.global_tools as gt
import global_setting.global_dic as glv
from data.data_prepare import data_prepare
from data.data_processing import data_processing
import pandas_ta as ta
import os
from sklearn.linear_model import LinearRegression
import numpy as np

class raw_data_processing:
    def __init__(self):
        dp=data_prepare()
        self.dp=dp
        self.df_candle=self.target_index_candle2()
        self.df_candle['pre_close']=self.df_candle['close'].shift(1)
        self.df_candle['return']=(self.df_candle['close']-self.df_candle['pre_close'])/self.df_candle['pre_close']
        self.df_psa=self.TargetIndex_PSA()
        self.df_kdj=self.TargetIndex_KDJ()
        self.df_bbd=self.targetIndex_BOLLBAND()
        self.df_macd=self.targetIndex_MACD()
        self.df_rsi=self.targetIndex_RSI()
        self.df_ma=self.MA_processing()
        self.df_indexreturn=dp.index_return_withdraw()
        self.df_volume=self.volume_std()
        self.df_return_std=self.return_std()
        self.df_momentum=self.Momentum_processing()
        self.df_final=self.final_df_processing()

    def target_index_candle2(self):
        inputpath_close=glv.get('raw_indexClose')
        inputpath_high=glv.get('raw_indexHigh')
        inputpath_low=glv.get('raw_indexLow')
        inputpath_open=glv.get('raw_indexOpen')
        
        # 读取原始数据
        df_close=gt.readcsv(inputpath_close)
        df_high = gt.readcsv(inputpath_high)
        df_low = gt.readcsv(inputpath_low)
        df_open=gt.readcsv(inputpath_open)
        
        # 选择需要的列
        columns = ['valuation_date','000300.SH','932000.CSI','000852.SH','399303.SZ']
        df_close = df_close[columns]
        df_high = df_high[columns]
        df_low = df_low[columns]
        df_open = df_open[columns]
        
        # 对每个DataFrame分别处理
        for df in [df_close, df_high, df_low, df_open]:
            # 1. 首先找出932000.CSI为空的区间
            mask_932_empty = df['932000.CSI'].isna()
            
            # 2. 对于每个空区间，检查399303.SZ和000852.SH的值
            for start_idx in df[mask_932_empty].index:
                # 如果932000.CSI为空
                if pd.isna(df.loc[start_idx, '932000.CSI']):
                    # 检查399303.SZ是否有值
                    if not pd.isna(df.loc[start_idx, '399303.SZ']):
                        # 使用399303.SZ的数据训练模型
                        valid_data = df[~df['399303.SZ'].isna() & ~df['932000.CSI'].isna()]
                        if len(valid_data) > 0:
                            # 使用当前DataFrame的数据训练模型
                            X = valid_data['399303.SZ'].values.reshape(-1, 1)
                            y = valid_data['932000.CSI'].values
                            model = LinearRegression()
                            model.fit(X, y)
                            # 预测并填充
                            df.loc[start_idx, '932000.CSI'] = model.predict([[df.loc[start_idx, '399303.SZ']]])[0]
                    
                    # 如果399303.SZ为空但000852.SH有值
                    elif not pd.isna(df.loc[start_idx, '000852.SH']):
                        # 使用000852.SH的数据训练模型
                        valid_data = df[~df['000852.SH'].isna() & ~df['932000.CSI'].isna()]
                        if len(valid_data) > 0:
                            # 使用当前DataFrame的数据训练模型
                            X = valid_data['000852.SH'].values.reshape(-1, 1)
                            y = valid_data['932000.CSI'].values
                            model = LinearRegression()
                            model.fit(X, y)
                            # 预测并填充
                            df.loc[start_idx, '932000.CSI'] = model.predict([[df.loc[start_idx, '000852.SH']]])[0]
        
        # 选择需要的列并重命名
        df_open = df_open[['valuation_date','000300.SH','932000.CSI']].copy()
        df_close = df_close[['valuation_date','000300.SH','932000.CSI']].copy()
        df_high = df_high[['valuation_date','000300.SH','932000.CSI']].copy()
        df_low = df_low[['valuation_date','000300.SH','932000.CSI']].copy()
        
        # 重命名列
        df_open.columns = ['valuation_date', '000300.SH_open', '932000.CSI_open']
        df_close.columns = ['valuation_date', '000300.SH_close', '932000.CSI_close']
        df_high.columns = ['valuation_date', '000300.SH_high', '932000.CSI_high']
        df_low.columns = ['valuation_date', '000300.SH_low', '932000.CSI_low']
        
        # 确保所有DataFrame的日期格式一致
        for df in [df_open, df_close, df_high, df_low]:
            df['valuation_date'] = pd.to_datetime(df['valuation_date']).dt.strftime('%Y-%m-%d')
        
        # 合并数据
        df_final = df_close.copy()
        df_final = df_final.merge(df_high, on='valuation_date', how='left')
        df_final = df_final.merge(df_low, on='valuation_date', how='left')
        df_final = df_final.merge(df_open, on='valuation_date', how='left')
        
        # 计算差值
        df_final['open'] = df_final['000300.SH_open'] - df_final['932000.CSI_open']
        df_final['close'] = df_final['000300.SH_close'] - df_final['932000.CSI_close']
        df_final['high'] = df_final['000300.SH_high'] - df_final['932000.CSI_low']
        df_final['low'] = df_final['000300.SH_low'] - df_final['932000.CSI_high']
        
        # 选择最终列
        df_final = df_final[['valuation_date', 'open', 'high', 'close', 'low']]
        
        # 按日期排序
        df_final = df_final.sort_values('valuation_date')
        
        return df_final
    def targetIndex_MACD(self):
        df=self.df_candle
        df.dropna(inplace=True)
        df=df[['valuation_date','close']]
        df['MACD'] = ta.macd(df['close'])['MACD_12_26_9']
        df['MACD_h'] = ta.macd(df['close'])['MACDh_12_26_9']
        df['MACD_s'] = ta.macd(df['close'])['MACDs_12_26_9']
        df.dropna(inplace=True)
        df.reset_index(inplace=True,drop=True)
        df.drop(columns='close',inplace=True)
        return df
    def targetIndex_RSI(self):
        df=self.df_candle
        df.dropna(inplace=True)
        df=df[['valuation_date','close']]
        df['RSI'] = ta.rsi(df['close'],14)
        df.dropna(inplace=True)
        df.reset_index(inplace=True,drop=True)
        df = df[['valuation_date', 'RSI']]
        return df
    def targetIndex_BOLLBAND(self):
        df=self.df_candle
        df.dropna(inplace=True)
        df = df[['valuation_date', 'close']]
        # 计算布林带指标
        # 使用正确的方式获取布林带指标
        bbands = ta.bbands(df['close'], length=20,std=1.5)
        if bbands is not None:
            df['upper'] = bbands['BBU_20_1.5']
            df['middle'] = bbands['BBM_20_1.5']
            df['lower'] = bbands['BBL_20_1.5']
        else:
            # 如果bbands返回None，设置默认值
            df['upper'] = df['close']
            df['middle'] = df['close']
            df['lower'] = df['close']
            print("Warning: Bollinger Bands calculation returned None. Using default values.")
        df=df[['valuation_date','upper','middle','lower']]
        df.columns = ['valuation_date', 'BBU_20_1.5', 'BBM_20_1.5', 'BBL_20_1.5']
        df.dropna(inplace=True)
        df.reset_index(inplace=True,drop=True)
        return df
    def TargetIndex_KDJ(self):
        df=self.df_candle
        df.dropna(inplace=True)
        df_kdj = ta.kdj(df['high'], df['low'], df['close'])
        # 将 KDJ 指标合并到原 DataFrame 中
        df = pd.concat([df, df_kdj], axis=1)
        df.dropna(inplace=True)
        df = df[['valuation_date', 'K_9_3', 'D_9_3', 'J_9_3']]
        return df
    def TargetIndex_PSA(self):
        df=self.df_candle
        df.dropna(inplace=True)
        # 计算抛物线指标
        psar = ta.psar(df['high'], df['low'])
        # 将抛物线指标合并到原 DataFrame 中
        df = pd.concat([df, psar], axis=1)
        df = df[['valuation_date', 'close', 'PSARl_0.02_0.2', 'PSARs_0.02_0.2']]
        df.loc[df['PSARl_0.02_0.2'].isna(), ['PSARl_0.02_0.2']] = df[df['PSARl_0.02_0.2'].isna()][
            'PSARs_0.02_0.2']
        df = df[['valuation_date', 'PSARl_0.02_0.2']]
        df.columns = ['valuation_date', 'psa']
        return df
    def MA_processing(self):
        df_ma=self.df_candle.copy()
        df_ma=df_ma[['valuation_date','close']]
        for i in [5,15,30,45,60,90,120]:
            df_ma['ma_'+str(i)]=df_ma['close'].rolling(i).mean()
        df_ma.drop(columns='close',inplace=True)
        return df_ma
    def Momentum_processing(self):
        df_ma=self.df_candle.copy()
        df_ma=df_ma[['valuation_date','close']]
        for i in [5,15,30,45,60,90,120]:
            df_ma['momentum_'+str(i)]=df_ma['close'].rolling(i).mean()
        df_ma.drop(columns='close',inplace=True)
        return df_ma
    def volume_processing(self):
        inputpath=glv.get('raw_indexVolume')
        df=gt.readcsv(inputpath)
        
        # 选择需要的列
        columns = ['valuation_date','000300.SH','932000.CSI','000852.SH','399303.SZ']
        df = df[columns]
        
        # 处理932000.CSI的空值
        mask_932_empty = df['932000.CSI'].isna()
        for start_idx in df[mask_932_empty].index:
            if pd.isna(df.loc[start_idx, '932000.CSI']):
                # 优先使用399303.SZ的值
                if not pd.isna(df.loc[start_idx, '399303.SZ']):
                    df.loc[start_idx, '932000.CSI'] = df.loc[start_idx, '399303.SZ']
                # 如果399303.SZ为空，使用000852.SH的值
                elif not pd.isna(df.loc[start_idx, '000852.SH']):
                    df.loc[start_idx, '932000.CSI'] = df.loc[start_idx, '000852.SH']
        
        # 选择需要的列并重命名
        df = df[['valuation_date', '000300.SH', '932000.CSI']]
        df.columns = ['valuation_date', '沪深300', '中证2000']
        
        # 确保日期格式一致
        df['valuation_date'] = pd.to_datetime(df['valuation_date']).dt.strftime('%Y-%m-%d')
        
        # 计算成交量差异
        df['volume_sum'] = df['沪深300'] + df['中证2000']
        df['volume_difference'] = df['沪深300'] - df['中证2000']
        
        # 选择最终列
        df = df[['valuation_date', 'volume_sum', 'volume_difference', '沪深300', '中证2000']]
        df.columns=['valuation_date', 'volume_sum', 'volume_difference', 'volume_hs300', 'volume_zz2000']
        
        # 按日期排序
        df = df.sort_values('valuation_date')
        
        return df
    def volume_std(self):
        df = self.volume_processing()
        df['hs300_volume_std']=df['volume_hs300'].rolling(40).std()
        df['zz2000_volume_std'] = df['volume_zz2000'].rolling(40).std()
        df['relative_volume_std']=df['hs300_volume_std']/df['zz2000_volume_std']
        return df
    def return_std(self):
        df=self.df_indexreturn
        df['hs300_return_std']=df['沪深300'].rolling(40).std()
        df['zz2000_return_std'] = df['中证2000'].rolling(40).std()
        df['relative_return_std'] = df['hs300_return_std'] / df['zz2000_return_std']
        df = df[['valuation_date', 'hs300_return_std', 'zz2000_return_std', 'relative_return_std']]
        return df
    def Bank_Momentum(self):
        inputpath = glv.get('raw_indexFinanceDifference')
        df = gt.readcsv(inputpath)
        df.set_index('valuation_date', inplace=True)
        df = (1 + df).cumprod()
        df['difference'] = df['finance_return'] - df['gz2000_return']
        df=df[['difference','finance_return']]
        df.columns=['FinanceDifference','Finance']
        df.reset_index(inplace=True)
        return df
    def USStock(self):
        df_djus = self.dp.raw_DJUS()
        df_djus = df_djus[['valuation_date', 'DJUS']]
        df_ndaq = self.dp.raw_NDAQ()
        df_ndaq = df_ndaq[['valuation_date', 'NDAQ']]
        df = df_djus.merge(df_ndaq, on='valuation_date', how='left')
        df.set_index('valuation_date', inplace=True)
        df = (1 + df).cumprod()
        df['relative_usratio'] = df['DJUS'] / df['NDAQ']
        df.reset_index(inplace=True)
        return df
    def final_df_processing(self):
        df_final=self.df_candle.merge(self.df_psa,on='valuation_date',how='left')
        df_final = df_final.merge(self.df_kdj, on='valuation_date', how='left')
        df_final = df_final.merge(self.df_bbd, on='valuation_date', how='left')
        df_final = df_final.merge(self.df_macd, on='valuation_date', how='left')
        df_final = df_final.merge(self.df_rsi, on='valuation_date', how='left')
        df_final = df_final.merge(self.df_ma, on='valuation_date', how='left')
        df_final = df_final.merge(self.df_volume, on='valuation_date', how='left')
        df_final = df_final.merge(self.df_return_std, on='valuation_date', how='left')
        df_final = df_final.merge(self.df_momentum, on='valuation_date', how='left')
        df_final.dropna(inplace=True)
        df_final.to_csv('D:\OneDrive\Data_prepared_test\data_timeSeries\\raw_timeselecting_exposure.csv',index=False)
        return df_final
    def rawData_savingMain(self,start_date,end_date):
        outputpath=glv.get('raw_signal_output')
        gt.folder_creator2(outputpath)
        df_final=self.df_final.copy()
        df_final.dropna(inplace=True)
        working_days_list=gt.working_days_list(start_date,end_date)
        for available_date in working_days_list:
            print(available_date)
            available_date2=gt.intdate_transfer(available_date)
            outputpath_daily=os.path.join(outputpath,'raw_signalData_'+str(available_date2)+'.csv')
            df_daily=df_final[df_final['valuation_date']==available_date]
            df_daily.to_csv(outputpath_daily,index=False)
if __name__ == "__main__":
    dfp=raw_data_processing()
    dfp.rawData_savingMain('2005-01-04','2025-05-28')



