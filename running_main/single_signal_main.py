import pandas as pd
from numpy import *
from data.data_prepare import data_prepare
from data.data_processing import data_processing
from factor_processing.signal_constructing import signal_construct, factor_processing
import os
import sys
path = os.getenv('GLOBAL_TOOLSFUNC')
sys.path.append(path)
import global_tools as gt
import global_setting.global_dic as glv
import os
class single_signal_main:
    def __init__(self,signal_name,mode):
        self.signal_name=signal_name
        self.mode = mode
        self.dp=data_prepare()
        self.dpro=data_processing()
        self.fp=factor_processing()
        self.sc=signal_construct()
        self.df_signal,self.sc_mode=self.raw_data_preparing()
        self.df_signal['valuation_date']=self.df_signal['valuation_date'].apply(lambda x: gt.strdate_transfer(x))
    def final_signal_construction(self,final_signal,x):
        if final_signal > x:
            return 1  # 沪深300
        elif final_signal<(1-x):
            return 0
        else:
            return 0.5  # 中证2000
    def raw_data_preparing(self):
        #mode_1为带damping的正向direction
        #mode_2为带damping的反向direction
        #mode_3为无damping的正向direction
        #mode_4为无damping的反向direction
        #mode_5为M1M2专用
        #mode_6为MA_difference的正向direction
        #mode_7为MA_difference的反向direction
        #mode_8为大单因子专用
        #mode_9为月度效应专用
        #mode_10为月度因子专用
        #mode_11为月度反向因子专用
        #mode_12为stock_HL专用
        #mode_13为技术指标专用
        #mode_14 只有两列单纯判断大于0的正向direction
        # mode_15 只有两列单纯判断大于0的反向向direction
        if self.signal_name=='Shibor_2W': #正
            df=self.dp.raw_shibor(period='2W')
            sc_mode='mode_1'
        elif self.signal_name=='Shibor_9M': #正
            df = self.dp.raw_shibor(period='9W')
            sc_mode = 'mode_1'
        elif self.signal_name=='Bond_3Y':
            df=self.dp.raw_bond(period='3Y')
            sc_mode = 'mode_1'
        elif self.signal_name=='Bond_10Y':
            df=self.dp.raw_bond(period='10Y')
            sc_mode = 'mode_1'
        elif self.signal_name=='USDX':
            df=self.dp.raw_usdx()
            sc_mode = 'mode_2'
        elif self.signal_name=='CreditSpread_3M':
            df=self.dpro.credit_spread_3M()
            sc_mode = 'mode_6'
        elif self.signal_name == 'CreditSpread_9M':
            df = self.dpro.credit_spread_9M()
            sc_mode = 'mode_6'
        elif self.signal_name == 'CreditSpread_5Y':
            df = self.dpro.credit_spread_5Y()
            sc_mode = 'mode_6'
        elif self.signal_name=='TermSpread_9Y':
            df=self.dpro.term_spread_9Y()
            sc_mode='mode_2'
        elif self.signal_name=='M1M2':
            df=self.dpro.M1M2()
            sc_mode = 'mode_5'  #专属mode
        elif self.signal_name=='USStock':
            df=self.dpro.US_stock()
            sc_mode = 'mode_1'
        elif self.signal_name=='RelativeVolume_std':
            df=self.dpro.relativeVolume_std()
            sc_mode='mode_3'
        elif self.signal_name=='RelativeReturn_std':
            df=self.dpro.relativeReturn_std()
            sc_mode='mode_3'
        elif self.signal_name=='EarningsYield_Reverse':
            df=self.dp.raw_index_earningsyield()
            sc_mode='mode_7'
        elif self.signal_name=='Growth':
            df=self.dp.raw_index_growth()
            sc_mode='mode_6'
        elif self.signal_name=='LHBProportion':
            df=self.dp.raw_LHBProportion_withdraw()
            sc_mode='mode_7'
        elif self.signal_name=='NLBP_difference':
            df=self.dp.raw_NetLeverageBuying_withdraw()
            sc_mode='mode_7'
        elif self.signal_name=='LargeOrder_difference':
            df=self.dpro.LargeOrder_difference()
            sc_mode='mode_8'
        elif self.signal_name=='Monthly_effect':
            df=self.dpro.monthly_effect()
            sc_mode='mode_9'
        elif self.signal_name=='CPI':
            df=self.dp.raw_CPI_withdraw()
            sc_mode = 'mode_10'
        elif self.signal_name=='PPI':
            df=self.dp.raw_PPI_withdraw()
            sc_mode = 'mode_10'
        elif self.signal_name=='PMI':
            df=self.dp.raw_PMI_withdraw()
            sc_mode = 'mode_11'
        elif self.signal_name=='Stock_HL':
            df = self.dpro.stock_highLow()
            sc_mode = 'mode_12'
        elif self.signal_name=='Rsi_difference':
            df=self.dpro.stock_rsi()
            sc_mode='mode_12'
        elif self.signal_name=='RaisingTrend_proportion':
            df=self.dpro.stock_trend()
            sc_mode='mode_7'
        elif self.signal_name=='TargetIndex_MACD':
            df=self.dpro.targetIndex_MACD()
            sc_mode='mode_13'
        elif self.signal_name=='TargetIndex_RSI':
            df=self.dpro.targetIndex_RSI()
            sc_mode='mode_13'
        elif self.signal_name=='TargetIndex_BBANDS':
            df=self.dpro.targetIndex_BOLLBAND()
            sc_mode='mode_13'
        elif self.signal_name=='TargetIndex_MOMENTUM':
            df=self.dpro.TargetIndex_MOMENTUM()
            sc_mode='mode_6'
        elif self.signal_name=='TargetIndex_KDJ':
            df=self.dpro.TargetIndex_KDJ()
            sc_mode='mode_13'
        elif self.signal_name=='TargetIndex_PSA':
            df=self.dpro.TargetIndex_PSA()
            sc_mode='mode_13'
        elif self.signal_name=='TargetIndex_MOMENTUM2':
            df=self.dpro.TargetIndex_MOMENTUM2()
            sc_mode='mode_13'
        elif self.signal_name=='TargetIndex_REVERSE':
            df=self.dpro.TargetIndex_MOMENTUM3()
            sc_mode='mode_13'
        elif self.signal_name=='Future_difference':
            df=self.dp.future_difference_withdraw()
            sc_mode='mode_7'
        elif self.signal_name=='RRScore_difference':
            df=self.dpro.raw_rrscoreDifference()
            sc_mode='mode_6'
        elif self.signal_name=='Bank_Momentum':
            df=self.dp.BankMomentum_withdraw()
            sc_mode='mode_6'
        else:
            print('signal_name还没有纳入系统')
            raise ValueError
        df.dropna(inplace=True)
        return df,sc_mode
    def monthly_factor_processing(self):
        df=self.df_signal.copy()
        df.set_index('valuation_date',inplace=True,drop=True)
        df=df.shift(20)
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        return df
    def start_date_processing(self,start_date):
        self.df_signal.reset_index(inplace=True,drop=True)
        if self.signal_name=='M1M2' or self.signal_name=='CPI'or self.signal_name=='PPI' or self.signal_name=='PMI':
            df_signal=self.monthly_factor_processing()
        else:
            df_signal=self.df_signal.copy()
        slice_df=df_signal[df_signal['valuation_date']<start_date]
        if len(slice_df)==0:
            index=df_signal.index.tolist()[0]
        else:
            slice_df.reset_index(inplace=True,drop=True)
            index=slice_df.index.tolist()[-1]
        max_ma=250
        if len(slice_df)<max_ma:
            index=index+max_ma-len(slice_df)
            start_date=df_signal.iloc[index]['valuation_date']
        return start_date

    def signal_main(self,start_date,end_date):
        x_list=[0.55,0.6,0.65,0.7,0.75,0.8]
        signal_name=self.signal_name
        outputpath = glv.get('signal_data')
        outputpath=os.path.join(outputpath,str(self.mode))
        outputpath=os.path.join(outputpath,str(signal_name))
        gt.folder_creator2(outputpath)
        start_date=self.start_date_processing(start_date)
        working_days_list = gt.working_days_list(start_date, end_date)
        for date in working_days_list:
            df_final = pd.DataFrame()
            final_signal_list = []
            date2=gt.intdate_transfer(date)
            outputpath_daily=os.path.join(outputpath,str(signal_name)+'_'+date2+'.csv')
            signal_list = []
            daily_df = self.fp.slice_processing(self.df_signal, date)
            if self.sc_mode=='mode_1' or self.sc_mode=='mode_2':
                # Define short and long windows
                short_windows = [1, 5, 10, 15, 20]
                long_windows = [5, 10, 15, 20, 30, 40, 60, 90, 120, 180, 250]
                
                # Create combinations where short window is less than long window
                rolling_list = []
                for short_window in short_windows:
                    for long_window in long_windows:
                        if short_window < long_window:
                            rolling_list.append([short_window, long_window])
                for rolling_window in rolling_list:
                    signal = self.sc.MA_difference_signal_construct(daily_df, rolling_window, self.sc_mode)
                    signal_list.append(signal)
            final_signal=mean(signal_list)
            for x in x_list:
                final_signal2=self.final_signal_construction(final_signal,x)
                final_signal_list.append(final_signal2)
            df_final['x']=x_list
            df_final['final_signal']=final_signal_list
            df_final['valuation_date']=date
            df_final=df_final[['valuation_date','final_signal','x']]
            if len(df_final)>0:
                 df_final.to_csv(outputpath_daily,index=False)
if __name__ == "__main__":
    ssm=single_signal_main(signal_name='Bond_10Y',mode='test')
    ssm.signal_main(start_date='2015-01-01',end_date='2025-06-04')