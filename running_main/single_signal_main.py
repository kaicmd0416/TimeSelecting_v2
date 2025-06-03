import pandas as pd
from numpy import *
import global_setting.global_dic as glv
from data.data_prepare import data_prepare
from data.data_processing import data_processing
from factor_processing.signal_constructing import signal_construct, factor_processing
import global_tools_func.global_tools as gt
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
        if self.signal_name=='Shibor_2W':
            df=self.dp.raw_shibor(period='2W')
            sc_mode='mode_1'
        elif self.signal_name=='Shibor_9M':
            df = self.dp.raw_shibor(period='9M')
            sc_mode = 'mode_3'
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
        return df,sc_mode
    def signal_combination(self,x):
        if x<0.5:
            return 0
        elif x==0.5:
            return 0.5
        else:
            return 1
    def df_transformer(self,df):
        df_final=pd.DataFrame()
        date_list=df['valuation_date'].unique().tolist()
        portfolio_list=df['portfolio'].unique().tolist()
        df_final['valuation_date']=date_list
        for name in portfolio_list:
            final_signal=df[df['portfolio']==name]['signal'].tolist()
            df_final[name]=final_signal
        return df_final
    def monthly_factor_processing(self):
        df=self.df_signal.copy()
        df.set_index('valuation_date',inplace=True,drop=True)
        df=df.shift(20)
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        return df
    def start_date_processing(self,start_date,rolling_window_list):
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
        max_ma=max(rolling_window_list)
        if len(slice_df)<max_ma:
            index=index+max_ma-len(slice_df)
            start_date=df_signal.iloc[index]['valuation_date']
        return start_date

    def signal_main(self,start_date,end_date,damping_list,rolling_window_list):
        signal_name=self.signal_name
        outputpath = glv.get('signal_data')
        outputpath=os.path.join(outputpath,self.mode)
        outputpath=os.path.join(outputpath,signal_name)
        gt.folder_creator2(outputpath)
        start_date=self.start_date_processing(start_date, rolling_window_list)
        working_days_list = gt.working_days_list(start_date, end_date)
        for date in working_days_list:
            date2=gt.intdate_transfer(date)
            outputpath2=os.path.join(outputpath,str(signal_name)+'_'+str(date2)+'.csv')
            df = pd.DataFrame()
            final_signal_list = []
            parameters_list = []
            if len(damping_list)>0:
               for damping in damping_list:
                   parameters = self.signal_name + '_' + str(damping)
                   daily_df = self.fp.Kal_processing(self.df_signal, date, damping)
                   for rolling_window in rolling_window_list:
                       parameters2 = parameters + '_' + str(rolling_window)
                       signal = self.sc.Dp_MA_signal_construct(daily_df, rolling_window,self.sc_mode)
                       parameters_list.append(parameters2)
                       final_signal_list.append(signal)
            else:
                parameters = self.signal_name
                daily_df = self.fp.slice_processing(self.df_signal, date)
                if self.sc_mode=='mode_6' or self.sc_mode=='mode_7':
                    rolling_list = []
                    sorted_windows = sorted(rolling_window_list)
                    if len(rolling_window_list) >= 5:
                        # 最小的数字与其他更大的数字组合
                        min_window = sorted_windows[0]
                        rolling_list.extend([[min_window, window] for window in sorted_windows[1:]])
                        # 第二小的数字与其他更大的数字组合
                        second_min_window = sorted_windows[1]
                        rolling_list.extend([[second_min_window, window] for window in sorted_windows[2:]])
                    else:
                        # 如果长度小于3，只用最小的数字组合
                        min_window = sorted_windows[0]
                        rolling_list.extend([[min_window, window] for window in sorted_windows[1:]])
                    for rolling_window in rolling_list:
                        parameters2 = parameters + '_' + str(rolling_window[0]) + '_' + str(rolling_window[1])
                        signal = self.sc.MA_difference_signal_construct(daily_df, rolling_window, self.sc_mode)
                        parameters_list.append(parameters2)
                        final_signal_list.append(signal)
                elif self.sc_mode=='mode_8':
                    signal=self.sc.LargeOrder_difference_signal_construct(daily_df)
                    parameters_list.append(parameters)
                    final_signal_list.append(signal)
                elif self.sc_mode=='mode_9':
                    daily_df2=self.fp.slice_processing2(self.df_signal,date)
                    signal=self.sc.Monthly_effect_signal_construct(daily_df2)
                    parameters_list.append(parameters)
                    final_signal_list.append(signal)
                elif self.sc_mode=='mode_10' or self.sc_mode=='mode_11':
                    daily_df2=self.fp.slice_processing_Monthly(self.df_signal, date)
                    signal = self.sc.Monthly_factor_signal_construct(daily_df2,self.sc_mode)
                    parameters_list.append(parameters)
                    final_signal_list.append(signal)
                elif self.sc_mode=='mode_12':
                    signal = self.sc.stock_HL_signal_construct(daily_df,self.signal_name)
                    parameters_list.append(parameters)
                    final_signal_list.append(signal)
                elif self.sc_mode=='mode_13':
                    signal=self.sc.technical_signal_construct(daily_df,self.signal_name)
                    parameters_list.append(parameters)
                    final_signal_list.append(signal)
                elif self.sc_mode=='mode_14' or self.sc_mode=='mode_15':
                    if self.sc_mode=='mode_14':
                        signal=self.sc.single_direction_decision(daily_df)
                    else:
                        signal=self.sc.single_direction_decision2(daily_df)
                    parameters_list.append(parameters)
                    final_signal_list.append(signal)
                else:
                    for rolling_window in rolling_window_list:
                        parameters2 = parameters + '_' + str(rolling_window)
                        if self.signal_name=='M1M2':
                            daily_df=self.fp.slice_processing_Monthly(self.df_signal, date)
                        signal = self.sc.M1M2_signal_construct(daily_df, rolling_window) if self.signal_name=='M1M2' else self.sc.MA_signal_construct(daily_df, rolling_window, self.sc_mode)
                        parameters_list.append(parameters2)
                        final_signal_list.append(signal)
            df['signal'] = final_signal_list
            df['portfolio'] = parameters_list
            df.set_index('portfolio',inplace=True)
            df=df.T
            df['valuation_date']=date
            df=df[['valuation_date']+df.columns.tolist()[:-1]]
            if len(df)>0:
                 df.to_csv(outputpath2,index=False)
            else:
                print(self.signal_name+'在'+str(date)+'的时候，更新存在问题')
if __name__ == "__main__":
    ssm=single_signal_main(signal_name='RelativeReturn_std',mode='test')
    ssm.signal_main(start_date='2015-01-01',end_date='2025-01-03',damping_list=None,rolling_window_list=[20,30,40,60,90,120])