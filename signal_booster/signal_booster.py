import global_setting.global_dic as glv
import pandas as pd
from numpy import *
import global_tools_func.global_tools as gt
import numpy as np
import os
class signal_booster:
    def __init__(self,signal_name,start_date,end_date, rolling_window, quantile_lower,mode_type,mode):
        self.save=save
        self.start_date=start_date
        self.end_date=end_date
        self.signal_name = signal_name
        self.rolling_window = rolling_window
        self.quantile_lower = quantile_lower
        self.mode = mode
        self.mode_type=mode_type
        self.df_index_return = self.index_return_withdraw()
        self.start_date,self.running_date=self.start_date_processing()
    def start_date_processing(self):
        inputpath = glv.get('signal_data')
        inputpath = os.path.join(inputpath, self.mode)
        inputpath = os.path.join(inputpath, self.signal_name)
        input_list=os.listdir(inputpath)
        input_list.sort()
        if len(input_list)==0:
            start_date='2016-01-01'
            running_date='2016-01-01'
        else:
            df = pd.DataFrame()
            df['input_name'] = input_list
            df['valuation_date'] = df['input_name'].apply(lambda x: str(x)[-12:-4])
            df['valuation_date'] = df['valuation_date'].apply(lambda x: gt.strdate_transfer(x))
            slice_df = df[df['valuation_date'] < self.start_date]
            if len(slice_df) == 0:
                index = df.index.tolist()[0]
            else:
                index = slice_df.index.tolist()[-1]
            max_ma = self.rolling_window
            if len(slice_df) < max_ma:
                index = index + max_ma - len(slice_df)
                index=int(index)
                start_date = df.iloc[index]['valuation_date']
                print('输入的start_date有问题，已经自动调整到' + str(start_date))
                running_date = df.iloc[0]['valuation_date']
            else:
                index2 = index - max_ma
                start_date = df.iloc[index + 1]['valuation_date']
                running_date = df.iloc[index2]['valuation_date']
        return start_date,running_date
    def index_return_withdraw(self):
        df_return = gt.timeSeries_index_return_withdraw()
        df_return = df_return[['valuation_date', '沪深300', '中证2000']]
        df_return['沪深300'] = df_return['沪深300'].astype(float)
        df_return['中证2000'] = df_return['中证2000'].astype(float)
        df_return['difference']=df_return['沪深300']-df_return['中证2000']
        df_return['target_value']=0
        df_return.loc[df_return['difference']<0,['target_value']]=1
        df_return=df_return[['valuation_date', '沪深300', '中证2000','target_value']]
        df_return['index_return'] = 0.5 * df_return['沪深300'] + 0.5 * df_return['中证2000']
        return df_return
    def signal_withdraw(self):
        df_final=pd.DataFrame()
        inputpath = glv.get('signal_data')
        inputpath = os.path.join(inputpath, self.mode)
        inputpath = os.path.join(inputpath, self.signal_name)
        working_days_list=gt.working_days_list(self.running_date,self.end_date)
        for available_date in working_days_list:
            available_date2=gt.intdate_transfer(available_date)
            inputpath_daily=gt.file_withdraw(inputpath,available_date2)
            df = gt.readcsv(inputpath_daily)
            df_final=pd.concat([df_final,df])
        df_index = self.index_return_withdraw()
        list_1=df_index.columns.tolist()
        df_final.dropna(inplace=True)
        list_2=df_final.columns.tolist()
        portfolio_list=list(set(list_2)-set(list_1))
        df_final=df_final.merge(df_index,on='valuation_date',how='left')
        return df_final,portfolio_list
    def rank_list(self,x):
        sorted_list = sorted(x)
        serial_numbers = [sorted_list.index(item) for item in x]
        return serial_numbers
    def signal_analysis(self,df_signal2,portfolio_list):
        df_result = pd.DataFrame()
        cor_pro_list=[]
        sharpe_list=[]
        df_signal2.dropna(inplace=True)
        for portfolio in portfolio_list:
            df_signal=df_signal2[['valuation_date', '沪深300', '中证2000','target_value','index_return',portfolio]]
            df_signal['difference'] = df_signal[portfolio] - df_signal['target_value']
            cor_pro = len(df_signal[df_signal['difference'] == 0]) / len(df_signal)
            df_signal['turn_over'] = df_signal[portfolio] - df_signal[portfolio].shift(1)
            df_signal['turn_over'] = abs(df_signal['turn_over']) * 2
            df_signal.loc[df_signal[portfolio] == 0, [portfolio]] = \
                df_signal.loc[df_signal[portfolio] == 0]['沪深300'].tolist()
            df_signal.loc[df_signal[portfolio] == 1, [portfolio]] = \
                df_signal.loc[df_signal[portfolio] == 1]['中证2000'].tolist()
            df_signal.fillna(method='ffill', inplace=True)
            df_signal.fillna(method='bfill', inplace=True)
            df_signal['turn_over'] = df_signal['turn_over'] * 0.00085
            df_signal[portfolio] = df_signal[portfolio].astype(float) - df_signal['turn_over']
            df_signal['ex_return']=df_signal[portfolio]-df_signal['index_return']
            annual_returns=(((1 + df_signal['ex_return']).cumprod()).tolist()[-1] - 1) * 252 / len(df_signal)
            vol = df_signal['ex_return'].std() * np.sqrt(252)
            sharpe = annual_returns / vol
            cor_pro_list.append(cor_pro)
            sharpe_list.append(sharpe)
        df_result['signal_name']=portfolio_list
        df_result['correct_prob']=cor_pro_list
        df_result['sharpe_ratio']=sharpe_list
        df_result['cor_score']=self.rank_list(df_result['correct_prob'].tolist())
        df_result['sharpe_score'] = self.rank_list(df_result['sharpe_ratio'].tolist())
        df_result['combine_score']=(df_result['cor_score']+df_result['sharpe_score'])/2
        df_result['valuation_date']=df_signal2['valuation_date'].tolist()[-1]
        df_result=df_result[['valuation_date','signal_name','cor_score','sharpe_score','combine_score']]
        return df_result
    def rolling_signal_selecting(self,df_signal,portfolio_list,target_date,rolling_window,quantile_lower,mode_type):
        df_signal2=df_signal[df_signal['valuation_date']<target_date]
        df_signal2.reset_index(inplace=True,drop=True)
        if len(df_signal2)>rolling_window:
            rolling_window=int(rolling_window)
            df_signal2=df_signal2.iloc[-rolling_window:]
            df_result=self.signal_analysis(df_signal2,portfolio_list)
            df_result['rolling_window']= df_result['signal_name'].apply(lambda x: x[x.rfind('_')+1:])
            if mode_type=='pro':
                score_name='cor_score'
                df_result.sort_values(by='cor_score',ascending=False,inplace=True)
            elif mode_type=='sharpe':
                score_name = 'sharpe_score'
                df_result.sort_values(by='sharpe_score', ascending=False, inplace=True)
            elif mode_type=='combine':
                score_name = 'combine_score'
                df_result.sort_values(by='combine_score', ascending=False, inplace=True)
            else:
                raise ValueError
            df_result=df_result[df_result[score_name]>=df_result[score_name].quantile(quantile_lower)]
            if len(df_result)!=0:
                signal_name_list = df_result['signal_name'].tolist()
                if len(df_result)<5 or 'combine' in self.signal_name:
                    final_list=df_signal[df_signal['valuation_date'] == target_date][signal_name_list].mean(axis=1).tolist()
                else:
                    final_list = []
                    for i in df_result['rolling_window'].unique().tolist():
                        slice_df_result = df_result[df_result['rolling_window'] == i]
                        max = slice_df_result[score_name].max()
                        signal_name_list = slice_df_result[slice_df_result[score_name] == max]['signal_name'].tolist()
                        single_signal = \
                            df_signal[df_signal['valuation_date'] == target_date][signal_name_list].mean(
                                axis=1).tolist()[0]
                        final_list.append(single_signal)
                signal = mean(final_list)
                if signal > 0.5:
                    return 1
                elif signal == 0.5:
                    return 0.5
                else:
                    return 0
            else:
                rest_name=df_signal.columns.tolist()[1:]
                signal=df_signal[df_signal['valuation_date'] == target_date][rest_name].mean(axis=1).tolist()[0]
                return signal
        else:
             return None
    def signal_boosting_main(self):
        if self.signal_name=='combine' and self.mode=='prod':
            outputpath = glv.get('signal_combine')
        else:
            outputpath = glv.get('signal_booster_output')
            outputpath = os.path.join(outputpath, self.mode)
        boost_name = self.signal_name + '_' + str(int(self.rolling_window)) + 'D_' + str(
            int(10 * self.quantile_lower)) + '_' + self.mode_type
        outputpath = os.path.join(outputpath, boost_name)
        gt.folder_creator2(outputpath)
        df_signal, portfolio_list = self.signal_withdraw()
        working_days_list=gt.working_days_list(self.start_date,self.end_date)
        for target_date in working_days_list:
            print(target_date)
            target_date2=gt.intdate_transfer(target_date)
            df_final = pd.DataFrame()
            signal_list = []
            date_list = []
            signal=self.rolling_signal_selecting(df_signal, portfolio_list, target_date, self.rolling_window, self.quantile_lower,
                                     self.mode_type)
            date_list.append(target_date)
            signal_list.append(signal)
            df_final['valuation_date'] = date_list
            df_final[self.signal_name] = signal_list
            df_final.dropna(inplace=True)
            outputpath2=os.path.join(outputpath,self.signal_name+'_'+str(target_date2)+'.csv')
            if len(df_final)>0:
                df_final.to_csv(outputpath2,index=False)
            else:
                print(self.signal_name+'在'+str(target_date2)+'更新有问题')
if __name__ == "__main__":
    sb=signal_booster('Stock_HL','2015-06-01','2025-03-20',1,0.9,'combine','test')
    sb.signal_boosting_main()