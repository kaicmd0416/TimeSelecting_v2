import os
import sys
import pandas as pd
import numpy as np
path = os.getenv('GLOBAL_TOOLSFUNC')
sys.path.append(path)
import global_tools as gt
import global_setting.global_dic as glv
from backtesting.backtesting_tools import Back_testing_processing
class factor_backtesting_main:
    def __init__(self,signal_name,start_date,end_date,cost,mode,x):
        self.df_index_return=self.index_return_withdraw()
        self.signal_name=signal_name
        self.start_date=start_date
        self.end_date=end_date
        self.cost=cost
        self.mode=mode
        self.x=x
        self.start_date=self.start_date_processing()
    def start_date_processing(self):
        inputpath = glv.get('signal_data')
        inputpath  = os.path.join(inputpath, self.mode)
        inputpath=os.path.join(inputpath,self.signal_name)
        input_list=os.listdir(inputpath)
        input_list.sort()
        running_date=input_list[0][-12:-4]
        running_date=gt.strdate_transfer(running_date)
        if self.start_date<running_date:
            start_date=running_date
            print(self.start_date+'目前没有数据，已经自动调整到:'+str(running_date))
        else:
            start_date=self.start_date
        return start_date
    def index_return_withdraw(self):
        df_return = gt.timeSeries_index_return_withdraw()
        df_return = df_return[['valuation_date', '000300.SH', '932000.CSI']]
        df_return.columns=['valuation_date','沪深300','中证2000']
        df_return['沪深300'] = df_return['沪深300'].astype(float)
        df_return['中证2000'] = df_return['中证2000'].astype(float)
        return df_return

    def raw_signal_withdraw(self):
        inputpath = glv.get(self.signal_name+'_signal')
        df = pd.read_excel(inputpath)
        df.rename(columns={self.signal_name:'final_signal'},inplace=True)
        df.dropna(inplace=True)
        start_date=df['valuation_date'].tolist()[0]
        end_date=df['valuation_date'].tolist()[-1]
        working_list=gt.working_days_list(start_date,end_date)
        df=df[df['valuation_date'].isin(working_list)]
        return df
    def raw_signal_withdraw2(self,is_replace=False):
        df=pd.DataFrame()
        inputpath = glv.get('signal_data')
        inputpath=os.path.join(inputpath,self.mode)
        inputpath=os.path.join(inputpath,self.signal_name)
        working_days_list=gt.working_days_list(self.start_date,self.end_date)
        for target_date in working_days_list:
            target_date2=gt.intdate_transfer(target_date)
            inputpath_daily=gt.file_withdraw(inputpath,target_date2)
            df_daily=gt.readcsv(inputpath_daily)
            df_daily=df_daily[df_daily['x']==self.x]
            df=pd.concat([df,df_daily])
        df=df[['valuation_date','final_signal']]
        if is_replace==True:
            df.replace(0.5,None,inplace=True)
            df.fillna(method='ffill',inplace=True)
        return df
    def probability_processing(self,df_signal):
        df_index = self.index_return_withdraw()
        df_signal = df_signal.merge(df_index, on='valuation_date', how='left')
        df_final=pd.DataFrame()
        df_signal['target'] = df_signal['沪深300'] - df_signal['中证2000']
        df_signal.loc[df_signal['target'] > 0, ['target']] = 0
        df_signal.loc[df_signal['target'] < 0, ['target']] = 1
        df_signal['target'] = df_signal['target'].shift(-1)
        df_signal.dropna(inplace=True)
        number_0 = len(df_signal[df_signal['final_signal'] == 0])
        number_1 = len(df_signal[df_signal['final_signal'] == 1])
        number_0_correct = len(df_signal[(df_signal['final_signal'] == 0) & (df_signal['target'] == 0)])
        number_1_correct = len(df_signal[(df_signal['final_signal'] == 1) & (df_signal['target'] == 1)])
        if number_0==0:
            number_0=1
        if number_1==0:
            number_1=1
        pb_0_correct = number_0_correct / number_0
        pb_0_wrong = 1 - pb_0_correct
        pb_1_correct=number_1_correct/number_1
        pb_1_wrong=1-pb_1_correct
        df_final['沪深300']=[pb_0_correct,pb_0_wrong]
        df_final['中证2000']=[pb_1_correct,pb_1_wrong]
        return df_final

    def signal_return_processing(self,df_signal,index_name):
        df_index = self.index_return_withdraw()
        df_index['大小盘等权']=0.5*df_index['沪深300']+0.5*df_index['中证2000']
        df_signal = df_index.merge(df_signal, on='valuation_date', how='left')
        df_signal.dropna(inplace=True)
        df_signal['signal_return'] = 0
        df_signal.loc[df_signal['final_signal'] == 0, ['signal_return']] = \
        df_signal.loc[df_signal['final_signal'] == 0]['沪深300'].tolist()
        df_signal.loc[df_signal['final_signal'] == 1, ['signal_return']] = \
        df_signal.loc[df_signal['final_signal'] == 1]['中证2000'].tolist()
        if index_name=='沪深300':
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5]['沪深300'].tolist()
        elif index_name=='中证2000':
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5]['中证2000'].tolist()
        else:
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5]['大小盘等权'].tolist()
        df_signal['turn_over'] = df_signal['final_signal'] - df_signal['final_signal'].shift(1)
        df_signal['turn_over'] = abs(df_signal['turn_over']) * 2
        df_signal.fillna(method='ffill',inplace=True)
        df_signal.fillna(method='bfill',inplace=True)
        df_signal['turn_over'] = df_signal['turn_over'] * self.cost
        df_signal['portfolio'] = df_signal['signal_return'].astype(float) - df_signal['turn_over']
        df_signal = df_signal[['valuation_date', 'portfolio',index_name]]
        df_signal.rename(columns={index_name:'index'},inplace=True)
        return df_signal
    def backtesting_main(self,is_replace):
        bp = Back_testing_processing(self.df_index_return)
        outputpath = glv.get('backtest_output')
        outputpath=os.path.join(outputpath,self.mode)
        outputpath=os.path.join(outputpath,self.signal_name)
        outputpath=os.path.join(outputpath,'x_'+str(self.x))
        outputpath_prob=os.path.join(outputpath,'positive_negative_probabilities.xlsx')
        gt.folder_creator2(outputpath)
        df_signal=self.raw_signal_withdraw2(is_replace)
        df_prob=self.probability_processing(df_signal)
        df_prob.to_excel(outputpath_prob, index=False)
        for index_name in ['沪深300','中证2000','大小盘等权']:
            if index_name=='大小盘等权':
                index_type='combine'
            else:
                index_type='single'
            df_portfolio = self.signal_return_processing(df_signal, index_name)
            bp.back_testing_history(df_portfolio, outputpath, index_type, index_name, self.signal_name)
        return outputpath
def technical_signal_calculator(df):
    # Initialize result DataFrame
    result_df = pd.DataFrame(columns=['portfolio_name', 'annual_return', 'regression_annual_return', 'max_drawdown', 'longest_new_high_days'])
    
    # Get portfolio columns (excluding valuation_date)
    portfolio_cols = [col for col in df.columns if col != 'valuation_date']
    
    for portfolio in portfolio_cols:
        # Calculate annualized return
        nav0 = df[portfolio].iloc[0]
        navt = df[portfolio].iloc[-1]
        total_return = navt / nav0
        t = len(df)  # Use total number of trading days
        annual_return = (total_return ** (365/t) - 1) * 100  # Convert to annual rate and percentage
        
        # Calculate regression annualized return using ln(navt) - ln(navo) = kt
        k = (np.log(navt) - np.log(nav0)) / t
        regression_annual_return = k * 252 * 100  # Convert to annual rate and percentage
        
        # Calculate maximum drawdown
        rolling_max = df[portfolio].expanding().max()
        drawdowns = (df[portfolio] - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) * 100
        
        # Calculate longest new high days
        rolling_max = df[portfolio].expanding().max()
        new_highs = df[portfolio] >= rolling_max
        longest_streak = 0
        current_streak = 0
        for is_new_high in new_highs:
            if is_new_high:
                current_streak += 1
                longest_streak = max(longest_streak, current_streak)
            else:
                current_streak = 0
        
        # Add results to DataFrame
        result_df = pd.concat([result_df, pd.DataFrame({
            'portfolio_name': [portfolio],
            'annual_return': [annual_return],
            'regression_annual_return': [regression_annual_return],
            'max_drawdown': [max_drawdown],
            'longest_new_high_days': [longest_streak]
        })], ignore_index=True)
    
    return result_df
def singleSingal_backtesting(signal_name,start_date,end_date,mode='test'):
    df_final=pd.DataFrame()
    outputpath_tech = glv.get('backtest_output')
    outputpath_tech=os.path.join(outputpath_tech,mode)
    outputpath_tech=os.path.join(outputpath_tech,signal_name)
    outputpath_tech=os.path.join(outputpath_tech,'综合回测报告.xlsx')
    for x in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        fb = factor_backtesting_main(signal_name, start_date,end_date, 0.00085, 'test', x)
        outputpath=fb.backtesting_main(is_replace=False)
        outputpath=os.path.join(outputpath,str(signal_name)+'_大小盘等权')
        outputpath=os.path.join(outputpath,str(signal_name)+'回测.xlsx')
        df = pd.read_excel(outputpath)
        df = df[['valuation_date', '超额净值']]
        df.columns = ['valuation_date', 'x_'+str(x)]
        if x==0.55:
            df_final=df
        else:
            df_final=df_final.merge(df,on='valuation_date',how='left')
    df_technical=technical_signal_calculator(df_final)
    df_technical.to_excel(outputpath_tech,index=False)
if __name__ == "__main__":
    singleSingal_backtesting('Shibor_9M','2016-01-01','2025-06-04')


