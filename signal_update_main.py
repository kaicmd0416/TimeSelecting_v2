from running_main.signal_construct_main import signal_constructing_main
import global_setting.global_dic as glv
import pandas as pd
import global_tools_func.global_tools as gt
from running_main.signal_combination_main import signalCombination
from datetime import date
import datetime
from data_check.data_check import DataChecker
import os
from data.raw_data_processing import raw_data_processing
def signal_config_withdraw():
    inputpath = glv.get('signal_dic')
    df = pd.read_excel(inputpath)
    df=df[df['activate']==1]
    return df
def generate_boost_name(df):
        """Generate boost name column based on signal_name and rolling_window columns"""
        df['boost_name'] = df.apply(lambda row: row['signal_name'] + '_' + str(int(row['boost_rolling_window'])) + 'D_' +
                                  str(int(10 * row['boost_quantile_lower'])) + '_' + row['boost_mode'], axis=1)
        return df
def target_date_decision():
        # inputpath = glv.get('time_tools_config')
        # df_config = pd.read_excel(inputpath, sheet_name='critical_time')
        # critical_time = df_config[df_config['zoom_name'] == 'time_1']['critical_time'].tolist()[0]
        # critical_time = critical_time.strftime("%H:%M")
        critical_time='20:00'
        if gt.is_workday2() == True:
            today = date.today()
            next_day = gt.next_workday_calculate(today)
            time_now = datetime.datetime.now().strftime("%H:%M")
            if time_now >= critical_time:
                return next_day
            else:
                today = gt.strdate_transfer(today)
                return today
        else:
            today = date.today()
            next_day = gt.next_workday_calculate(today)
            return next_day
def update_main(): #触发这个
    df=signal_config_withdraw()
    df=generate_boost_name(df)
    target_date=target_date_decision()
    print(target_date)
    start_date=target_date
    for i in range(2):
        start_date=gt.last_workday_calculate(start_date)
    checker = DataChecker(target_date)
    for i in range(len(df)):
        signal_name=df['signal_name'].tolist()[i]
        boost_name=df['boost_name'].tolist()[i]
        status=checker.check_raw_data(signal_name)
        if status=='error':
            print(signal_name+'在过去一年中出现数据缺失的问题')
        else:
            status2,update_date=checker.check_signal_data(boost_name) 
            if status2=='error' and update_date is None:
                 print(boost_name+'在过去一年中出现数据缺失的问题')
            else:
                if update_date>start_date:
                    scm=signal_constructing_main(signal_name,'prod',start_date,target_date)
                else:
                    scm=signal_constructing_main(signal_name,'prod',update_date,target_date)
                scm.signal_constructing_prod_main()
    if status=='normal'  and status2=='normal':
        outputpath = glv.get('signal_data')
        outputpath = os.path.join(outputpath, 'prod')
        outputpath = os.path.join(outputpath, 'combine')
        try:
            inputlist=os.listdir(outputpath)
        except:
            inputlist=[]
        if len(inputlist)==0:
            start_date='2016-01-01'
        scb=signalCombination(start_date,target_date,'prod')
        scb.signalCombination_main('mean')
        available_date=gt.last_workday_calculate(target_date)
        rdp=raw_data_processing()
        if start_date!='2016-01-01':
              rdp.rawData_savingMain(available_date,available_date)
        else:
              rdp.rawData_savingMain(start_date,available_date)

if __name__ == "__main__":
    for i in ['NLBP_difference']:
        scm = signal_constructing_main(i, 'prod', '2016-01-01', '2025-05-06')
        scm.signal_constructing_prod_main()
    # rdp = raw_data_processing()
    # rdp.rawData_savingMain('2015-07-01', '2025-04-30')
    # scb = signalCombination('2015-06-01', '2025-03-20', 'prod')
    # scb.signalCombination_main()
    # update_main()



