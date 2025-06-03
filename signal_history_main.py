from running_main.signal_construct_main import signal_constructing_main
import global_setting.global_dic as glv
import pandas as pd
import global_tools_func.global_tools as gt
def history_config_withdraw():
    inputpath = glv.get('signal_parameters_history')
    df = pd.read_excel(inputpath)
    return df
def history_main(): #触发这个
    df=history_config_withdraw()
    df['start_date']=df['start_date'].apply(lambda x:gt.strdate_transfer(x))
    df['end_date'] = df['end_date'].apply(lambda x: gt.strdate_transfer(x))
    for i in range(len(df)):
        signal_name=df['signal_name'].tolist()[i]
        start_date=df['start_date'].tolist()[i]
        end_date = df['end_date'].tolist()[i]
        mode=df['mode'].tolist()[i]
        mode_his=df['mode_his'].tolist()[i]
        scm=signal_constructing_main(signal_name,mode_his,start_date,end_date)
        scm.signalConstruct_main(mode)
if __name__ == "__main__":
    history_main()