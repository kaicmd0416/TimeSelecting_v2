import os
import pandas as pd
import global_setting.global_dic as glv
import ast
class parameters_getting:
    def __init__(self,mode,signal_name):
        self.mode=mode
        self.signal_name=signal_name
    def singleSignal_prod_parametersGetting(self):
        inputpath = glv.get('signal_parameters_prod')
        df = pd.read_excel(inputpath)
        try:
            damping_list = ast.literal_eval(df[df['signal_name'] == self.signal_name]['damping_list'].tolist()[0])
        except:
            damping_list = []
        rolling_window_list = ast.literal_eval(df[df['signal_name'] == self.signal_name]['rolling_window'].tolist()[0])
        return damping_list, rolling_window_list
    def singleSignal_hist_parametersGetting(self):
        inputpath = glv.get('signal_parameters_history')
        df = pd.read_excel(inputpath)
        try:
            damping_list = ast.literal_eval(df[df['signal_name'] == self.signal_name]['damping_list'].tolist()[0])
        except:
            damping_list = []
        rolling_window_list = ast.literal_eval(df[df['signal_name'] == self.signal_name]['rolling_window'].tolist()[0])
        return damping_list, rolling_window_list
    def signagBooster_prod_parametersGetting(self):
        inputpath = glv.get('signal_parameters_prod')
        df = pd.read_excel(inputpath)
        df = df[df['signal_name'] ==self.signal_name]
        rolling_window = df['boost_rolling_window'].tolist()[0]
        quantile_lower = df['boost_quantile_lower'].tolist()[0]
        boost_mode = df['boost_mode'].tolist()[0]
        return rolling_window,quantile_lower,boost_mode
    def signagBooster_his_parametersGetting(self):
        inputpath = glv.get('signal_parameters_history')
        df = pd.read_excel(inputpath)
        df = df[df['signal_name'] ==self.signal_name]
        try:
            rolling_window = df['boost_rolling_window'].tolist()[0]
        except:
            rolling_window=None
            print('rolling_window为空请检查config_history的配置文件')
        try:
            quantile_lower = df['boost_quantile_lower'].tolist()[0]
        except:
            quantile_lower=None
            print('quantile_lower为空请检查config_history的配置文件')
        try:
            boost_mode = df['boost_mode'].tolist()[0]
        except:
            boost_mode=None
            print('boost_mode为空请检查config_history的配置文件')
        return rolling_window, quantile_lower, boost_mode
    def singleSignal_parametersGetting_main(self):
        if self.mode=='prod':
            damping_list, rolling_window_list=self.singleSignal_prod_parametersGetting()
        else:
            damping_list, rolling_window_list = self. singleSignal_hist_parametersGetting()
        return damping_list,rolling_window_list
    def signalBooster_parametersGetting_main(self):
        if self.mode=='prod':
            rolling_window, quantile_lower, boost_mode=self.signagBooster_prod_parametersGetting()
        else:
            rolling_window, quantile_lower, boost_mode = self.signagBooster_his_parametersGetting()
        return rolling_window, quantile_lower, boost_mode

