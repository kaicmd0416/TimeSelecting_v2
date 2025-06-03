import os
import pandas as pd
from datetime import datetime, timedelta
import global_setting.global_dic as glv
from running_main.single_signal_main import single_signal_main
from signal_booster.signal_booster import signal_booster
from parameters_getting.parameters_getting import parameters_getting
from backtesting.factor_backtesting import factor_backtesting_main
import global_tools_func.global_tools as gt
class signal_constructing_main:
    def __init__(self,signal_name,mode,start_date,end_date):
        #mode是直接run 不储存数据，或者储存数据
        self.signal_name=signal_name
        self.mode=mode
        self.original_date=start_date
        self.start_date=start_date
        self.end_date=end_date
        self.pg=parameters_getting(mode,signal_name)
    def signal_constructing_prod_main(self):
        damping_list,rolling_window_list=self.pg.singleSignal_parametersGetting_main()
        boost_rolling_window, quantile_lower, boost_mode=self.pg.signalBooster_parametersGetting_main()
        ssm = single_signal_main(self.signal_name, self.mode)
        ssm.signal_main(self.start_date, self.end_date, damping_list, rolling_window_list)
        sb = signal_booster(self.signal_name,self.start_date,self.end_date, boost_rolling_window, quantile_lower,boost_mode,self.mode)
        sb.signal_boosting_main()
    def signal_constructing_his_main(self,mode_his):
        damping_list, rolling_window_list = self.pg.singleSignal_parametersGetting_main()
        boost_rolling_window, quantile_lower, boost_mode = self.pg.signalBooster_parametersGetting_main()
        ssm = single_signal_main(self.signal_name, self.mode)
        if mode_his=='part_1': #只跑原始数part
            ssm.signal_main(self.start_date, self.end_date, damping_list, rolling_window_list)
        elif mode_his=='part_2': #只跑boost
            try:
               sb = signal_booster(self.signal_name, self.start_date, self.end_date, boost_rolling_window, quantile_lower,
                                boost_mode, self.mode)
               sb.signal_boosting_main()
            except:
                print(str(self.signal_name)+'因子原始数据没有准备好')
        elif mode_his=='part_3': #只跑backtest
            boost_name = self.signal_name + '_' + str(int(boost_rolling_window)) + 'D_' + str(
            int(10 * quantile_lower)) + '_' + boost_mode
            try:
                fb = factor_backtesting_main(boost_name, self.start_date, self.end_date, 0.00085, self.mode)
                fb.backtesting_main()
            except:
                print(str(boost_name)+'因子增强数据没有准备好')

        elif mode_his=='part_4':#跑boost和backtest
            boost_name = self.signal_name + '_' + str(int(boost_rolling_window)) + 'D_' + str(
            int(10 * quantile_lower)) + '_' + boost_mode
            try:
                sb = signal_booster(self.signal_name, self.start_date, self.end_date, boost_rolling_window,
                                    quantile_lower,
                                    boost_mode, self.mode)
                sb.signal_boosting_main()
            except:
                print(str(self.signal_name) + '因子原始数据没有准备好')
            try:
                fb = factor_backtesting_main(boost_name, self.start_date, self.end_date, 0.00085, self.mode)
                fb.backtesting_main()
            except:
                print(str(boost_name)+'因子增强数据没有准备好')
        elif mode_his=='all':
            print('开始运行原始因子')
            boost_name = self.signal_name + '_' + str(int(boost_rolling_window)) + 'D_' + str(
            int(10 * quantile_lower)) + '_' + boost_mode
            ssm.signal_main(self.start_date, self.end_date, damping_list, rolling_window_list)
            print('开始运行因子增强')
            sb = signal_booster(self.signal_name, self.start_date, self.end_date, boost_rolling_window, quantile_lower,
                                boost_mode, self.mode)
            sb.signal_boosting_main()
            print('开始运行因子回测')
            fb = factor_backtesting_main(boost_name, self.start_date, self.end_date, 0.00085, self.mode)
            fb.backtesting_main()
        else:
            print('当前mode:'+str(mode_his)+'不在脚本范围内')
            raise TypeError
    def signalConstruct_main(self,mode_his):
        if self.mode=='prod':
            self.signal_constructing_prod_main()
        else:
            if mode_his==None:
                print(mode_his+'为空')
                raise ValueError
            self.signal_constructing_his_main(mode_his)


if __name__ == "__main__":
    scm=signal_constructing_main(signal_type='Bond',mode='run',start_date='2016-01-01',end_date='2025-01-03')
    scm.signal_constructing_main()





