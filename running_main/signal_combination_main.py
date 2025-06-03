import global_setting.global_dic as glv
import pandas as pd
from signal_booster.signal_booster import signal_booster
import global_tools_func.global_tools as gt
import os
import yaml
from KCluster.K_cluster import calculate_autocorrelation
from portfolio.portfolio_construction import portfolio_updating
class signalCombination:
    def __init__(self,start_date,end_date,mode):
        self.start_date=start_date
        self.end_date=end_date
        self.mode=mode
    def generate_boost_name(self, df):
        """Generate boost name column based on signal_name and rolling_window columns"""
        df['boost_name'] = df.apply(lambda row: row['signal_name'] + '_' + str(int(row['boost_rolling_window'])) + 'D_' +
                                  str(int(10 * row['boost_quantile_lower'])) + '_' + row['boost_mode'], axis=1)
        return df
    def activation_signalname_withdraw(self):
        inputpath = glv.get('signal_parameters_prod')
        df = pd.read_excel(inputpath)
        df = df[df['activate'] == 1]
        df=self.generate_boost_name(df)
        signal_name_list=df['boost_name'].tolist()
        signal_name_list2=df['signal_name'].tolist()
        return signal_name_list,signal_name_list2
    def load_cluster_config(self):
        """加载聚类配置文件"""
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        current_dir = os.path.join(current_dir, 'config_project')
        config_path = os.path.join(current_dir, 'kcluster.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    def process_cluster_signals(self, df, cluster_config):
        """处理聚类信号"""
        result = pd.DataFrame()
        result['valuation_date'] = df['valuation_date']
        
        # 对每个聚类进行处理
        for cluster_id, factors in cluster_config.items():
            if not factors:  # 如果该聚类没有因子，跳过
                continue
                
            # 获取该聚类所有因子的值
            cluster_values = df[factors]
            
            # 计算均值
            cluster_mean = cluster_values.mean(axis=1)
            
            # 根据阈值进行离散化
            cluster_signal = cluster_mean.apply(lambda x: 0 if x < 0.5 else (0.5 if x == 0.5 else 1))
            
            # 添加到结果DataFrame
            result[f'K{cluster_id}'] = cluster_signal
            
        return result

    def signalCombination_main(self,mode):
        inputpath= glv.get('signal_booster_output')
        inputpath = os.path.join(inputpath, self.mode)
        gt.folder_creator2(inputpath)
        outputpath = glv.get('signal_data')
        outputpath = os.path.join(outputpath, self.mode)
        outputpath = os.path.join(outputpath, 'combine')
        gt.folder_creator2(outputpath)
        try:
            outputlist=os.listdir(outputpath)
        except:
            outputlist=[]
        if len(outputlist)==0:
            self.start_date='2016-01-01'
        signal_name_list,signal_name_list2 = self.activation_signalname_withdraw()
        working_days_list = gt.working_days_list(self.start_date, self.end_date)
        
        # 加载聚类配置
        cluster_config = self.load_cluster_config()
        
        for available_date in working_days_list:
            df_final=pd.DataFrame()
            df_final['valuation_date']=[available_date]
            available_date2=gt.intdate_transfer(available_date)
            daily_outputpath=os.path.join(outputpath,'combine_'+str(available_date2)+'.csv')
            
            # 读取原始信号
            for signal_name,signal_name2 in zip(signal_name_list,signal_name_list2):
                inputpath_daily=os.path.join(inputpath,signal_name)
                available_date2 = gt.intdate_transfer(available_date)
                inputpath_daily = gt.file_withdraw(inputpath_daily, available_date2)
                df = gt.readcsv(inputpath_daily)
                if len(df)!=0:
                    signal=df[signal_name2].tolist()[0]
                    df_final[signal_name2]=[signal]
                else:
                    df_final[signal_name2]=[0.5]
                    print('在'+available_date+'缺少'+signal_name2+'的数据')
            
            # 处理聚类信号
            cluster_results = self.process_cluster_signals(df_final, cluster_config)
            # 保存结果
            cluster_results.to_csv(daily_outputpath,index=False)
            
            if mode=='mean':
                outputpath_mean = glv.get('signal_combine')
                outputpath_mean = os.path.join(outputpath_mean, 'combine_mean')
                gt.folder_creator2(outputpath_mean)
                daily_outputpath_mean=os.path.join(outputpath_mean,'combine_'+str(available_date2)+'.csv')
                
                # 使用所有K因子计算均值
                cluster_results.set_index('valuation_date', inplace=True)
                k_columns = [col for col in cluster_results.columns if col.startswith('K')]
                
                def calculate_weighted_mean(row):
                    k23_value = row['K23']
                    if k23_value == 0.5:
                        # 如果K23=0.5，所有因子权重相等
                        return row[k_columns].mean()
                    else:
                        # 如果K23!=0.5，K23权重为0.5，其他因子平分剩余0.5
                        other_columns = [col for col in k_columns if col != 'K23']
                        other_weight = 0.5 / len(other_columns)
                        return 0.5 * k23_value + sum(row[col] * other_weight for col in other_columns)
                
                cluster_results['combine_value'] = cluster_results.apply(calculate_weighted_mean, axis=1)
                cluster_results.reset_index(inplace=True)
                
                # 保存均值结果
                result_mean = cluster_results[['valuation_date', 'combine_value']]
                def combinesign_decision(x):
                    if x>0.5:
                        return 1
                    elif x==0.5:
                        return 0.5
                    else:
                        return 0
                result_mean['combine_sign'] = result_mean['combine_value'].apply(lambda x: combinesign_decision(x))
                result_mean.to_csv(daily_outputpath_mean,index=False)
                try:
                    pu = portfolio_updating(available_date)
                    pu.portfolio_saving_main()
                except:
                   print(str(available_date)+'portfolio更新有误')
        
        # # 合并所有日期的数据并计算自相关系数
        # combined_df = combine_all_signals(outputpath)
        # if combined_df is not None:
        #     calculate_autocorrelation(combined_df)
        #
        if mode=='boost':
            sb = signal_booster('combine', self.start_date, self.end_date, 90,
                                0.2, 'combine', self.mode)
            sb.signal_boosting_main()



if __name__ == "__main__":
    scm = signalCombination('2016-01-01', '2025-05-01', 'prod')
    scm.signalCombination_main('mean')
    # sb = signal_booster('combine', '2016-01-01','2025-04-10', 45,
    #                     0.1, 'combine', 'prod')
    # sb.signal_boosting_main('mean')



