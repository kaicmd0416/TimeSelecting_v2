import global_setting.global_dic as glv
import pandas as pd
import global_tools_func.global_tools as gt
import os
import yaml

class portfolio_updating:
    def __init__(self,target_date):
        self.target_date=target_date
    def timeselecting_signalWithdraw(self):
        available_date2 = gt.intdate_transfer(self.target_date)
        inputpath_mean = glv.get('signal_combine')
        inputpath_mean = os.path.join(inputpath_mean, 'combine_mean')
        gt.folder_creator2(inputpath_mean)
        daily_inputpath_mean = os.path.join(inputpath_mean, 'combine_' + str(available_date2) + '.csv')
        df = gt.readcsv(daily_inputpath_mean)
        combine_value = df['combine_value'].tolist()[0]
        return combine_value
    def portfoliol_info_withdraw(self, index_abbr):
        """
        Get portfolio information for a specific index
        
        Args:
            index_abbr (str): Index abbreviation (e.g., 'hs300', 'zz500', 'zz1000')
            
        Returns:
            pd.DataFrame: DataFrame with columns 'portfolio_name' and 'weight'
        """
        # Get the absolute path to the workspace
        workspace_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(workspace_path, 'config_project', 'portfolio.yaml')
        
        # Read portfolio configuration from yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            portfolio_config = yaml.safe_load(f)
        
        # Check if the index exists
        if index_abbr not in portfolio_config['indices']:
            raise ValueError(f"Index {index_abbr} not found in configuration")
        
        # Get portfolios for the specified index
        portfolios = portfolio_config['indices'][index_abbr]['portfolios']
        
        # Create DataFrame
        df = pd.DataFrame(portfolios)
        df = df[['name', 'weight']]  # Select only name and weight columns
        df.columns = ['portfolio_name', 'weight']  # Rename columns
        
        return df
    def portfolio_withdraw(self):
        inputpath=glv.get('portfolio_output')
        target2=gt.intdate_transfer(self.target_date)
        df_info_hs300=self.portfoliol_info_withdraw('hs300')
        df_info_zz500=self.portfoliol_info_withdraw('zz500')
        df_info_zz1000=self.portfoliol_info_withdraw('zz1000')
        df_final_300=pd.DataFrame()
        df_final_500=pd.DataFrame()
        df_final_1000=pd.DataFrame()
        for i in range(len(df_info_hs300)):
            portfolio_name=df_info_hs300['portfolio_name'][i]
            weight=df_info_hs300['weight'][i]
            inputpath_hs300=os.path.join(inputpath,portfolio_name)
            inputpath_hs300=gt.file_withdraw(inputpath_hs300,target2)
            df_hs300=gt.readcsv(inputpath_hs300)
            
            # Ensure df_hs300 has code and weight columns
            if 'code' not in df_hs300.columns or 'weight' not in df_hs300.columns:
                raise ValueError(f"Portfolio {portfolio_name} must have 'code' and 'weight' columns")
            
            # Multiply the portfolio weight by the individual stock weights
            df_hs300['weight'] = df_hs300['weight'] * weight
            
            # Merge with final DataFrame
            if df_final_300.empty:
                df_final_300 = df_hs300
            else:
                # Group by code and sum the weights
                df_final_300 = pd.concat([df_final_300, df_hs300])
                df_final_300 = df_final_300.groupby('code')['weight'].sum().reset_index()
        
        # Process ZZ500 portfolios
        for i in range(len(df_info_zz500)):
            portfolio_name=df_info_zz500['portfolio_name'][i]
            weight=df_info_zz500['weight'][i]
            inputpath_zz500=os.path.join(inputpath,portfolio_name)
            inputpath_zz500=gt.file_withdraw(inputpath_zz500,target2)
            df_zz500=gt.readcsv(inputpath_zz500)
            
            if 'code' not in df_zz500.columns or 'weight' not in df_zz500.columns:
                raise ValueError(f"Portfolio {portfolio_name} must have 'code' and 'weight' columns")
            
            df_zz500['weight'] = df_zz500['weight'] * weight
            
            if df_final_500.empty:
                df_final_500 = df_zz500
            else:
                df_final_500 = pd.concat([df_final_500, df_zz500])
                df_final_500 = df_final_500.groupby('code')['weight'].sum().reset_index()
        
        # Process ZZ1000 portfolios
        for i in range(len(df_info_zz1000)):
            portfolio_name=df_info_zz1000['portfolio_name'][i]
            weight=df_info_zz1000['weight'][i]
            inputpath_zz1000=os.path.join(inputpath,portfolio_name)
            inputpath_zz1000=gt.file_withdraw(inputpath_zz1000,target2)
            df_zz1000=gt.readcsv(inputpath_zz1000)
            
            if 'code' not in df_zz1000.columns or 'weight' not in df_zz1000.columns:
                raise ValueError(f"Portfolio {portfolio_name} must have 'code' and 'weight' columns")
            
            df_zz1000['weight'] = df_zz1000['weight'] * weight
            
            if df_final_1000.empty:
                df_final_1000 = df_zz1000
            else:
                df_final_1000 = pd.concat([df_final_1000, df_zz1000])
                df_final_1000 = df_final_1000.groupby('code')['weight'].sum().reset_index()
        
        return df_final_300, df_final_500, df_final_1000
    def portfolio_construction(self):
        # Get base portfolios
        df_300, df_500, df_1000 = self.portfolio_withdraw()
        combine_value = self.timeselecting_signalWithdraw()
        
        if combine_value > 0.5:
            # Rename weight columns
            df_300 = df_300.rename(columns={'weight': 'weight_hs300'})
            df_500 = df_500.rename(columns={'weight': 'weight_zz500'})
            df_1000 = df_1000.rename(columns={'weight': 'weight_zz1000'})
            
            # Merge all DataFrames
            df_merged = df_300.merge(df_500, on='code', how='outer').merge(df_1000, on='code', how='outer')
            
            # Fill NaN values with 0
            df_merged = df_merged.fillna(0)
            
            # Calculate new portfolios
            df_timeselecting_zz500 = pd.DataFrame()
            df_timeselecting_zz500['code'] = df_merged['code']
            df_timeselecting_zz500['weight'] = (1 - combine_value) * df_merged['weight_zz500'] + combine_value * df_merged['weight_zz1000']
            
            df_timeselecting_hs300 = pd.DataFrame()
            df_timeselecting_hs300['code'] = df_merged['code']
            df_timeselecting_hs300['weight'] = 0.8 * df_merged['weight_hs300'] + 0.2 * df_merged['weight_zz1000']
            
            df_timeselecting_hs300_pro = pd.DataFrame()
            df_timeselecting_hs300_pro['code'] = df_merged['code']
            df_timeselecting_hs300_pro['weight'] = (1 - combine_value) * df_merged['weight_hs300'] + combine_value * df_merged['weight_zz1000']
            
            # Remove rows with zero weights
            df_timeselecting_zz500 = df_timeselecting_zz500[df_timeselecting_zz500['weight'] != 0]
            df_timeselecting_hs300 = df_timeselecting_hs300[df_timeselecting_hs300['weight'] != 0]
            df_timeselecting_hs300_pro = df_timeselecting_hs300_pro[df_timeselecting_hs300_pro['weight'] != 0]
            
            return df_timeselecting_hs300, df_timeselecting_hs300_pro, df_timeselecting_zz500
        elif combine_value == 0.5:
            # When combine_value = 0.5, return original portfolios with consistent column names
            df_timeselecting_zz500 = df_500.copy()
            df_timeselecting_hs300 = df_300.copy()
            df_timeselecting_hs300_pro = df_300.copy()
            
            # Ensure column names are consistent
            df_timeselecting_zz500 = df_timeselecting_zz500[['code', 'weight']]
            df_timeselecting_hs300 = df_timeselecting_hs300[['code', 'weight']]
            df_timeselecting_hs300_pro = df_timeselecting_hs300_pro[['code', 'weight']]
            
            return df_timeselecting_hs300, df_timeselecting_hs300_pro, df_timeselecting_zz500
        else:
            # When combine_value < 0.5
            # Rename weight columns
            df_300 = df_300.rename(columns={'weight': 'weight_hs300'})
            df_500 = df_500.rename(columns={'weight': 'weight_zz500'})
            
            # Merge DataFrames
            df_merged = df_300.merge(df_500, on='code', how='outer')
            
            # Fill NaN values with 0
            df_merged = df_merged.fillna(0)
            
            # Calculate new portfolios
            df_timeselecting_zz500 = pd.DataFrame()
            df_timeselecting_zz500['code'] = df_merged['code']
            df_timeselecting_zz500['weight'] = (1 - combine_value) * df_merged['weight_hs300'] + combine_value * df_merged['weight_zz500']
            
            # df_timeselecting_hs300 and df_timeselecting_hs300_pro are just df_300
            df_timeselecting_hs300 = df_300.copy()
            df_timeselecting_hs300_pro = df_300.copy()
            
            # Ensure column names are consistent
            df_timeselecting_hs300.columns=['code','weight']
            df_timeselecting_hs300_pro.columns=['code','weight']
            
            # Remove rows with zero weights
            df_timeselecting_zz500 = df_timeselecting_zz500[df_timeselecting_zz500['weight'] != 0]
            
            return df_timeselecting_hs300, df_timeselecting_hs300_pro, df_timeselecting_zz500
    def portfolio_saving_main(self):
        target_date=gt.intdate_transfer(self.target_date)
        outputpath=glv.get('portfolio_output')
        df_timeselecting_hs300, df_timeselecting_hs300_pro, df_timeselecting_zz500 = self.portfolio_construction()
        outputpath_hs300=os.path.join(outputpath,'timeselecting_hs300')
        outputpath_hs300_pro=os.path.join(outputpath,'timeselecting_hs300_pro')
        outputpath_zz500=os.path.join(outputpath,'timeselecting_zz500')
        gt.folder_creator2(outputpath_hs300)
        gt.folder_creator2(outputpath_hs300_pro)
        gt.folder_creator2(outputpath_zz500)
        outputpath_hs300=os.path.join(outputpath_hs300,'timeselecting_hs300_'+str(target_date)+'.csv')
        outputpath_hs300_pro=os.path.join(outputpath_hs300_pro,'timeselecting_hs300_pro_'+str(target_date)+'.csv')
        outputpath_zz500=os.path.join(outputpath_zz500,'timeselecting_zz500_'+str(target_date)+'.csv')
        df_timeselecting_hs300.to_csv(outputpath_hs300,index=False)
        df_timeselecting_hs300_pro.to_csv(outputpath_hs300_pro,index=False)
        df_timeselecting_zz500.to_csv(outputpath_zz500,index=False)
if __name__ == "__main__":
    pu=portfolio_updating('2025-04-30')
    pu.portfolio_saving_main()