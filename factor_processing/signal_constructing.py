import pandas as pd
import global_tools_func.global_tools as gt
from numpy import *
from pykalman import KalmanFilter
import numpy as np
class signal_construct:
    def direction_decision(self,x):
        if x > 0:
            return 0  # 沪深300
        else:
            return 1  # 中证2000
    def direction_decision2(self,x):
        if x < 0:
            return 0  # 沪深300
        else:
            return 1  # 中证2000
    def Dp_MA_signal_construct(self,df,rolling_window,mode_type):
        signal_name=df.columns.tolist()[1]
        MA=mean(df[signal_name].tolist()[-rolling_window:])
        signal=df[signal_name+'_KL'].tolist()[-1]
        difference = signal - MA
        if mode_type=='mode_1':
             final_signal = self.direction_decision(difference)
        elif mode_type=='mode_2':
            final_signal = self.direction_decision2(difference)
        else:
             final_signal=None
             print('没有mode_type')
             raise ValueError
        return final_signal
    def MA_signal_construct(self,df,rolling_window,mode_type):
        signal_name = df.columns.tolist()[1]
        MA = mean(df[signal_name].tolist()[-rolling_window:])
        signal = df[signal_name].tolist()[-1]
        difference = signal - MA
        if mode_type=='mode_3':
              final_signal = self.direction_decision(difference)
        elif mode_type=='mode_4':
             final_signal = self.direction_decision2(difference)
        else:
             final_signal=None
             print('没有mode_type')
             raise ValueError
        return final_signal
    def MA_difference_signal_construct(self,df,rolling_window_list,mode_type):
        signal_name=df.columns.tolist()[1]
        rolling_window_short=min(rolling_window_list)
        rolling_window_long=max(rolling_window_list)       
        MA_short=mean(df[signal_name].tolist()[-rolling_window_short:])
        MA_long=mean(df[signal_name].tolist()[-rolling_window_long :])
        difference = MA_short-MA_long
        if mode_type=='mode_6':
             final_signal = self.direction_decision(difference)
        elif mode_type=='mode_7':
            final_signal = self.direction_decision2(difference)
        else:
             final_signal=None
             print('没有mode_type')
             raise ValueError
        return final_signal
    def M1M2_signal_construct(self,df,rolling_window):
        MA_1 = mean(df['M2'].tolist()[-rolling_window:])
        MA_2 = mean(df['difference'].tolist()[-rolling_window:])
        M2 = df['M2'].tolist()[-1]
        M1_M2 = df['difference'].tolist()[-1]
        difference_1 = M2 - MA_1
        difference_2 = M1_M2 - MA_2
        if difference_1 < 0 and difference_2 < 0:
            final_signal = 0
        elif difference_1 > 0 and difference_2 > 0:
            final_signal = 0
        else:
            final_signal = 1
        return final_signal
    def LargeOrder_difference_signal_construct(self,df):
         a=df['LargeOrder_difference'].tolist()[-1]
         if a>0.01:
             final_signal=0
         else:
             final_signal=1
         return final_signal
    def Monthly_effect_signal_construct(self,df):
         a=df['monthly_effect'].tolist()[-1]
         if a>0:
             final_signal=0
         else:
             final_signal=1
         return final_signal
    def single_direction_decision(self,df):
        a=df[df.columns.tolist()[1]].tolist()[-1]
        if a > 0:
            final_signal = 0
        else:
            final_signal = 1
        return final_signal
    def single_direction_decision2(self,df):

        a=df[df.columns.tolist()[1]].tolist()[-1]
        if a > 0:
            final_signal = 1
        else:
            final_signal = 0
        return final_signal
    def Monthly_factor_signal_construct(self,df,mode_type):
        signal_list = df[df.columns.tolist()[1]].unique().tolist()
        if len(signal_list)==1:
            a=signal_list[0]
            b=signal_list[0]
        else:
            a = signal_list[-1]
            b = signal_list[-2]
        if a > b:
            if mode_type=='mode_10':
                   final_signal = 1
            else:
                   final_signal=0
        elif a==b:
            final_signal=0.5
        else:
            if mode_type == 'mode_10':
                final_signal = 0
            else:
                final_signal = 1
        return final_signal
    def stock_HL_signal_construct(self,df,signal_name):
        signal_name2 = df.columns.tolist()[1]
        signal_values = df[signal_name2].tolist()
        # last_value = mean(signal_values[-15:])
        last_value = signal_values[-1]
        # 初始化final_signal
        if len(signal_values) >= 252:
            signal_values2 = [abs(i) for i in signal_values]
            if signal_name=='Stock_HL':
                   percentile = np.percentile(signal_values2, 20)
            else:
                   percentile= np.percentile(signal_values2, 20)
            if abs(last_value) >= percentile:
                final_signal = 0 if last_value <= 0 else 1
            else:
                final_signal = 0.5
        else:
            final_signal = 0 if last_value <= 0 else 1
        return final_signal
    def technical_signal_construct(self,df,signal_name):
        # 初始化final_signal为默认值
        final_signal = 0.5
        
        if signal_name=='TargetIndex_MACD':
            macd = df['MACD'].tolist()[-1]
            macd_s = df['MACD_s'].tolist()[-1]
            if macd > macd_s:
                final_signal = 0
            else:
                final_signal = 1
        elif signal_name=='TargetIndex_RSI':
            # 获取RSI值列表
            rsi_values = df['RSI'].tolist()
            
            # 检查是否有足够的数据
            if len(rsi_values) < 20:  # 假设N=20，可以根据需要调整
                return final_signal
            
            # 获取过去N天的数据（包含今天）
            N = 5  # 可以根据需要调整
            past_N_days = rsi_values[-N:]  # 包括今天
            
            # RSI条件
            if all(40 <= rsi <= 100 for rsi in past_N_days) and max(past_N_days) > 70:
                final_signal = 1
            else:
                final_signal = 0
            
            # 新增difference条件
            difference = df['difference'].tolist()[-1]
            past_differences = df['difference'].tolist()[-10:]  # 过去10天的difference值
            
            # Case 1: difference > 0.12
            if difference > 0.12:
                final_signal = 1
            # Case 2: difference < -0.12
            elif difference < -0.12:
                final_signal = 0
            else:
                # 检查过去10天是否有Case 1发生
                case1_occurred = any(d > 0.12 for d in past_differences)
                if case1_occurred:
                    # 找到最近一次Case 1发生的位置
                    case1_index = next(i for i, d in enumerate(past_differences) if d > 0.12)
                    # 检查从Case 1发生到现在是否有difference < -0.03
                    if any(d < -0.03 for d in past_differences[case1_index:]):
                        final_signal = 0.5
                    else:
                        final_signal = 1
                
                # 检查过去10天是否有Case 2发生
                case2_occurred = any(d < -0.12 for d in past_differences)
                if case2_occurred:
                    # 找到最近一次Case 2发生的位置
                    case2_index = next(i for i, d in enumerate(past_differences) if d < -0.12)
                    # 检查从Case 2发生到现在是否有difference > 0.03
                    if any(d > 0.03 for d in past_differences[case2_index:]):
                        final_signal = 0.5
                    else:
                        final_signal = 0
            
            return final_signal
        elif signal_name=='TargetIndex_BBANDS':
            upper = df['upper'].tolist()[-1]
            lower = df['lower'].tolist()[-1]
            target_index=df['target_index'].tolist()[-1]
            if target_index>=upper:
                final_signal=0
            elif target_index<=lower:
                final_signal=1
            else:
                final_signal=0.5
        elif signal_name=='TargetIndex_KDJ':
            final_signal=0.5
            # 获取KDJ数据
            K = df['K_9_3'].tolist()
            D = df['D_9_3'].tolist()
            J = df['J_9_3'].tolist()
            # 检查是否有足够的数据
            if len(K) < 10:
                return final_signal
            # 查找过去10天中case1（K在20左右向上交叉D）的触发时间
            case1_trigger_index = -1
            for i in range(1, min(11, len(K))):
                if (K[-i-1] <= D[-i-1] and K[-i] > D[-i] and 
                    K[-i-1]<= 25 and D[-i-1]<= 25):
                    case1_trigger_index = -i
                    break
            # 查找过去10天中case2（K在80左右向下交叉D）的触发时间
            case2_trigger_index = -1
            for i in range(1, min(11, len(K))):
                if (K[-i-1] >= D[-i-1] and K[-i] < D[-i] and 
                    K[-i-1] >= 75 and D[-i-1] >= 75):
                    case2_trigger_index = -i
                    break
            # 检查从case1触发到现在是否有case2发生
            if case1_trigger_index != -1:
                case2_occurred = False
                for i in range(case1_trigger_index, 0):
                    if K[i]<D[i] or (K[i]>80 and D[i]>80):
                        case2_occurred = True
                        break
                if not case2_occurred:
                    final_signal = 1
                    print(df.iloc[case2_trigger_index:],final_signal)
                    return final_signal
            # 检查从case2触发到现在是否有case1发生
            if case2_trigger_index != -1:
                case1_occurred = False
                for i in range(case2_trigger_index, 0):
                    if  K[i]>D[i] or (K[i]<20 and D[i]<20):
                        case1_occurred = True
                        break
                if not case1_occurred:
                    final_signal = 0
                    print(df.iloc[case2_trigger_index:],final_signal)
                    return final_signal
        elif signal_name=='TargetIndex_PSA':
            PSAL=df['PSARl_0.02_0.2'].tolist()[-1]
            PSAS=df['PSARs_0.02_0.2'].tolist()[-1]
            
            # 检查两个值是否都是NaN
            if np.isnan(PSAL) and np.isnan(PSAS):
                final_signal = 0.5
            # 如果PSAL不是NaN，返回1
            elif not np.isnan(PSAL) and np.isnan(PSAS):
                final_signal = 0
            # 如果PSAS不是NaN，返回0
            elif not np.isnan(PSAS) and np.isnan(PSAL):
                final_signal = 1
            # 其他情况返回0.5
            else:
                final_signal = 0.5
        elif signal_name=='TargetIndex_MOMENTUM2':
            difference=df['difference'].tolist()[-1]
            if difference>0:
                final_signal=0
            elif difference<0:
                final_signal=1
            else:
                final_signal=0.5
        elif signal_name=='TargetIndex_REVERSE':
            past_differences = df['difference'].tolist()[-10:]  # 过去10天的difference值
            df_vix_300=df[['valuation_date','hs300']]
            df_vix_1000=df[['valuation_date','zz1000']]
            df_vix_300.dropna(inplace=True)
            df_vix_1000.dropna(inplace=True)
            if len(df_vix_300)==0 and len(df_vix_1000)==0:
                vix=True
            else:
                if len(df_vix_1000)<252:
                    df_vix=df_vix_300
                else:
                    df_vix=df_vix_1000
                if len(df_vix)<252:
                    vix=True
                else:
                    df_vix['quantile_09'] = df_vix[df_vix.columns.tolist()[1]].rolling(252).quantile(0.8)
                    vix_last = df_vix[df_vix.columns.tolist()[1]].tolist()[-1]
                    vix_quantile = df_vix['quantile_09'].tolist()[-1]
                    if vix_last >= vix_quantile:
                        vix = True
                    else:
                        vix = False
            final_signal = 0.5  # 默认值
            case1_trigger_index = -1
            for i in range(1, min(11, len(past_differences))):
                if past_differences[-i] > 0.075 and vix==True:
                    case1_trigger_index = -i
                    break
            # 查找过去10天中case2（K在80左右向下交叉D）的触发时间
            case2_trigger_index = -1
            for i in range(1, min(11, len(past_differences))):
                if past_differences[-i]<-0.3:
                    case2_trigger_index = -i
                    break
            if case1_trigger_index!=-1:
                case1_occurred = False
                for i in range(case1_trigger_index, 0):
                    if past_differences[i]<-0.05:
                        case1_occurred = True
                        break
                if not case1_occurred:
                    final_signal = 1
            if case2_trigger_index!=-1:
                case2_occurred = False
                for i in range(case2_trigger_index, 0):
                    if past_differences[i]>0.02:
                        case2_occurred = True
                        break
                if not case2_occurred:
                    final_signal = 0
        return final_signal


class factor_processing:
    #tool_box
    def Kalman1D(self,x, damping=1):
        # To return the smoothed time series data
        observation_covariance = damping
        initial_value_guess = x[0]
        transition_matrix = 1
        transition_covariance = 0.1
        kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
        x_hat, state_cov = kf.smooth(x)
        x_hat=x_hat.flatten()
        return x_hat.tolist()
    #data_processing
    def Kal_processing(self,df,target_date,damping):
        signal_name=df.columns.tolist()[1]
        df = df[df['valuation_date'] < target_date]
        df.reset_index(inplace=True, drop=True)
        df = df.iloc[-250:]
        df.reset_index(inplace=True, drop=True)
        df[signal_name+'_KL'] = self.Kalman1D(df[signal_name].tolist(), damping)
        return df
    def slice_processing(self,df,target_date):
        df = df[df['valuation_date'] < target_date]
        df.reset_index(inplace=True, drop=True)
        return df
    def slice_processing2(self,df,target_date):
        df = df[df['valuation_date'] <= target_date]
        df.reset_index(inplace=True, drop=True)
        return df
    def slice_processing_Monthly(self,df,target_date):
        df_final=df.copy()
        if target_date<='2025-03-01':
            df_final.set_index('valuation_date', inplace=True, drop=True)
            df_final = df_final.shift(20)
            df_final.dropna(inplace=True)
            df_final.reset_index(inplace=True)
        df_final=df_final[df_final['valuation_date'] < target_date]
        return df_final