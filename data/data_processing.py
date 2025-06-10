from matplotlib import pyplot as plt
import os
import sys
import pandas as pd
path = os.getenv('GLOBAL_TOOLSFUNC')
sys.path.append(path)
import global_tools as gt
import global_setting.global_dic as glv
from data.data_prepare import data_prepare
import pandas_ta as ta
class data_processing:
    def __init__(self):
        self.dp=data_prepare()
    def credit_spread_3M(self):
         df_zzgk=self.dp.raw_ZZGK(period='3M')
         df_zzzd = self.dp.raw_ZZZD(period='3M')
         df_zzzd=df_zzzd.merge(df_zzgk,on='valuation_date',how='outer')
         df_zzzd.dropna(inplace=True)
         df_zzzd['credit_spread_3M']=abs(df_zzzd['CMTN_3M']-df_zzzd['CDBB_3M'])
         df_zzzd=df_zzzd[['valuation_date','credit_spread_3M']]
         return df_zzzd
    def credit_spread_9M(self):
         df_zzgk=self.dp.raw_ZZGK(period='9M')
         df_zzzd = self.dp.raw_ZZZD(period='9M')
         df_zzzd=df_zzzd.merge(df_zzgk,on='valuation_date',how='outer')
         df_zzzd.dropna(inplace=True)
         df_zzzd['credit_spread_9M']=abs(df_zzzd['CMTN_9M']-df_zzzd['CDBB_9M'])
         df_zzzd=df_zzzd[['valuation_date','credit_spread_9M']]
         return df_zzzd
    def credit_spread_5Y(self):
         df_zzgk=self.dp.raw_ZZGK(period='5Y')
         df_zzzd = self.dp.raw_ZZZD(period='5Y')
         df_zzzd=df_zzzd.merge(df_zzgk,on='valuation_date',how='outer')
         df_zzzd.dropna(inplace=True)
         df_zzzd['credit_spread_5Y']=abs(df_zzzd['CMTN_5Y']-df_zzzd['CDBB_5Y'])
         df_zzzd=df_zzzd[['valuation_date','credit_spread_5Y']]
         return df_zzzd
    def term_spread_9Y(self):
         df_10Y=self.dp.raw_ZZGK(period='10Y')
         df_1Y= self.dp.raw_ZZGK(period='1Y')
         df_zzzd=df_1Y.merge(df_10Y,on='valuation_date',how='outer')
         df_zzzd.dropna(inplace=True)
         df_zzzd=df_zzzd[df_zzzd['valuation_date']>'2017-08-01']
         df_zzzd['term_spread_9Y']=abs(df_zzzd['CDBB_10Y']-df_zzzd['CDBB_1Y'])
         df_zzzd=df_zzzd[['valuation_date','term_spread_9Y']]
         return df_zzzd
    def M1M2(self):
         df_M1=self.dp.raw_M1M2(signal_name='M1')
         df_M2=self.dp.raw_M1M2(signal_name='M2')
         df_M1M2=df_M1.merge(df_M2,on='valuation_date',how='outer')
         df_M1M2.dropna(inplace=True)
         working_days_list=gt.working_days_list(df_M1M2['valuation_date'].tolist()[0],df_M1M2['valuation_date'].tolist()[-1])
         df_final=pd.DataFrame()
         df_final['valuation_date']=working_days_list
         df_final=df_final.merge(df_M1M2,on='valuation_date',how='outer')
         df_final.fillna(method='ffill',inplace=True)
         df_final=df_final[df_final['valuation_date'].isin(working_days_list)]
         df_final['difference']=df_final['M1']-df_final['M2']
         return df_final
    def US_stock(self):
         df_D=self.dp.raw_DJUS()
         df_N=self.dp.raw_NDAQ()
         df_us=df_D.merge(df_N,on='valuation_date',how='outer')
         df_us.dropna(inplace=True)
         df_us.set_index('valuation_date', inplace=True, drop=True)
         df_us = (1 + df_us).cumprod()
         df_us['D/N'] = df_us['DJUS'] / df_us['NDAQ']
         df_us.reset_index(inplace=True)
         df_us['valuation_date']=pd.to_datetime(df_us['valuation_date'])
         df_us['valuation_date']=df_us['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
         return df_us
    def relativeVolume_std(self):
         df_300=self.dp.raw_index_amt(index_name='沪深300')
         df_2000=self.dp.raw_index_amt(index_name='中证2000')
         df_300['std_300']=df_300['沪深300'].rolling(40).std()
         df_2000['std_2000']=df_2000['中证2000'].rolling(40).std()
         df_final=df_300.merge(df_2000,on='valuation_date',how='left')
         df_final.dropna(inplace=True)
         df_final['RelativeVolume_std']=df_final['std_300']/df_final['std_2000']
         df_final=df_final[['valuation_date','RelativeVolume_std']]
         return df_final
    def relativeReturn_std(self):
         df=self.dp.index_return_withdraw()
         df['std_300']=df['沪深300'].rolling(40).std()
         df['std_2000']=df['中证2000'].rolling(40).std()
         df.dropna(inplace=True)
         df['RelativeReturn_std']=df['std_300']/df['std_2000']
         df=df[['valuation_date','RelativeReturn_std']]
         return df
    def LargeOrder_difference(self):
        df=self.dp.raw_LargeOrder_withdraw()
        df_hs300amt=self.dp.raw_index_amt(index_name='沪深300')
        df_hs300amt.columns=['valuation_date','hs300amt']
        df_zz1000amt=self.dp.raw_index_amt(index_name='中证1000')
        df_zz1000amt.columns=['valuation_date','zz1000amt']
        df_zz2000amt=self.dp.raw_index_amt(index_name='国证2000')
        df_zz2000amt.columns=['valuation_date','zz2000amt']
        df=df[['valuation_date','000300.SH','000852.SH','399303.SZ']]
        df=df.merge(df_hs300amt,on='valuation_date',how='left')
        df=df.merge(df_zz1000amt,on='valuation_date',how='left')
        df=df.merge(df_zz2000amt,on='valuation_date',how='left')
        df['000300.SH']=df['000300.SH'].astype(float)/df['hs300amt']
        df['000852.SH']=df['000852.SH'].astype(float)/df['zz1000amt']
        df['399303.SZ']=df['399303.SZ'].astype(float)/df['zz2000amt']
        
        # 将数据转换为月频
        df['valuation_date'] = pd.to_datetime(df['valuation_date'])
        df.set_index('valuation_date', inplace=True)
        
        # 计算每个月内部数值的和
        monthly_df = df.resample('M').sum()
        
        # 对每个指数列计算月度差值
        for col in ['000300.SH', '000852.SH', '399303.SZ']:
            # 计算当前月份的和
            current_month_sum = monthly_df[col]
            # 计算上个月的和
            last_month_sum = monthly_df[col].shift(1)
            # 计算差值
            change_col = col.replace('.SH', '_change').replace('.SZ', '_change')
            monthly_df[change_col] = current_month_sum - last_month_sum
        
        # 重置索引，将日期转回字符串格式
        monthly_df.reset_index(inplace=True)
        monthly_df['valuation_date'] = monthly_df['valuation_date'].dt.strftime('%Y-%m-%d')
        
        # 选择需要的列
        monthly_df = monthly_df[['valuation_date','000300_change','000852_change','399303_change']]
        # 将原始数据也转换为月频
        df.reset_index(inplace=True)
        df['valuation_date'] = pd.to_datetime(df['valuation_date'])
        df.set_index('valuation_date', inplace=True)
        df_monthly = df.resample('M').sum()
        df_monthly.reset_index(inplace=True)
        df_monthly['valuation_date'] = df_monthly['valuation_date'].dt.strftime('%Y-%m-%d')
        
        # 合并数据
        df_monthly = df_monthly.merge(monthly_df, on='valuation_date', how='left')
        df_monthly.fillna(method='ffill', inplace=True)
        df_monthly.dropna(inplace=True)
        df_monthly['hs300']=0.6*df_monthly['000300.SH']+0.4*df_monthly['000300_change']
        df_monthly['zz1000']=0.6*df_monthly['000852.SH']+0.4*df_monthly['000852_change']
        df_monthly['zz2000']=0.6*df_monthly['399303.SZ']+0.4*df_monthly['399303_change']
        df_monthly['LargeOrder_difference']=df_monthly['hs300']-df_monthly['zz2000']
        df_monthly=df_monthly[['valuation_date','LargeOrder_difference']]
        return df_monthly
    def monthly_effect(self):
        """
        计算历史同月份的沪深300与国证2000收益率差值的平均数
        不包括当年的数据
        
        Returns:
            DataFrame: 包含日期和历史同月份平均收益率差值
        """
        # 获取原始数据
        df_return = self.dp.index_return_withdraw2()
        
        # 转换日期格式
        df_return['valuation_date'] = pd.to_datetime(df_return['valuation_date'])
        
        # 添加月份和年份列
        df_return['month'] = df_return['valuation_date'].dt.month
        df_return['year'] = df_return['valuation_date'].dt.year
        
        # 计算沪深300和国证2000的收益率差值
        df_return['return_diff'] = df_return['沪深300'] - df_return['国证2000']
        
        # 创建结果DataFrame
        result_df = pd.DataFrame()
        result_df['valuation_date'] = df_return['valuation_date'].unique()
        result_df['valuation_date'] = pd.to_datetime(result_df['valuation_date'])
        
        # 对每个日期计算历史同月份的平均收益率差值
        def calc_historical_avg(date):
            current_month = date.month
            current_year = date.year
            historical_data = df_return[
                (df_return['valuation_date'] < date) & 
                (df_return['month'] == current_month) &
                (df_return['year'] < current_year)  # 排除当年的数据
            ]
            return historical_data['return_diff'].mean() if not historical_data.empty else None
        
        # 应用计算函数
        result_df['monthly_effect'] = result_df['valuation_date'].apply(calc_historical_avg)
        
        # 转换日期格式为字符串
        result_df['valuation_date'] = result_df['valuation_date'].dt.strftime('%Y-%m-%d')
        result_df.dropna(inplace=True)
        return result_df[['valuation_date', 'monthly_effect']]
    def stock_highLow(self):
        """
        计算每只股票过去三个月的最高价和最低价信号
        如果当天股价大于过去三个月最高价返回1
        如果当天股价小于过去三个月最低价返回-1
        其他情况返回0
        最后计算每天信号为1的股票数量减去信号为-1的股票数量的差值
        
        Returns:
            DataFrame: 包含日期和信号差值
        """
        # 获取股票收盘价数据
        df_stock = self.dp.raw_stockClose_withdraw()
        
        # 转换日期列为datetime类型，并删除无效日期
        df_stock['valuation_date'] = pd.to_datetime(df_stock['valuation_date'])
        df_stock = df_stock.dropna(subset=['valuation_date'])
        
        # 设置日期为索引以便进行时间序列操作
        df_stock.set_index('valuation_date', inplace=True)
        
        # 计算过去三个月的最高价和最低价（不包含当天）
        high_3m = df_stock.rolling(window='90D', min_periods=1).max().shift(1)
        low_3m = df_stock.rolling(window='90D', min_periods=1).min().shift(1)
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=df_stock.index)
        
        # 对每只股票计算信号
        for col in df_stock.columns:
            # 获取当前价格
            current_price = df_stock[col]
            # 获取历史最高价和最低价（不包含当天）
            hist_high = high_3m[col]
            hist_low = low_3m[col]
            
            # 计算信号
            signal = pd.Series(0, index=current_price.index)
            # 只在有足够历史数据时计算信号
            valid_mask = current_price.notna() & hist_high.notna() & hist_low.notna()
            
            # 计算信号
            signal[valid_mask & (current_price > hist_high)] = 1
            signal[valid_mask & (current_price < hist_low)] = -1
            
            # 将信号添加到结果DataFrame
            result[col] = signal

        
        # 计算每天的有效股票数量（收盘价不为空的股票）
        valid_stocks = df_stock.notna().sum(axis=1)
        
        # 计算每天信号为1和-1的股票数量
        signal_1_count = (result == 1).sum(axis=1)
        signal_minus_1_count = (result == -1).sum(axis=1)
        
        # 计算信号差值
        signal_diff = signal_1_count - signal_minus_1_count
        
        # 创建最终结果DataFrame
        final_result = pd.DataFrame({
            'valuation_date': signal_diff.index,
            'stock_highLow': signal_diff,
            'valid_stocks': valid_stocks
        })
        
        # 去掉前90天的数据
        final_result = final_result.iloc[90:]
        
        # 将日期转换回字符串格式，确保没有NaT值
        final_result = final_result.dropna(subset=['valuation_date'])
        final_result['valuation_date'] = final_result['valuation_date'].dt.strftime('%Y-%m-%d')
        final_result.reset_index(inplace=True,drop=True)
        
        return final_result[['valuation_date', 'stock_highLow']]
    def stock_rsi(self):
        """
        计算每只股票的RSI指标
        1. 计算每只股票的日收益率
        2. 使用14天窗口计算RSI
        3. RSI = A/(A+abs(B))*100，其中A为14天内正收益之和，B为14天内负收益之和
        4. 计算每日RSI>70的股票数量与RSI<30的股票数量之差
        
        Returns:
            DataFrame: 包含日期和RSI差值（RSI>70的股票数量 - RSI<30的股票数量）
        """
        # 获取股票收盘价数据
        df_stock = self.dp.raw_stockClose_withdraw()
        
        # 转换日期列为datetime类型，并删除无效日期
        df_stock['valuation_date'] = pd.to_datetime(df_stock['valuation_date'])
        df_stock = df_stock.dropna(subset=['valuation_date'])
        
        # 设置日期为索引以便进行时间序列操作
        df_stock.set_index('valuation_date', inplace=True)
        
        # 计算每只股票的日收益率
        df_returns = df_stock.pct_change()
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=df_stock.index)
        
        # 对每只股票计算RSI
        for col in df_returns.columns:
            # 获取收益率序列
            returns = df_returns[col]
            
            # 分离正收益和负收益
            positive_returns = returns.copy()
            negative_returns = returns.copy()
            positive_returns[positive_returns < 0] = 0
            negative_returns[negative_returns > 0] = 0
            
            # 计算14天窗口内的正收益和负收益之和
            positive_sum = positive_returns.rolling(window=14, min_periods=1).sum()
            negative_sum = negative_returns.rolling(window=14, min_periods=1).sum()
            
            # 计算RSI
            rsi = positive_sum / (positive_sum + abs(negative_sum)) * 100
            
            # 将RSI添加到结果DataFrame
            result[col] = rsi
        
        # 计算每天RSI大于70和小于30的股票数量
        rsi_high_count = (result > 70).sum(axis=1)
        rsi_low_count = (result < 30).sum(axis=1)
        
        # 计算信号差值（超买股票数量减去超卖股票数量）
        signal_diff = rsi_high_count - rsi_low_count
        
        # 创建最终结果DataFrame
        final_result = pd.DataFrame({
            'valuation_date': signal_diff.index,
            'rsi_difference': signal_diff
        })
        
        # 去掉前14天的数据（因为RSI需要14天数据）
        final_result = final_result.iloc[14:]
        
        # 将日期转换回字符串格式，确保没有NaT值
        final_result = final_result.dropna(subset=['valuation_date'])
        final_result['valuation_date'] = final_result['valuation_date'].dt.strftime('%Y-%m-%d')
        final_result.reset_index(inplace=True, drop=True)
        return final_result
    def stock_trend(self):
        """
        计算每日收盘价格在月均线上方的股票比例
        1. 计算每只股票的20日移动平均线
        2. 计算收盘价在均线上方的股票数量
        3. 计算该数量与有效股票总数的比例
        
        Returns:
            DataFrame: 包含日期和上涨趋势股票比例
        """
        # 获取股票收盘价数据
        df_stock = self.dp.raw_stockClose_withdraw()
        
        # 转换日期列为datetime类型，并删除无效日期
        df_stock['valuation_date'] = pd.to_datetime(df_stock['valuation_date'])
        df_stock = df_stock.dropna(subset=['valuation_date'])
        
        # 设置日期为索引以便进行时间序列操作
        df_stock.set_index('valuation_date', inplace=True)
        
        # 计算20日移动平均线
        ma20 = df_stock.rolling(window=20, min_periods=1).mean()
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=df_stock.index)
        
        # 对每只股票判断是否在均线上方
        for col in df_stock.columns:
            # 获取收盘价和均线
            close = df_stock[col]
            ma = ma20[col]
            
            # 判断是否在均线上方
            above_ma = (close > ma).astype(int)
            
            # 将结果添加到结果DataFrame
            result[col] = above_ma
        
        # 计算每日有效股票数量（收盘价不为0的股票）
        valid_stocks = (df_stock != 0).sum(axis=1)
        
        # 计算每日在均线上方的股票数量
        above_ma_count = result.sum(axis=1)
        
        # 计算比例
        raising_trend_proportion = above_ma_count / valid_stocks * 100
        
        # 创建最终结果DataFrame
        final_result = pd.DataFrame({
            'valuation_date': raising_trend_proportion.index,
            'RaisingTrend_proportion': raising_trend_proportion
        })
        
        # 去掉前20天的数据（因为MA需要20天数据）
        final_result = final_result.iloc[20:]
        
        # 将日期转换回字符串格式，确保没有NaT值
        final_result = final_result.dropna(subset=['valuation_date'])
        final_result['valuation_date'] = final_result['valuation_date'].dt.strftime('%Y-%m-%d')
        final_result.reset_index(inplace=True, drop=True)
        
        return final_result
    def targetIndex_MACD(self):
        df=self.dp.target_index()
        df.dropna(inplace=True)
        df=df[['valuation_date','target_index']]
        df['MACD'] = ta.macd(df['target_index'])['MACD_12_26_9']
        df['MACD_h'] = ta.macd(df['target_index'])['MACDh_12_26_9']
        df['MACD_s'] = ta.macd(df['target_index'])['MACDs_12_26_9']
        df.dropna(inplace=True)
        df.reset_index(inplace=True,drop=True)
        return df
    def targetIndex_RSI(self):
        df=self.dp.target_index()
        df.dropna(inplace=True)
        df=df[['valuation_date','target_index']]
        df['RSI'] = ta.rsi(df['target_index'],14)
        df.dropna(inplace=True)
        df.reset_index(inplace=True,drop=True)
        df = df[['valuation_date', 'RSI']]
        return df
    def targetIndex_BOLLBAND(self):
        df = self.dp.target_index()
        df.dropna(inplace=True)
        df = df[['valuation_date', 'target_index']]
        # 计算布林带指标
        # 使用正确的方式获取布林带指标
        bbands = ta.bbands(df['target_index'], length=20,std=1.5)
        if bbands is not None:
            df['upper'] = bbands['BBU_20_1.5']
            df['middle'] = bbands['BBM_20_1.5']
            df['lower'] = bbands['BBL_20_1.5']
        else:
            # 如果bbands返回None，设置默认值
            df['upper'] = df['target_index']
            df['middle'] = df['target_index']
            df['lower'] = df['target_index']
            print("Warning: Bollinger Bands calculation returned None. Using default values.")
        df=df[['valuation_date','target_index','upper','middle','lower']]
        df.dropna(inplace=True)
        df.reset_index(inplace=True,drop=True)
        return df
    def TargetIndex_MOMENTUM(self):
        df = self.dp.target_index()
        df.dropna(inplace=True)
        df = df[['valuation_date', 'target_index']]
        df.columns=['valuation_date','TargetIndex_MOMENTUM']
        return df
    def TargetIndex_KDJ(self):
        df = self.dp.target_index_candle()
        df.dropna(inplace=True)
        df_kdj = ta.kdj(df['high'], df['low'], df['close'])
        # 将 KDJ 指标合并到原 DataFrame 中
        df = pd.concat([df, df_kdj], axis=1)
        df.dropna(inplace=True)
        df=df[['valuation_date','K_9_3', 'D_9_3', 'J_9_3']]
        return df
    def TargetIndex_PSA(self):
        df = self.dp.target_index_candle()
        df.dropna(inplace=True)
        # 计算抛物线指标
        psar = ta.psar(df['high'], df['low'])
        # 将抛物线指标合并到原 DataFrame 中
        df = pd.concat([df, psar], axis=1)
        df=df[['valuation_date','close','PSARl_0.02_0.2','PSARs_0.02_0.2']]
        return df
    def TargetIndex_MOMENTUM2(self):
        """
        计算过去20天的涨跌幅累乘
        
        Returns:
            DataFrame: 包含日期和过去20天涨跌幅累乘的结果
        """
        df = self.dp.index_return_withdraw2()
        df.set_index('valuation_date', inplace=True)
        
        # 对每一列分别计算20天涨跌幅累乘
        for col in df.columns:
            df[col] = df[col].rolling(20).apply(lambda x: (1 + x).prod() - 1)
        
        # 重置索引
        df.reset_index(inplace=True)
        df['difference']=df['沪深300']-df['国证2000']
        df=df[['valuation_date','difference']]
        df.dropna(inplace=True)
        return df

    def TargetIndex_MOMENTUM3(self):
        """
        计算过去20天的涨跌幅累乘

        Returns:
            DataFrame: 包含日期和过去20天涨跌幅累乘的结果
        """
        df = self.dp.index_return_withdraw2()
        df.set_index('valuation_date', inplace=True)

        # 对每一列分别计算20天涨跌幅累乘
        for col in df.columns:
            df[col] = df[col].rolling(10).apply(lambda x: (1 + x).prod() - 1)

        # 重置索引
        df.reset_index(inplace=True)
        df['difference'] = df['沪深300'] - df['国证2000']
        df = df[['valuation_date', 'difference']]
        df.dropna(inplace=True)
        df_vix=self.dp.raw_vix_withdraw()
        df=df.merge(df_vix,on='valuation_date',how='left')
        return df



if __name__ == "__main__":
    dpro=data_processing()
    df=dpro.term_spread_9Y()
    df.set_index('valuation_date',inplace=True)
    df.plot()
    plt.show()