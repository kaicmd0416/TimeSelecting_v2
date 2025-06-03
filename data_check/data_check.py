import pandas as pd
import os
import sys
import yaml
import logging
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import global_setting.global_dic as glv
import global_tools_func.global_tools as gt

class DataChecker:
    def __init__(self, target_date):
        """
        初始化数据检查器
        
        Args:
            target_date: 目标日期，格式为'YYYY-MM-DD'
        """
        target_date=gt.last_workday_calculate(target_date)
        self.target_date = target_date
        # 计算过去一年的日期范围
        target_datetime = pd.to_datetime(target_date)
        self.start_date = (target_datetime - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        self.end_date = target_date
        self.working_days = gt.working_days_list(self.start_date, self.end_date)
        self.config = self._load_config()
        self._ensure_directories()
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志"""
        log_filename = f"timeSelectingChecking_{datetime.now().strftime('%Y%m%d')}.log"
        self.log_path = os.path.join(self.report_dir, log_filename)
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_path, encoding='utf-8'),
                logging.StreamHandler()  # 同时输出到控制台
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"开始数据检查 - 检查期间: {self.start_date} 到 {self.end_date}")
        self.logger.info(f"目标日期: {self.target_date}")
        
    def _load_config(self):
        """加载配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), 'config_checking.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        # 创建报告保存目录
        self.report_dir = os.path.join(project_root, 'data_check', 'reports')
        os.makedirs(self.report_dir, exist_ok=True)

    def get_data_paths(self, signal_name):
        """获取信号对应的数据路径"""
        if signal_name not in self.config['raw_data_paths']:
            raise ValueError(f"信号 {signal_name} 未在配置文件中找到")
            
        paths = self.config['raw_data_paths'][signal_name]['path']
        if isinstance(paths, str):
            paths = [paths]
        return paths
        
    def check_file_dates(self, file_path):
        """检查单个文件的日期完整性"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']  # 常见的中文编码
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                if 'valuation_date' not in df.columns:
                    return False, "文件缺少valuation_date列"
                
                # 确保日期格式正确
                df['valuation_date'] = pd.to_datetime(df['valuation_date'])
                
                # 获取文件中的日期列表
                file_dates = set(df['valuation_date'].dt.strftime('%Y-%m-%d').tolist())
                
                # 检查是否包含所有工作日
                missing_dates = set(self.working_days) - file_dates
                if missing_dates:
                    return False, f"缺少以下日期: {sorted(list(missing_dates))}"
                
                return True, "日期完整"
                
            except UnicodeDecodeError:
                continue  # 尝试下一个编码
            except Exception as e:
                return False, f"文件读取错误: {str(e)}"
        
        return False, "无法使用支持的编码读取文件（尝试了：utf-8, gbk, gb2312, gb18030）"
    
    def check_raw_data(self, signal_name):
        """检查指定信号的数据完整性"""
        status = 'normal'  # 初始化状态为normal
        try:
            data_paths = self.get_data_paths(signal_name)
            self.logger.info(f"\n检查信号: {signal_name}")
            results = []
            
            for path_key in data_paths:
                try:
                    data_path = glv.get(path_key)
                    
                    if not data_path:
                        msg = f"路径 {path_key} 未在global_dic中定义"
                        self.logger.error(msg)
                        status = 'error'
                        continue
                        
                    if not os.path.exists(data_path):
                        msg = f"文件不存在: {data_path}"
                        self.logger.error(msg)
                        status = 'error'
                        continue
                    
                    # 直接检查CSV文件
                    is_complete, message = self.check_file_dates(data_path)
                    if not is_complete:
                        self.logger.warning(f"文件 {os.path.basename(data_path)} 数据不完整: {message}")
                        status = 'error'
                    else:
                        self.logger.info(f"文件 {os.path.basename(data_path)} 数据完整")
                
                except Exception as e:
                    msg = f"处理路径 {path_key} 时出错: {str(e)}"
                    self.logger.error(msg)
                    status = 'error'
            
            self.logger.info(f"信号 {signal_name} 检查完成，状态: {status}\n")
            return status
            
        except Exception as e:
            self.logger.error(f"检查过程出错: {str(e)}")
            return 'error'

    def check_signal_data(self, signal_name):
        """
        检查信号数据文件夹中的子文件夹数据完整性
        
        Args:
            signal_name: 信号名称，对应子文件夹名
            
        Returns:
            tuple: (status, latest_date)
                - status: 'normal' 或 'error'
                - latest_date: 最新数据日期，格式为 'YYYY-MM-DD'，如果出错则为 None
        """
        status = 'normal'
        latest_date = None
        try:
            # 获取信号数据根路径
            base_path = glv.get('signal_booster_output')
            if not base_path:
                self.logger.error("signal_booster_output 路径未在global_dic中定义")
                return 'error', None
                
            # 构建完整的信号数据路径
            signal_path = os.path.join(base_path, 'prod', signal_name)
            self.logger.info(f"\n检查信号数据: {signal_name}")
            self.logger.info(f"数据路径: {signal_path}")
            
            if not os.path.exists(signal_path):
                self.logger.error(f"信号数据文件夹不存在: {signal_path}")
                return 'error', '2015-01-01'
            
            # 获取文件夹中的所有CSV文件
            csv_files = [f for f in os.listdir(signal_path) if f.endswith('.csv')]
            if not csv_files:
                self.logger.error(f"信号数据文件夹中没有CSV文件: {signal_path}")
                return 'error', '2015-01-01'
            
            # 从文件名中提取日期并检查完整性
            file_dates = set()
            
            for file_name in csv_files:
                try:
                    # 从文件名中提取日期（假设格式为XXXX_YYYYMMDD.csv）
                    date_str = file_name.split('_')[-1].replace('.csv', '')
                    if len(date_str) == 8:  # 确保日期格式正确
                        file_date = pd.to_datetime(date_str, format='%Y%m%d')
                        file_dates.add(file_date.strftime('%Y-%m-%d'))
                        
                        # 更新最新日期
                        if latest_date is None or file_date > latest_date:
                            latest_date = file_date
                except Exception as e:
                    self.logger.warning(f"从文件名 {file_name} 提取日期时出错: {str(e)}")
                    continue
            
            if not file_dates:
                self.logger.error("未能从任何文件名中提取到有效日期")
                return 'error', None
            
            # 检查是否包含所有工作日
            missing_dates = set(self.working_days) - file_dates
            if missing_dates:
                self.logger.warning(f"缺少以下日期的数据: {sorted(list(missing_dates))}")
                status = 'error'
            else:
                self.logger.info("所有工作日数据完整")
            
            # 输出最新数据日期
            if latest_date:
                self.logger.info(f"最新数据日期: {latest_date.strftime('%Y-%m-%d')}")
            
            self.logger.info(f"信号 {signal_name} 检查完成，状态: {status}\n")
            return status, latest_date.strftime('%Y-%m-%d') if latest_date else None
            
        except Exception as e:
            self.logger.error(f"检查过程出错: {str(e)}")
            return 'error', None

if __name__ == "__main__":
    # 使用示例
    checker = DataChecker(target_date='2025-03-18')
    
    # 检查信号数据
    signal_names = ['Shibor_2W_45D_9_combine']
    for signal in signal_names:
        status, latest_date = checker.check_signal_data(signal)
        print(f"信号 {signal} 状态: {status}, 最新日期: {latest_date}") 
