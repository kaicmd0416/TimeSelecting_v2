import os
import sys
import pandas as pd
path = os.getenv('GLOBAL_TOOLSFUNC')
sys.path.append(path)
import global_tools as gt
import global_setting.global_dic as glv

class L2_signalConstruction:
    def __init__(self, signal_name, mode):
        """
        初始化L2信号构建类
        
        Parameters:
        -----------
        signal_name : str
            信号名称
        mode : str
            模式（如'test'或'prod'）
        """
        self.signal_name = signal_name
        self.mode = mode
        self.input_path = self._get_input_path()
        self.output_path = self._get_output_path()
        
    def _get_input_path(self):
        """获取输入路径"""
        input_path = glv.get('signal_data')
        input_path = os.path.join(input_path, self.mode)
        input_path = os.path.join(input_path, self.signal_name)
        return input_path
    
    def _get_output_path(self):
        """获取输出路径"""
        output_path = glv.get('signal_data')
        output_path = os.path.join(output_path, self.mode)
        output_path = os.path.join(output_path, self.signal_name)
        return output_path
    
    def process_signal(self):
        """处理信号的主函数"""
        # 确保输出目录存在
        gt.folder_creator2(self.output_path)
        
        # 获取所有输入文件
        input_files = [f for f in os.listdir(self.input_path) if f.endswith('.csv')]
        input_files.sort()
        
        for file in input_files:
            input_file_path = os.path.join(self.input_path, file)
            output_file_path = os.path.join(self.output_path, file)
            
            # 读取数据
            df = gt.readcsv(input_file_path)
            
            # 处理数据
            df = self._process_data(df)
            
            # 保存结果
            gt.savecsv(df, output_file_path)
            
    def _process_data(self, df):
        """
        处理数据的函数，可以根据具体需求进行修改
        
        Parameters:
        -----------
        df : pandas.DataFrame
            输入数据框
            
        Returns:
        --------
        pandas.DataFrame
            处理后的数据框
        """
        # 这里添加具体的数据处理逻辑
        return df
    
    def run(self):
        """运行信号构建的主函数"""
        try:
            print(f"开始处理信号: {self.signal_name}")
            self.process_signal()
            print(f"信号处理完成: {self.signal_name}")
        except Exception as e:
            print(f"处理信号时出错: {str(e)}")
            raise

if __name__ == "__main__":
    # 示例使用
    signal_name = "M1M2"  # 示例信号名称
    mode = "test"         # 示例模式
    
    signal_constructor = L2_signalConstruction(signal_name, mode)
    signal_constructor.run() 