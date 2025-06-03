import pandas as pd
import os
import glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import silhouette_score
import warnings
import yaml
warnings.filterwarnings('ignore')

def load_signal_data(directory_path):
    """
    Load all signal data files from the specified directory and combine them into a single DataFrame.
    Each CSV file has multiple columns, where the first column is 'valuation_date' and other columns
    are factor names with their values.

    Args:
        directory_path: Path to the directory containing signal data files
        remove_constant_cols: Whether to remove columns where all values are the same (default: False)
                             Set to False for dummy variables (0/1 indicators)
        min_valid_dates_pct: Minimum percentage of valid dates required for a row to be kept (0-100)
                            Default is 1%, meaning rows with less than 1% valid data will be dropped

    Returns:
        Combined DataFrame with date as index and signals as columns
    """
    print(f"Loading signal data from {directory_path}")

    # Get a list of all CSV files in the directory
    file_list = glob.glob(os.path.join(directory_path, "*.csv"))
    print(f"Found {len(file_list)} files")

    if len(file_list) == 0:
        print("No CSV files found in the directory!")
        return None

    # Try to load all data from a single file first
    # If there's only one file and it contains all the data we need, use it directly
    if len(file_list) == 1:
        try:
            file_path = file_list[0]
            df = pd.read_csv(file_path)

            # Validate the data format
            if 'valuation_date' in df.columns and len(df.columns) > 1:
                print(f"Loading data from single file: {os.path.basename(file_path)}")

                # Convert date to datetime
                df['valuation_date'] = pd.to_datetime(df['valuation_date'], errors='coerce')
                df = df.dropna(subset=['valuation_date'])

                # Set date as index
                df.set_index('valuation_date', inplace=True)

                print(f"Loaded data with shape: {df.shape}")
                print(f"Non-null values: {df.count().sum()} of {df.size} ({df.count().sum() / df.size * 100:.2f}%)")

                # Print the first few rows to verify
                print("\nFirst few rows of data:")
                print(df.head())

                return df

        except Exception as e:
            print(f"Error loading single file: {str(e)}")
            # Continue with the regular loading process

    # Initialize a dictionary to store data by date
    date_data = {}

    # Process each file
    for file_path in file_list:
        file_name = os.path.basename(file_path).replace('.csv', '')
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Check if 'valuation_date' exists in the columns
            if 'valuation_date' not in df.columns:
                print(f"Skipping {file_name}: 'valuation_date' column not found")
                continue

            # Check for additional columns
            if len(df.columns) <= 1:
                print(f"Skipping {file_name}: No signal columns found besides valuation_date")
                continue

            # Convert date to datetime
            df['valuation_date'] = pd.to_datetime(df['valuation_date'], errors='coerce')

            # Check for invalid dates
            invalid_dates = df['valuation_date'].isna().sum()
            if invalid_dates > 0:
                print(f"Warning: {file_name} contains {invalid_dates} invalid dates which will be dropped")
                df = df.dropna(subset=['valuation_date'])

            # Process each row to build our date_data dictionary
            for _, row in df.iterrows():
                date = row['valuation_date']
                if date not in date_data:
                    date_data[date] = {}

                # Add each column (except valuation_date) to the date's data
                for col in df.columns:
                    if col != 'valuation_date':
                        # Don't overwrite existing data
                        if col not in date_data[date]:
                            date_data[date][col] = row[col]

            print(f"Processed {file_name}: {len(df)} rows, {len(df.columns) - 1} signals/factors")

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

    # Create DataFrame from the collected data
    if date_data:
        # Convert dictionary to DataFrame
        index = sorted(date_data.keys())
        columns = set()
        for date_dict in date_data.values():
            columns.update(date_dict.keys())
        columns = sorted(columns)

        # Create empty DataFrame
        combined_df = pd.DataFrame(index=index, columns=columns)
        combined_df.index.name = 'valuation_date'

        # Fill in the data
        for date, date_dict in date_data.items():
            for col, value in date_dict.items():
                combined_df.loc[date, col] = value

        print(f"Combined DataFrame shape: {combined_df.shape}")
        print(
            f"Non-null values: {combined_df.count().sum()} of {combined_df.size} ({combined_df.count().sum() / combined_df.size * 100:.2f}%)")
        calculate_autocorrelation(combined_df)
        return combined_df
    else:
        print("No valid data found")
        return None

def preprocess_data(df, min_valid_dates_pct=10):
    """
    预处理信号数据用于聚类
    
    Args:
        df: DataFrame包含信号数据
        min_valid_dates_pct: 最小有效日期百分比阈值
        
    Returns:
        处理后的数据和特征名称
    """
    print("\n预处理数据用于聚类...")
    
    # 创建数据副本
    data = df.copy()
    
    # 检查缺失值
    total_cells = data.size
    missing_cells = data.isna().sum().sum()
    print(f"数据中缺失值比例: {missing_cells/total_cells*100:.2f}%")
    
    # 移除缺失值过多的列
    threshold = data.shape[0] * (min_valid_dates_pct/100)
    min_count = data.count()
    drop_cols = min_count[min_count < threshold].index.tolist()
    
    if drop_cols:
        print(f"移除 {len(drop_cols)} 个缺失值过多的列...")
        data = data.drop(columns=drop_cols)
    
    # 填充剩余缺失值
    data = data.fillna(data.mean())
    
    # 转置数据: 因子为行，日期为列
    transposed_data = data.T
    print(f"转置后的数据形状: {transposed_data.shape} (因子 × 日期)")
    
    # 获取特征名称
    feature_names = transposed_data.index.tolist()
    
    return transposed_data, feature_names

def find_optimal_clusters(data, max_clusters=30):
    """
    找到最优的聚类数量
    
    Args:
        data: 预处理后的数据
        max_clusters: 最大聚类数量
        
    Returns:
        最优聚类数量
    """
    print("\n寻找最优聚类数量...")
    
    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 计算不同聚类数量的惯性(inertia)和轮廓分数(silhouette score)
    inertia_values = []
    silhouette_values = []
    k_values = range(2, min(max_clusters + 1, data.shape[0]))
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertia_values.append(kmeans.inertia_)
        
        # 计算轮廓分数 (只在数据量较小时计算，否则会很慢)
        if data.shape[0] <= 1000:
            try:
                silhouette_values.append(silhouette_score(scaled_data, kmeans.labels_))
            except:
                silhouette_values.append(0)
    
    # 绘制肘部法则图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(list(k_values), inertia_values, 'bo-')
    plt.xlabel('聚类数量')
    plt.ylabel('惯性 (Inertia)')
    plt.title('肘部法则')
    plt.grid(True, alpha=0.3)
    
    # 绘制轮廓分数图
    if silhouette_values:
        plt.subplot(1, 2, 2)
        plt.plot(list(k_values), silhouette_values, 'ro-')
        plt.xlabel('聚类数量')
        plt.ylabel('轮廓分数 (Silhouette Score)')
        plt.title('轮廓分数分析')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 创建输出目录
    output_dir = 'kcluster_results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'optimal_clusters.png'), dpi=300)
    plt.close()
    
    # 使用肘部法则确定最优聚类数量
    # 计算曲线的曲率，曲率最大的点为肘部点
    k_diff = np.diff(inertia_values)
    k_diff2 = np.diff(k_diff)
    
    # 选择肘部点或使用比例方法
    if len(k_diff2) > 0:
        elbow_idx = np.argmax(np.abs(k_diff2)) + 1  # +1 因为我们对差分做了两次
        optimal_k = list(k_values)[elbow_idx]
    else:
        # 如果无法确定肘部点，使用经验法则
        optimal_k = max(2, int(np.sqrt(data.shape[0] / 2)))
    
    # 如果有轮廓分数，也考虑轮廓分数最大的点
    if silhouette_values:
        silhouette_k = list(k_values)[np.argmax(silhouette_values)]
        # 输出两种方法的结果
        print(f"肘部法则建议的聚类数量: {optimal_k}")
        print(f"轮廓分数最大的聚类数量: {silhouette_k}")
        
        # 选择较小的一个作为最终结果
        optimal_k = min(optimal_k, silhouette_k)
    
    print(f"选择的最优聚类数量: {optimal_k}")
    return optimal_k

def perform_kmeans_clustering(data, n_clusters, feature_names):
    """
    执行K-means聚类
    
    Args:
        data: 预处理后的数据
        n_clusters: 聚类数量
        feature_names: 特征名称列表
        
    Returns:
        聚类结果和聚类中心
    """
    print(f"\n执行K-means聚类 (k={n_clusters})...")
    
    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # 获取聚类中心
    centers = kmeans.cluster_centers_
    
    # 创建聚类结果DataFrame
    cluster_results = pd.DataFrame({
        'factor': feature_names,
        'cluster': cluster_labels
    })
    
    # 计算每个聚类的大小
    cluster_sizes = cluster_results['cluster'].value_counts().sort_index()
    print("\n各聚类大小:")
    for cluster_id, size in cluster_sizes.items():
        print(f"聚类 {cluster_id}: {size} 个因子")
    
    return cluster_results, centers, scaled_data

def visualize_clusters(data, labels, centers, feature_names):
    """
    可视化聚类结果
    
    Args:
        data: 标准化后的数据
        labels: 聚类标签
        centers: 聚类中心
        feature_names: 特征名称
    """
    print("\n可视化聚类结果...")
    
    # 创建输出目录
    output_dir = 'kcluster_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 使用PCA降维以便可视化
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centers_2d = pca.transform(centers)
    
    # 绘制PCA降维后的聚类
    plt.figure(figsize=(12, 10))
    
    # 绘制数据点
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    
    # 绘制聚类中心
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', s=100, marker='X')
    
    # 添加因子标签（仅当数据点不太多时）
    if len(feature_names) <= 50:
        for i, (x, y) in enumerate(data_2d):
            plt.annotate(feature_names[i], (x, y), fontsize=8, alpha=0.7)
    
    plt.title('K-Means聚类结果 (PCA降维)')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.colorbar(scatter, label='聚类')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_visualization_pca.png'), dpi=300)
    plt.close()
    
    # 2. 热图展示各聚类的时间序列特征
    try:
        original_data = data.T  # 转换回日期×因子
        n_clusters = len(np.unique(labels))
        
        # 计算每个聚类的平均时间序列
        cluster_avg_series = []
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) > 0:
                cluster_data = original_data.iloc[:, cluster_indices]
                cluster_avg = cluster_data.mean(axis=1)
                cluster_avg_series.append(cluster_avg)
            else:
                # 如果聚类为空，添加零序列
                cluster_avg_series.append(pd.Series(0, index=original_data.index))
        
        # 创建热图数据
        heatmap_data = pd.DataFrame({f'聚类 {i}': series for i, series in enumerate(cluster_avg_series)})
        
        # 绘制热图
        plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_data.T, cmap='coolwarm', center=0)
        plt.title('各聚类平均时间序列特征')
        plt.xlabel('日期')
        plt.ylabel('聚类')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_time_series_heatmap.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"绘制时间序列热图时出错: {str(e)}")
    
    # 3. 聚类结果饼图
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(10, 8))
    plt.pie(cluster_counts, labels=[f'聚类 {i}\n({count}个因子)' for i, count in enumerate(cluster_counts)],
            autopct='%1.1f%%', startangle=90, shadow=False)
    plt.axis('equal')
    plt.title('因子聚类分布')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_distribution_pie.png'), dpi=300)
    plt.close()

def save_cluster_results(cluster_results, data_shape, original_df, n_clusters):
    """
    保存聚类结果
    
    Args:
        cluster_results: 聚类结果DataFrame
        data_shape: 原始数据形状
        original_df: 原始数据DataFrame
        n_clusters: 聚类数量
    """
    print("\n保存聚类结果...")
    
    # 创建输出目录（用于存储图片）
    output_dir = 'kcluster_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为YAML格式并保存到config_project文件夹
    config_dir = 'config_project'
    os.makedirs(config_dir, exist_ok=True)
    
    # 将DataFrame转换为字典结构，适合YAML格式
    yaml_data = {}
    for _, row in cluster_results.iterrows():
        factor = row['factor']
        cluster = int(row['cluster'])
        if cluster not in yaml_data:
            yaml_data[cluster] = []
        yaml_data[cluster].append(factor)
    
    # 保存为YAML文件
    yaml_path = os.path.join(config_dir, 'kcluster_new.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=True)
    
    print(f"聚类结果以YAML格式保存到: {config_dir}/kcluster.yaml")

def select_representative_factors(cluster_results, original_df, n_clusters):
    """
    为每个聚类选择代表性因子
    
    Args:
        cluster_results: 聚类结果DataFrame
        original_df: 原始数据DataFrame
        n_clusters: 聚类数量
        
    Returns:
        每个聚类的代表性因子
    """
    print("\n为每个聚类选择代表性因子...")
    
    representative_factors = {}
    
    # 对每个聚类选择代表性因子
    for cluster_id in range(n_clusters):
        # 获取该聚类的所有因子
        cluster_factors = cluster_results[cluster_results['cluster'] == cluster_id]['factor'].tolist()
        
        if len(cluster_factors) == 0:
            representative_factors[cluster_id] = None
            continue
        
        if len(cluster_factors) == 1:
            representative_factors[cluster_id] = cluster_factors[0]
            continue
        
        # 使用方法1: 选择与聚类内其他因子平均相关性最高的因子
        try:
            # 获取该聚类所有因子的数据
            cluster_data = original_df[cluster_factors]
            
            # 计算相关性矩阵
            corr_matrix = cluster_data.corr()
            
            # 计算每个因子与其他因子的平均相关性
            avg_correlations = {}
            for factor in cluster_factors:
                factor_corr = corr_matrix[factor].drop(factor)  # 排除自身
                avg_correlations[factor] = factor_corr.abs().mean()  # 使用绝对值相关性
            
            # 选择平均相关性最高的因子作为代表
            best_factor = max(avg_correlations, key=avg_correlations.get)
            representative_factors[cluster_id] = best_factor
            
        except Exception as e:
            # 如果上述方法失败，使用方法2: 简单选择第一个因子
            print(f"选择聚类 {cluster_id} 代表性因子时出错: {str(e)}")
            representative_factors[cluster_id] = cluster_factors[0]
    
    # 输出结果
    print("\n各聚类代表性因子:")
    for cluster_id, factor in representative_factors.items():
        if factor:
            print(f"聚类 {cluster_id}: {factor}")
        else:
            print(f"聚类 {cluster_id}: 无因子")
    
    return representative_factors

def generate_cluster_signals(cluster_results, original_df, n_clusters):
    """
    为每个聚类生成代表性时间序列信号
    
    Args:
        cluster_results: 聚类结果DataFrame
        original_df: 原始数据DataFrame
        n_clusters: 聚类数量
        
    Returns:
        包含每个聚类代表性信号的DataFrame
    """
    print("\n生成聚类代表性时间序列信号...")
    
    # 创建输出目录
    output_dir = 'kcluster_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建结果DataFrame，索引使用原始数据的日期
    result_df = pd.DataFrame(index=original_df.index)
    
    # 对每个聚类进行处理
    for cluster_id in range(n_clusters):
        # 获取该聚类的所有因子
        cluster_factors = cluster_results[cluster_results['cluster'] == cluster_id]['factor'].tolist()
        
        if len(cluster_factors) == 0:
            # 如果聚类为空，创建全0的序列
            result_df[f'K{cluster_id+1}'] = 0
            print(f"聚类 {cluster_id+1} (K{cluster_id+1}): 空聚类，使用全0序列")
            continue
        
        # 提取该聚类所有因子的数据
        cluster_data = original_df[cluster_factors]
        
        # 计算每个时间点的均值
        cluster_mean = cluster_data.mean(axis=1)
        
        # 根据阈值将均值离散化 (< 0.5 -> 0, == 0.5 -> 0.5, > 0.5 -> 1)
        discretized = cluster_mean.apply(lambda x: 0 if x < 0.5 else (0.5 if x == 0.5 else 1))
        
        # 添加到结果DataFrame
        result_df[f'K{cluster_id+1}'] = discretized
        
        # 统计0、0.5和1的数量
        count_0 = (discretized == 0).sum()
        count_05 = (discretized == 0.5).sum()
        count_1 = (discretized == 1).sum()
        total = len(discretized)
        
        print(f"聚类 {cluster_id+1} (K{cluster_id+1}): {len(cluster_factors)}个因子, 信号分布: 0={count_0}({count_0/total*100:.1f}%), 0.5={count_05}({count_05/total*100:.1f}%), 1={count_1}({count_1/total*100:.1f}%)")
    
    try:
        # 生成热图可视化
        plt.figure(figsize=(16, 10))
        sns.heatmap(result_df, cmap='coolwarm', center=0.5, vmin=0, vmax=1)
        plt.title('聚类代表性信号时间序列')
        plt.ylabel('日期')
        plt.xlabel('聚类信号')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_signals_heatmap.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"生成热图时出错: {str(e)}")
    
    try:
        # 计算并可视化协方差矩阵
        plot_covariance_matrix(result_df, output_dir)
    except Exception as e:
        print(f"生成协方差矩阵时出错: {str(e)}")
    
    return result_df

def plot_covariance_matrix(df, output_dir):
    """
    计算并可视化聚类信号的协方差矩阵和相关系数矩阵
    
    Args:
        df: 包含聚类信号的DataFrame
        output_dir: 输出目录
    """
    print("\n计算并可视化协方差矩阵...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算协方差矩阵
    cov_matrix = df.cov()
    
    try:
        # 创建协方差矩阵热图
        plt.figure(figsize=(14, 12))
        sns.heatmap(cov_matrix, annot=True, fmt=".3f", cmap='coolwarm', center=0, 
                    square=True, linewidths=.5, cbar_kws={"shrink": .8})
        plt.title('聚类信号协方差矩阵', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_covariance_matrix.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"生成协方差矩阵热图时出错: {str(e)}")
    
    # 计算相关系数矩阵
    corr_matrix = df.corr()
    
    try:
        # 创建相关系数矩阵热图
        plt.figure(figsize=(14, 12))
        # 生成掩码用于隐藏上三角矩阵
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                    vmin=-1, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8})
        plt.title('聚类信号相关系数矩阵', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_correlation_matrix.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"生成相关系数矩阵热图时出错: {str(e)}")
    
    # 计算并显示协方差矩阵和相关系数矩阵的统计信息
    print(f"协方差矩阵大小: {cov_matrix.shape}")
    
    # 统计强相关信号对的数量 (|相关系数| > 0.5)
    strong_corr = (corr_matrix.abs() > 0.5).sum().sum() - corr_matrix.shape[0]  # 减去对角线元素
    print(f"强相关信号对数量 (|相关系数| > 0.5): {strong_corr//2}")  # 除以2因为矩阵是对称的
    
    # 计算平均相关系数（不包括对角线）
    diag_mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    mean_corr = corr_matrix.values[diag_mask].mean()
    print(f"平均相关系数: {mean_corr:.4f}")
    
    return cov_matrix, corr_matrix

def calculate_autocorrelation(signals_df):
    """
    计算每个信号的自相关系数
    
    Args:
        signals_df: 包含所有信号的DataFrame
        
    Returns:
        自相关分析结果的DataFrame
    """
    print("\n计算信号自相关系数...")
    
    # 创建结果存储字典
    results = {
        'signal': [],
        'lag1_corr': [],
        'lag5_corr': [],
        'lag10_corr': [],
        'mean': [],
        'std': []
    }
    
    # 对每个信号列进行处理
    for column in signals_df.columns:
        if column != 'valuation_date':  # 跳过日期列
            series = signals_df[column]
            
            # 计算不同lag的自相关系数
            lag1_corr = series.autocorr(lag=1) if len(series) > 1 else np.nan
            lag5_corr = series.autocorr(lag=5) if len(series) > 5 else np.nan
            lag10_corr = series.autocorr(lag=10) if len(series) > 10 else np.nan
            
            # 计算基本统计量
            mean_val = series.mean()
            std_val = series.std()
            
            # 存储结果
            results['signal'].append(column)
            results['lag1_corr'].append(lag1_corr)
            results['lag5_corr'].append(lag5_corr)
            results['lag10_corr'].append(lag10_corr)
            results['mean'].append(mean_val)
            results['std'].append(std_val)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    
    # 打印结果
    print("\n自相关分析结果:")
    print(results_df.to_string(float_format=lambda x: '{:.4f}'.format(x)))
    
    # 保存结果
    output_dir = 'kcluster_results'
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'autocorrelation_analysis.csv'), index=False)
    print(f"\n自相关分析结果已保存至: {output_dir}/autocorrelation_analysis.csv")
    
    # 创建自相关系数热图
    plt.figure(figsize=(12, 8))
    corr_data = results_df[['lag1_corr', 'lag5_corr', 'lag10_corr']]
    sns.heatmap(corr_data.T, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                xticklabels=results_df['signal'], yticklabels=['Lag 1', 'Lag 5', 'Lag 10'])
    plt.title('信号自相关系数热图')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'autocorrelation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return results_df

def perform_factor_clustering(data_directory, max_clusters=30, auto_select_k=True, n_clusters=15):
    """
    执行因子聚类分析主函数
    
    Args:
        data_directory: 数据目录路径
        max_clusters: 最大聚类数量
        auto_select_k: 是否自动选择聚类数量
        n_clusters: 如果不自动选择，指定的聚类数量
        
    Returns:
        聚类结果和代表性因子
    """
    # 加载数据
    df = load_signal_data(data_directory)
    
    if df is None or df.empty:
        print("未能加载有效数据，分析终止")
        return None, None, None
    
    # 预处理数据
    transposed_data, feature_names = preprocess_data(df)
    
    # 如果需要自动选择聚类数量
    if auto_select_k:
        n_clusters = find_optimal_clusters(transposed_data, max_clusters)
    else:
        print(f"\n使用指定的聚类数量: {n_clusters}")
    
    # 执行K-means聚类
    cluster_results, centers, scaled_data = perform_kmeans_clustering(transposed_data, n_clusters, feature_names)
    
    # 可视化聚类结果
    visualize_clusters(scaled_data, cluster_results['cluster'].values, centers, feature_names)
    
    # 选择代表性因子
    representative_factors = select_representative_factors(cluster_results, df, n_clusters)
    
    # 保存聚类结果
    save_cluster_results(cluster_results, transposed_data.shape, df, n_clusters)
    
    # 生成聚类信号
    cluster_signals = generate_cluster_signals(cluster_results, df, n_clusters)
    
    print("\nK-means聚类分析完成")
    
    return cluster_results, representative_factors, cluster_signals

def main():
    """
    主函数 - 执行聚类分析流程
    """
    # 设置数据目录
    data_directory = r"D:\Signal\signal_data\prod\combine"
    
    # 检查目录是否存在
    if not os.path.exists(data_directory):
        print(f"错误: 目录 {data_directory} 不存在!")
        print("请指定正确的信号数据目录路径。")
        return
    
    # 创建输出目录
    output_dir = 'kcluster_results'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 执行因子聚类分析
        # auto_select_k=True 表示自动选择最优聚类数量
        # n_clusters=22 表示当auto_select_k=False时使用的聚类数量
        cluster_results, representative_factors, cluster_signals = perform_factor_clustering(
            data_directory, 
            max_clusters=30, 
            auto_select_k=False,
            n_clusters=22
        )
        
        if cluster_results is not None:
            print("\n因子聚类分析完成。")
            print(f"可视化结果保存在 'kcluster_results' 目录")
            print(f"聚类配置保存在 'config_project/kcluster.yaml'")
            
            # 显示聚类信号DataFrame的形状
            if cluster_signals is not None:
                print(f"\n聚类信号DataFrame形状: {cluster_signals.shape}")
                
                print("\n已保存的关键图片:")
                print("- cluster_visualization_pca.png: 聚类PCA可视化")
                print("- cluster_covariance_matrix.png: 协方差矩阵可视化")
                print("- cluster_correlation_matrix.png: 相关系数矩阵可视化")
                print("- cluster_signals_heatmap.png: 聚类信号热图")
        else:
            print("\n因子聚类分析失败。")
    except Exception as e:
        import traceback
        print(f"\n执行聚类分析时发生错误: {str(e)}")
        traceback.print_exc()
        print("\n请检查数据路径和格式是否正确。")

if __name__ == "__main__":
    # main()
    data_directory = r"D:\Signal\signal_data\prod\combine"
    load_signal_data(data_directory)
