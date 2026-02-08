import numpy as np
from sklearn.preprocessing import MinMaxScaler
from train import config

def preprocess_data(df, test_ratio=0.2, validation_ratio=0.1):
    # 明确定义特征顺序
    feature_columns = [
        'Temperature (°C)', 'Package Energy', 'DRAM Energy', 'CPU Frequency (kHz)',
        'CPU Voltage', 'CPU Load', 'Instructions',
        'Cycles', 'Cache References', 'Cache Misses',
        'Bus Cycles', 'Branch Instructions',
        'Branch Misses', 'Task Clock',
        'L1 Data Reads', 'L1 Data Writes',
        'L1 Data Misses', 'L1 Ins Misses',
        'L2 Read Misses', 'Content Switches',
        'Page_Faults'
    ]
    features_df = df[feature_columns]

    # 填充缺失值
    features_df = features_df.interpolate(method='linear')

    # 数据归一化
    scalers = {}
    scaled_features = np.zeros_like(features_df.values)
    for i, col in enumerate(feature_columns):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features[:, i] = scaler.fit_transform(features_df[col].values.reshape(-1, 1)).flatten()
        scalers[col] = scaler

    # 划分训练集、验证集和测试集
    total_size = len(scaled_features)
    test_size = int(total_size * test_ratio)
    validation_size = int(total_size * validation_ratio)
    train_size = total_size - test_size - validation_size

    return scaled_features, scalers, feature_columns, train_size, validation_size, test_size


def create_sequences(data, feature_columns, seq_length, pred_length):
    target_idx = feature_columns.index(config["target_column"])

    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:i + seq_length])  # [seq_length, num_features]
        # 使用目标列索引获取目标值
        y.append(data[i + seq_length:i + seq_length + pred_length, target_idx])
    return np.array(X), np.array(y)