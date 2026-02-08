import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, r2_score
import matplotlib.pyplot as plt
from models import EnhancedCNNSeq2Seq
import warnings
import json
import os
import random
from utils import create_sequences, preprocess_data

config = {
    "seq_length": 30,
    "pred_length": 20,
    "batch_size": 64,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "num_features": 21,
    "test_ratio": 0.2,
    "validation_ratio": 0.1,
    "cnn_channels": 64,
    "target_column": "Temperature (°C)",
    "patience": 5,
    "min_delta": 0.0001,
}

warnings.filterwarnings('ignore')

# 设置随机种子
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

plt.rcParams["font.family"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

def calculate_validation_loss(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, targets)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=10,
                min_delta=0.001):
    model.train()
    losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        val_loss = calculate_validation_loss(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}')

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
            print(f"Validation loss improved to {best_val_loss:.6f}, saving best model")
        else:
            epochs_without_improvement += 1
            print(f"Validation loss did not improve for {epochs_without_improvement}/{patience} epochs")

            # 早停条件满足
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                model.load_state_dict(best_model_state)  # 恢复最佳模型
                break

    plt.figure(figsize=(12, 6))
    plt.plot(losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()

    return model

def evaluate_model(model, test_loader, scalers, feature_columns, device):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())

    predictions = np.concatenate(predictions).squeeze()  # [samples, pred_length]
    actuals = np.concatenate(actuals).squeeze()  # [samples, pred_length]

    # 反归一化温度数据
    temp_scaler = scalers['Temperature (°C)']
    predictions = temp_scaler.inverse_transform(predictions)
    actuals = temp_scaler.inverse_transform(actuals)

    # 基础指标
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    r2 = r2_score(actuals.ravel(), predictions.ravel())  # 新增

    # 热中断状态二值化（每个样本只要有一个时间步超过阈值即视为发生）
    heat_threshold = 90
    pred_heat = (predictions > heat_threshold).any(axis=1)  # [samples]
    actual_heat = (actuals > heat_threshold).any(axis=1)  # [samples]

    # 计算分类
    pred_scores = np.max(predictions, axis=1)
    fpr, tpr, _ = roc_curve(actual_heat, pred_scores)
    roc_auc = auc(fpr, tpr)

    tn, fp, fn, tp = confusion_matrix(actual_heat, pred_heat).ravel()
    confusion = {
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp
    }

    # 计算分类指标
    try:
        precision = precision_score(actual_heat, pred_heat)
        recall = recall_score(actual_heat, pred_heat)
        f1 = f1_score(actual_heat, pred_heat)
    except ZeroDivisionError:
        precision = recall = f1 = 0.0

    # 热中断发生时间差（仅计算同时发生的样本）
    heat_occurrence_time_diff = []
    for i in range(len(pred_heat)):
        if pred_heat[i] and actual_heat[i]:
            pred_first = np.argmax(predictions[i] > heat_threshold)
            actual_first = np.argmax(actuals[i] > heat_threshold)
            heat_occurrence_time_diff.append(abs(pred_first - actual_first))
        else:
            heat_occurrence_time_diff.append(-1)

    valid_diffs = [d for d in heat_occurrence_time_diff if d != -1]
    avg_time_diff = np.mean(valid_diffs) if valid_diffs else 0

    # 打印所有指标
    print(f'\n{"=" * 30} 热中断预测评估 {"=" * 30}')
    print(f'基础指标：')
    print(f'MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}')

    print('\n混淆矩阵：')
    cm = confusion_matrix(actual_heat, pred_heat, labels=[False, True])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['未发生', '发生'], yticklabels=['未发生', '发生'])
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('热中断预测混淆矩阵')
    plt.savefig('confusion_matrix.png')
    plt.close()

    print('\n分类指标：')
    print(f'精确率(Precision): {precision * 100:.2f}%')
    print(f'召回率(Recall): {recall * 100:.2f}%')
    print(f'F1分数(F1-Score): {f1 * 100:.2f}%')
    print(f'热中断预测准确率: {(confusion["TP"] + confusion["TN"]) / (tn + fp + fn + tp) * 100:.2f}%')

    print('\n时间差指标：')
    print(f'平均热中断时间差（步数）: {avg_time_diff:.1f} '
          f'(有效样本数: {len(valid_diffs)}/{len(pred_heat)})')

    print('\nROC曲线指标：')
    print(f'AUC值: {roc_auc:.4f}')

    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('热中断预测ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_vis.png')
    plt.close()

    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(actuals[-200:, 0], label='Actual Temperature')
    plt.plot(predictions[-200:, 0], label='Predicted Temperature')
    plt.axhline(heat_threshold, color='r', linestyle='--', label='热中断阈值')
    plt.title('Temperature Prediction with Heat Threshold')
    plt.legend()
    plt.savefig('predict_vis.png')
    plt.close()

    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'confusion': confusion,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'time_diff': avg_time_diff,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr
    }

# 主流程
def main():
    save_dir = ""
    model_path = os.path.join(save_dir, "PyroNet.pth")

    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    scalers_path = os.path.join(save_dir, "scalers.json")
    feature_columns_path = os.path.join(save_dir, "feature_columns.json")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = pd.read_csv('')
    scaled_data, scalers, feature_columns, train_size, val_size, test_size = preprocess_data(
        df,
        test_ratio=config["test_ratio"],
        validation_ratio=config["validation_ratio"]
    )

    with open(scalers_path, 'w') as f:
        scaler_data = {feature: {
            'min_': scaler.min_.tolist(),
            'scale_': scaler.scale_.tolist(),
            'data_min_': scaler.data_min_.tolist(),
            'data_max_': scaler.data_max_.tolist(),
            'feature_range': list(scaler.feature_range)
        } for feature, scaler in scalers.items()}
        json.dump(scaler_data, f)

    with open(feature_columns_path, 'w') as f:
        json.dump(feature_columns, f)

    X, y = create_sequences(scaled_data, feature_columns, config["seq_length"], config["pred_length"])

    dataset_size = len(X)
    train_end = train_size
    val_end = train_end + val_size

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(device)

    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).unsqueeze(-1).to(device)

    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    num_features = len(feature_columns)

    model = EnhancedCNNSeq2Seq(
        input_size=num_features,
        hidden_size=config["hidden_size"],
        cnn_channels=config["cnn_channels"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pred_length=config["pred_length"]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    print("\n开始训练模型...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        config["num_epochs"],
        device,
        patience=config["patience"],
        min_delta=config["min_delta"]
    )
    print("\n开始评估模型...")
    evaluate_model(model, test_loader, scalers, feature_columns, device)
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()