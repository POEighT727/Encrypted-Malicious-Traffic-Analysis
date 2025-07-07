#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import warnings
import argparse
from glob import glob
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    xgb_installed = True
except ImportError:
    xgb_installed = False

# 配置路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')  # 数据集目录
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'auto_feature_model.pkl')


def is_numeric_column(series):
    """检查列是否为数值类型"""
    try:
        pd.to_numeric(series)
        return True
    except:
        return False


def load_auto_features(data_dir):
    """自动从CSV提取特征（最后一列为标签）"""
    csv_files = glob(os.path.join(data_dir, '**/*.csv'), recursive=True)
    if not csv_files:
        raise ValueError("未找到CSV文件，请检查路径")

    # 读取第一个CSV获取特征列名
    sample_df = pd.read_csv(csv_files[0])
    all_cols = sample_df.columns.tolist()
    feature_cols = all_cols[:-1]  # 最后一列除外
    label_col = all_cols[-1]  # 最后一列作为标签

    print(f"原始特征列数: {len(feature_cols)}")

    # 合并所有数据
    X_list, y_list = [], []
    for file in csv_files:
        df = pd.read_csv(file)
        # 统一列名（处理大小写/空格问题）
        df.columns = df.columns.str.strip().str.lower()
        current_feature_cols = [col.strip().lower() for col in feature_cols]
        current_label_col = label_col.strip().lower()

        # 检查列是否匹配
        missing_cols = set(current_feature_cols) - set(df.columns)
        if missing_cols:
            print(f"警告：文件 {file} 缺少列 {missing_cols}，已跳过")
            continue

        # 处理数据
        features = df[current_feature_cols].copy()
        labels = df[current_label_col].copy()

        # 清理数据 - 只对数值列处理
        numeric_cols = [col for col in features.columns if is_numeric_column(features[col])]
        non_numeric_cols = [col for col in features.columns if col not in numeric_cols]

        for col in numeric_cols:
            features[col].replace([np.inf, -np.inf], np.nan, inplace=True)
            features[col].fillna(features[col].median(), inplace=True)

        # 只保留数值型特征
        features = features[numeric_cols]

        X_list.append(features)
        y_list.append(labels)

    X = pd.concat(X_list, axis=0)
    y = pd.concat(y_list, axis=0)

    # 标签编码（自动识别良性/恶意）
    le = LabelEncoder()
    y = le.fit_transform(y)
    if 'BENIGN' in le.classes_ or 'normal' in le.classes_:  # 良性标签
        y = np.where(y == le.transform(['BENIGN'])[0], 1, 0)
    else:
        y = np.where(y == le.transform([le.classes_[0]])[0], 1, 0)

    # 特征标准化
    if len(numeric_cols) > 0:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        # 保存scaler
        joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.pkl'))
        print(f"归一化scaler已保存至: {os.path.join(BASE_DIR, 'scaler.pkl')}")

    print(f"最终使用的数值型特征数量: {len(numeric_cols)}")
    return X, y, numeric_cols


def evaluate_model(model, x_test, y_test, feature_cols):
    """模型评估与可视化"""
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = model.predict(x_test)

    # 计算指标
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # 打印报告
    print("\n" + "=" * 50)
    print("模型评估结果")
    print("=" * 50)
    print(f"特征数量: {x_test.shape[1]}")
    print(f"准确率: {(tp + tn) / (tp + tn + fp + fn):.4f}")
    print(f"精确率: {(tp / (tp + fp)):.4f}")
    print(f"召回率: {(tp / (tp + fn)):.4f}")
    print(f"F1 Score: {(2 * tp / (2 * tp + fp + fn)):.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print("\n混淆矩阵:")
    print(cm)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 可视化
    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    plt.subplot(122)
    # 设置中文字体（如SimHei），防止中文乱码
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # 只显示最重要的前20个特征
        topk = 20
        indices = np.argsort(importances)[-topk:]
        plt.barh(range(topk),
                 importances[indices],
                 tick_label=np.array(feature_cols)[indices])
        plt.xlabel('特征重要性')
        plt.title('特征重要性（Top 20）')
        plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()


def train_model(X, y, feature_cols, use_xgb=False, use_svm=False):
    """训练模型（支持RF、XGBoost、SVM）"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    if use_svm:
        # SVM不支持NaN，需填充
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        model = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
        print("\n使用SVM训练模型中...")
    elif use_xgb:
        if not xgb_installed:
            print("未安装xgboost库，请先安装：pip install xgboost")
            return
        model = XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )
        print("\n使用XGBoost训练模型中...")
    else:
        model = RandomForestClassifier(
            n_estimators=200,  
            max_depth=100,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
            verbose=2
        )
        print("\n使用随机森林训练模型中...")
    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test, feature_cols)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\n模型已保存至: {MODEL_SAVE_PATH}")
    # 保存特征列
    with open(os.path.join(BASE_DIR, 'feature_cols.txt'), 'w', encoding='utf-8') as f:
        for col in feature_cols:
            f.write(col + '\n')
    print(f"特征列已保存至: {os.path.join(BASE_DIR, 'feature_cols.txt')}")
    return feature_cols


def predict_new(file_path, feature_cols):
    """预测新样本"""
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError("请先训练模型")
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("请先训练模型（缺少scaler.pkl）")
    scaler = joblib.load(scaler_path)

    model = joblib.load(MODEL_SAVE_PATH)
    df = pd.read_csv(file_path)

    # 统一列名格式
    df.columns = df.columns.str.strip().str.lower()
    available_cols = [col for col in feature_cols if col in df.columns]

    if len(available_cols) != len(feature_cols):
        print(f"警告：缺少特征列 {set(feature_cols) - set(available_cols)}")
        print(f"将使用可用特征: {available_cols}")

    # 只保留数值型特征
    numeric_cols = [col for col in df.columns if is_numeric_column(df[col])]
    df = df[numeric_cols]

    # 构造和训练时完全一致的特征DataFrame，缺失的特征补0
    features = pd.DataFrame()
    for col in feature_cols:
        if col in df.columns:
            features[col] = df[col]
        else:
            features[col] = 0  # 或 np.nan

    # 填充缺失值
    features = features.fillna(features.median())

    # 归一化
    features = scaler.transform(features)

    # 预测每一行
    if len(available_cols) > 0:
        probs = model.predict_proba(features)[:, 1]  # 这是Benign的概率
        preds = model.predict(features)
        for idx, (pred, prob) in enumerate(zip(preds, probs)):
            malware_prob = 1 - prob  # 恶意概率
            print(f"第{idx+1}行: 预测结果: {'Benign' if pred == 1 else 'Malware'}  恶意概率: {malware_prob:.4f}")
    else:
        print("错误：没有可用的匹配特征列")


def main():
    parser = argparse.ArgumentParser(description="加密恶意流量检测系统")
    parser.add_argument('--train', action='store_true', help='使用随机森林训练')
    parser.add_argument('--predict', type=str, help='预测单个CSV文件')
    parser.add_argument('--xgb', action='store_true', help='使用XGBoost训练')
    parser.add_argument('--svm', action='store_true', help='使用SVM训练')
    args = parser.parse_args()

    if args.train:
        print("正在自动提取特征...")
        X, y, feature_cols = load_auto_features(DATASET_DIR)
        print(f"数据集加载完成 - 样本: {len(X)}, 特征: {len(feature_cols)}")
        train_model(X, y, feature_cols, use_xgb=args.xgb, use_svm=args.svm)
    elif args.predict:
        # 直接读取训练时保存的特征列
        model_path = MODEL_SAVE_PATH
        feature_path = os.path.join(BASE_DIR, 'feature_cols.txt')
        if not os.path.exists(model_path):
            print(f"错误：未找到模型文件 {model_path}，请先运行 --train 生成模型")
            return
        if not os.path.exists(feature_path):
            print(f"错误：未找到特征列文件 {feature_path}，请先运行 --train 生成模型")
            return
        with open(feature_path, 'r', encoding='utf-8') as f:
            feature_cols = [line.strip() for line in f if line.strip()]
        predict_new(args.predict, feature_cols)
    else:
        parser.print_help()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()