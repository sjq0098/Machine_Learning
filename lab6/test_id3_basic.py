"""
ID3决策树基础测试脚本
使用离散特征的ID3算法在西瓜数据集上进行训练和预测
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from decision_tree_ID3 import DecisionTreeID3

def main():
    print("=" * 60)
    print("ID3决策树 - 离散特征西瓜数据集测试")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    train_data = pd.read_csv('Watermelon-train1.csv')
    test_data = pd.read_csv('Watermelon-test1.csv')
    
    print(f"训练集大小: {train_data.shape}")
    print(f"测试集大小: {test_data.shape}")
    
    print("\n训练集前5行:")
    print(train_data.head())
    
    print("\n训练集类别分布:")
    print(train_data['好瓜'].value_counts())
    
    # 2. 数据预处理
    print("\n[2] 数据预处理...")
    # 分离特征和目标变量
    X_train = train_data.drop(['编号', '好瓜'], axis=1)
    y_train = train_data['好瓜']
    
    X_test = test_data.drop(['编号', '好瓜'], axis=1)
    y_test = test_data['好瓜']
    
    print(f"特征列表: {X_train.columns.tolist()}")
    print(f"训练集特征形状: {X_train.shape}")
    print(f"测试集特征形状: {X_test.shape}")
    
    # 3. 训练ID3决策树
    print("\n[3] 训练ID3决策树...")
    print("-" * 60)
    
    id3_tree = DecisionTreeID3(min_samples_split=2)
    id3_tree.fit(X_train, y_train)
    
    print("✓ 训练完成！")
    
    # 4. 打印决策树结构
    print("\n[4] 决策树结构:")
    print("-" * 60)
    id3_tree.print_tree()
    
    # 5. 预测
    print("\n[5] 进行预测...")
    print("-" * 60)
    
    train_predictions = id3_tree.predict(X_train)
    test_predictions = id3_tree.predict(X_test)
    
    # 6. 评估性能
    print("\n[6] 性能评估:")
    print("-" * 60)
    
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    print(f"训练集准确率: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    print("\n训练集分类报告:")
    print(classification_report(y_train, train_predictions, target_names=['否', '是']))
    
    print("\n测试集分类报告:")
    print(classification_report(y_test, test_predictions, target_names=['否', '是']))
    
    print("\n测试集混淆矩阵:")
    cm = confusion_matrix(y_test, test_predictions, labels=['否', '是'])
    print("真实\\预测    否    是")
    print(f"否          {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"是          {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # 7. 显示一些预测样例
    print("\n[7] 预测样例:")
    print("-" * 60)
    print("测试集前5个样本的预测结果:")
    for i in range(min(5, len(X_test))):
        actual = y_test.iloc[i]
        predicted = test_predictions[i]
        status = "✓" if actual == predicted else "✗"
        print(f"样本 {i+1}: 真实={actual:2s}, 预测={predicted:2s} {status}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()

