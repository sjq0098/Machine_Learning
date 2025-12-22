import numpy as np
import pandas as pd
from collections import Counter


class TreeNodeC45:
    """
    C4.5决策树节点类
    
    属性:
        feature: 分裂特征名称
        threshold: 连续特征的分裂阈值
        is_continuous: 是否为连续特征
        label: 叶子节点的类别标签
        children: 子节点字典
        samples: 该节点的样本数
        class_distribution: 类别分布
    """
    def __init__(self, feature=None, threshold=None, is_continuous=False, 
                 label=None, samples=0, class_distribution=None):
        self.feature = feature
        self.threshold = threshold
        self.is_continuous = is_continuous
        self.label = label
        self.children = {}
        self.samples = samples
        self.class_distribution = class_distribution or {}


class DecisionTreeC45:
    """
    C4.5决策树分类器
    
    核心改进：使用信息增益率（Gain Ratio）进行特征选择
    
    公式:
        Gain_Ratio(D, A) = Gain(D, A) / IV(A)
        其中 IV(A) = -Σ(|D_v|/|D| * log2(|D_v|/|D|)) 是固有值（Intrinsic Value）
    """
    
    def __init__(self, min_samples_split=2, min_samples_leaf=1, 
                 max_depth=None, continuous_features=None):
        """
        初始化C4.5决策树
        
        参数:
            min_samples_split: 分裂所需的最小样本数
            min_samples_leaf: 叶子节点最小样本数
            max_depth: 最大深度（None表示不限制）
            continuous_features: 连续特征名称列表
        """
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.continuous_features = continuous_features
        self.feature_names = None
        self.feature_types = {}
    
    def calculate_entropy(self, y):
        """
        计算信息熵
        
        公式: H(D) = -Σ(p_k * log2(p_k))
        """
        if len(y) == 0:
            return 0
        
        counter = Counter(y)
        entropy = 0.0
        
        for count in counter.values():
            probability = count / len(y)
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def calculate_intrinsic_value(self, X, feature, threshold=None):
        """
        计算固有值（Intrinsic Value）
        
        固有值用于惩罚取值较多的特征
        
        公式: IV(A) = -Σ(|D_v|/|D| * log2(|D_v|/|D|))
        """
        n_samples = len(X)
        if n_samples == 0:
            return 0
        
        iv = 0.0
        
        if threshold is not None:
            # 连续特征：二分裂
            n_left = np.sum(X[feature] <= threshold)
            n_right = n_samples - n_left
            
            for n in [n_left, n_right]:
                if n > 0:
                    p = n / n_samples
                    iv -= p * np.log2(p)
        else:
            # 离散特征：多路分裂
            feature_values = X[feature].unique()
            
            for value in feature_values:
                n_v = np.sum(X[feature] == value)
                if n_v > 0:
                    p = n_v / n_samples
                    iv -= p * np.log2(p)
        
        return iv
    
    def calculate_information_gain(self, X, y, feature, threshold=None):
        """
        计算信息增益
        
        公式: Gain(D, A) = H(D) - H(D|A)
        """
        parent_entropy = self.calculate_entropy(y)
        n_samples = len(y)
        
        if threshold is not None:
            # 连续特征
            left_indices = X[feature] <= threshold
            right_indices = ~left_indices
            
            left_y = y[left_indices]
            right_y = y[right_indices]
            
            if len(left_y) == 0 or len(right_y) == 0:
                return 0
            
            weighted_entropy = (len(left_y) / n_samples * self.calculate_entropy(left_y) +
                              len(right_y) / n_samples * self.calculate_entropy(right_y))
        else:
            # 离散特征
            feature_values = X[feature].unique()
            weighted_entropy = 0.0
            
            for value in feature_values:
                indices = X[feature] == value
                subset_y = y[indices]
                
                weight = len(subset_y) / n_samples
                weighted_entropy += weight * self.calculate_entropy(subset_y)
        
        return parent_entropy - weighted_entropy
    
    def calculate_gain_ratio(self, X, y, feature, threshold=None):
        """
        计算信息增益率（C4.5的核心）
        
        公式: Gain_Ratio(D, A) = Gain(D, A) / IV(A)
        
        信息增益率通过除以固有值来惩罚取值较多的特征
        """
        gain = self.calculate_information_gain(X, y, feature, threshold)
        iv = self.calculate_intrinsic_value(X, feature, threshold)
        
        # 避免除以0
        if iv == 0:
            return 0
        
        return gain / iv
    
    def find_best_threshold(self, X, y, feature):
        """
        为连续特征找到最佳分裂阈值
        """
        unique_values = sorted(X[feature].unique())
        
        if len(unique_values) <= 1:
            return None, -1
        
        best_threshold = None
        best_gain_ratio = -1
        
        # 尝试所有相邻值的中点
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            
            # 检查分裂后每边至少有min_samples_leaf个样本
            n_left = np.sum(X[feature] <= threshold)
            n_right = len(X) - n_left
            
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            
            gain_ratio = self.calculate_gain_ratio(X, y, feature, threshold)
            
            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_threshold = threshold
        
        return best_threshold, best_gain_ratio
    
    def identify_feature_types(self, X):
        """
        识别特征类型（离散/连续）
        """
        if self.continuous_features is not None:
            for col in X.columns:
                self.feature_types[col] = 'continuous' if col in self.continuous_features else 'discrete'
        else:
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > 10:
                    self.feature_types[col] = 'continuous'
                else:
                    self.feature_types[col] = 'discrete'
    
    def select_best_feature(self, X, y, available_features):
        """
        选择最佳特征（基于信息增益率）
        """
        best_gain_ratio = -1
        best_feature = None
        best_threshold = None
        
        for feature in available_features:
            if self.feature_types[feature] == 'continuous':
                threshold, gain_ratio = self.find_best_threshold(X, y, feature)
            else:
                gain_ratio = self.calculate_gain_ratio(X, y, feature, None)
                threshold = None
            
            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_feature = feature
                best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, available_features, depth=0):
        """
        递归构建决策树
        """
        class_dist = dict(Counter(y))
        n_samples = len(y)
        
        # 终止条件1：所有样本属于同一类别
        if len(np.unique(y)) == 1:
            label = y.iloc[0]
            return TreeNodeC45(label=label, samples=n_samples, class_distribution=class_dist)
        
        if self.max_depth is not None and depth >= self.max_depth:
            most_common_label = Counter(y).most_common(1)[0][0]
            return TreeNodeC45(label=most_common_label, samples=n_samples, class_distribution=class_dist)
        
        if len(available_features) == 0 or n_samples < self.min_samples_split:
            most_common_label = Counter(y).most_common(1)[0][0]
            return TreeNodeC45(label=most_common_label, samples=n_samples, class_distribution=class_dist)
        
        best_feature, best_threshold = self.select_best_feature(X, y, available_features)
        
        if best_feature is None:
            most_common_label = Counter(y).most_common(1)[0][0]
            return TreeNodeC45(label=most_common_label, samples=n_samples, class_distribution=class_dist)
        
        is_continuous = (best_threshold is not None)
        node = TreeNodeC45(
            feature=best_feature,
            threshold=best_threshold,
            is_continuous=is_continuous,
            samples=n_samples,
            class_distribution=class_dist
        )
        
        if is_continuous:
            left_indices = X[best_feature] <= best_threshold
            left_X = X[left_indices]
            left_y = y[left_indices]
            
            if len(left_y) >= self.min_samples_leaf:
                node.children['left'] = self.build_tree(left_X, left_y, available_features, depth + 1)
            else:
                most_common = Counter(y).most_common(1)[0][0]
                node.children['left'] = TreeNodeC45(label=most_common, samples=len(left_y))
            right_indices = X[best_feature] > best_threshold
            right_X = X[right_indices]
            right_y = y[right_indices]
            
            if len(right_y) >= self.min_samples_leaf:
                node.children['right'] = self.build_tree(right_X, right_y, available_features, depth + 1)
            else:
                most_common = Counter(y).most_common(1)[0][0]
                node.children['right'] = TreeNodeC45(label=most_common, samples=len(right_y))
        else:
            feature_values = X[best_feature].unique()
            remaining_features = [f for f in available_features if f != best_feature]
            
            for value in feature_values:
                indices = X[best_feature] == value
                subset_X = X[indices]
                subset_y = y[indices]
                
                if len(subset_y) >= self.min_samples_leaf:
                    node.children[value] = self.build_tree(subset_X, subset_y, remaining_features, depth + 1)
                else:
                    most_common = Counter(y).most_common(1)[0][0]
                    node.children[value] = TreeNodeC45(label=most_common, samples=len(subset_y))
        
        return node
    
    def fit(self, X, y):
        """
        训练C4.5决策树
        """
        self.feature_names = X.columns.tolist()
        self.identify_feature_types(X)
        
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        self.root = self.build_tree(X, y, self.feature_names)
    
    def predict_sample(self, x, node):
        """
        预测单个样本
        """
        if node.label is not None:
            return node.label
        
        feature_value = x[node.feature]
        
        if node.is_continuous:
            if feature_value <= node.threshold:
                return self.predict_sample(x, node.children['left'])
            else:
                return self.predict_sample(x, node.children['right'])
        else:
            if feature_value in node.children:
                return self.predict_sample(x, node.children[feature_value])
            else:
                # 未见过的值，返回最常见类别
                if node.class_distribution:
                    return max(node.class_distribution, key=node.class_distribution.get)
                return list(node.children.values())[0].label
    
    def predict(self, X):
        """
        预测多个样本
        """
        predictions = []
        for idx in range(len(X)):
            pred = self.predict_sample(X.iloc[idx], self.root)
            predictions.append(pred)
        return np.array(predictions)
    
    def print_tree(self, node=None, depth=0):
        """
        打印决策树结构
        """
        if node is None:
            node = self.root
        
        indent = "│   " * depth
        
        if node.label is not None:
            print(f"{indent}└─→ 预测: 【{node.label}】 (样本数={node.samples})")
            return
        
        if depth == 0:
            print(f"\n决策树结构 (C4.5):")
            print("=" * 50)
            feature_type = "连续" if node.is_continuous else "离散"
            print(f"特征: {node.feature} ({feature_type})")
        
        if node.is_continuous:
            print(f"{indent}├── [{node.feature} <= {node.threshold:.4f}]")
            if node.children['left'].label is None:
                ft = "连续" if node.children['left'].is_continuous else "离散"
                print(f"{indent}│   → 特征: {node.children['left'].feature} ({ft})")
            self.print_tree(node.children['left'], depth + 1)
            
            print(f"{indent}└── [{node.feature} > {node.threshold:.4f}]")
            if node.children['right'].label is None:
                ft = "连续" if node.children['right'].is_continuous else "离散"
                print(f"{indent}    → 特征: {node.children['right'].feature} ({ft})")
            self.print_tree(node.children['right'], depth + 1)
        else:
            for i, (value, child) in enumerate(node.children.items()):
                is_last = (i == len(node.children) - 1)
                branch = "└──" if is_last else "├──"
                
                print(f"{indent}{branch} [{node.feature} = {value}]", end="")
                if child.label is None:
                    ft = "连续" if child.is_continuous else "离散"
                    print(f" → 特征: {child.feature} ({ft})")
                    self.print_tree(child, depth + 1)
                else:
                    print(f" → 预测: 【{child.label}】 (样本数={child.samples})")


# ========== 测试代码 ==========
if __name__ == "__main__":
    # 加载数据
    train_data = pd.read_csv('Watermelon-train2.csv')
    test_data = pd.read_csv('Watermelon-test2.csv')
    
    X_train = train_data.drop(['编号', '好瓜'], axis=1)
    y_train = train_data['好瓜']
    
    X_test = test_data.drop(['编号', '好瓜'], axis=1)
    y_test = test_data['好瓜']
    
    # 训练C4.5决策树
    tree = DecisionTreeC45(
        min_samples_split=2,
        min_samples_leaf=1,
        max_depth=None,
        continuous_features=['密度']  # 指定连续特征
    )
    tree.fit(X_train, y_train)
    
    # 可视化
    tree.print_tree()
    
    # 评估
    train_pred = tree.predict(X_train)
    train_acc = np.mean(train_pred == y_train.values)
    print(f"\n训练集准确率: {train_acc:.2%}")
    
    test_pred = tree.predict(X_test)
    test_acc = np.mean(test_pred == y_test.values)
    print(f"测试集准确率: {test_acc:.2%}")

