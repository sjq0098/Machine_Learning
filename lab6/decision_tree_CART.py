import numpy as np
import pandas as pd
from collections import Counter


# ========== 树节点类 ==========
class TreeNodeCART:
    """
    CART决策树节点类
    
    CART总是二叉树，每个节点最多有两个子节点
    
    属性:
        feature: 分裂特征名称
        threshold: 分裂阈值（连续特征）或分裂值（离散特征）
        is_continuous: 是否为连续特征
        label: 叶子节点的类别标签
        left: 左子节点（≤ threshold 或 = value）
        right: 右子节点（> threshold 或 != value）
        samples: 样本数
        gini: 基尼指数
        class_distribution: 类别分布
    """
    def __init__(self, feature=None, threshold=None, is_continuous=False,
                 label=None, samples=0, gini=0.0, class_distribution=None):
        self.feature = feature
        self.threshold = threshold
        self.is_continuous = is_continuous
        self.label = label
        self.left = None
        self.right = None
        self.samples = samples
        self.gini = gini
        self.class_distribution = class_distribution or {}


# ========== CART决策树分类器 ==========
class DecisionTreeCART:
    """
    CART决策树分类器
    
    核心特点：
    1. 使用基尼指数（Gini Index）作为分裂标准
    2. 总是生成二叉树
    3. 对离散特征也进行二分（选择最佳分裂值）
    
    公式:
        Gini(D) = 1 - Σ(p_k^2)
        Gini_split(D, A) = |D_1|/|D| * Gini(D_1) + |D_2|/|D| * Gini(D_2)
    """
    
    def __init__(self, min_samples_split=2, min_samples_leaf=1,
                 max_depth=None, continuous_features=None):
        """
        初始化CART决策树
        
        参数:
            min_samples_split: 分裂所需的最小样本数
            min_samples_leaf: 叶子节点最小样本数
            max_depth: 最大深度
            continuous_features: 连续特征名称列表
        """
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.continuous_features = continuous_features
        self.feature_names = None
        self.feature_types = {}
    
    def calculate_gini(self, y):
        """
        计算基尼指数
        
        公式: Gini(D) = 1 - Σ(p_k^2)
        
        基尼指数反映了数据集的不纯度：
        - Gini = 0 表示数据集完全纯净（所有样本属于同一类）
        - Gini 越大表示数据集越不纯
        """
        if len(y) == 0:
            return 0
        
        counter = Counter(y)
        gini = 1.0
        
        for count in counter.values():
            probability = count / len(y)
            gini -= probability ** 2
        
        return gini
    
    def calculate_gini_split(self, X, y, feature, threshold, is_continuous):
        """
        计算分裂后的基尼指数
        
        公式: Gini_split = |D_left|/|D| * Gini(D_left) + |D_right|/|D| * Gini(D_right)
        """
        n_samples = len(y)
        
        if is_continuous:
            # 连续特征：按阈值分裂
            left_indices = X[feature] <= threshold
            right_indices = ~left_indices
        else:
            # 离散特征：二分法（特征值 = threshold vs != threshold）
            left_indices = X[feature] == threshold
            right_indices = ~left_indices
        
        left_y = y[left_indices]
        right_y = y[right_indices]
        
        # 如果某一侧为空，返回无穷大（表示这个分裂无效）
        if len(left_y) == 0 or len(right_y) == 0:
            return float('inf')
        
        # 检查最小叶子节点样本数
        if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
            return float('inf')
        
        # 计算加权基尼指数
        left_weight = len(left_y) / n_samples
        right_weight = len(right_y) / n_samples
        
        gini_split = (left_weight * self.calculate_gini(left_y) +
                     right_weight * self.calculate_gini(right_y))
        
        return gini_split
    
    def find_best_split_continuous(self, X, y, feature):
        """
        为连续特征找到最佳分裂阈值
        """
        unique_values = sorted(X[feature].unique())
        
        if len(unique_values) <= 1:
            return None, float('inf')
        
        best_threshold = None
        best_gini = float('inf')
        
        # 尝试所有相邻值的中点
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            gini = self.calculate_gini_split(X, y, feature, threshold, is_continuous=True)
            
            if gini < best_gini:
                best_gini = gini
                best_threshold = threshold
        
        return best_threshold, best_gini
    
    def find_best_split_discrete(self, X, y, feature):
        """
        为离散特征找到最佳分裂值
        
        CART对离散特征也进行二分：
        选择一个值v，将数据分为 "特征=v" 和 "特征≠v" 两部分
        """
        unique_values = X[feature].unique()
        
        if len(unique_values) <= 1:
            return None, float('inf')
        
        best_value = None
        best_gini = float('inf')
        
        # 尝试每个可能的值作为分裂点
        for value in unique_values:
            gini = self.calculate_gini_split(X, y, feature, value, is_continuous=False)
            
            if gini < best_gini:
                best_gini = gini
                best_value = value
        
        return best_value, best_gini
    
    def identify_feature_types(self, X):
        """
        识别特征类型
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
        选择最佳特征和分裂点（基于基尼指数）
        
        返回: (best_feature, best_threshold, is_continuous)
        """
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        best_is_continuous = False
        
        for feature in available_features:
            if self.feature_types[feature] == 'continuous':
                threshold, gini = self.find_best_split_continuous(X, y, feature)
                is_continuous = True
            else:
                threshold, gini = self.find_best_split_discrete(X, y, feature)
                is_continuous = False
            
            if gini < best_gini:
                best_gini = gini
                best_feature = feature
                best_threshold = threshold
                best_is_continuous = is_continuous
        
        return best_feature, best_threshold, best_is_continuous
    
    def build_tree(self, X, y, available_features, depth=0):
        """
        递归构建CART决策树（二叉树）
        """
        class_dist = dict(Counter(y))
        n_samples = len(y)
        gini = self.calculate_gini(y)
        
        # 终止条件1：所有样本属于同一类别
        if len(np.unique(y)) == 1:
            label = y.iloc[0]
            return TreeNodeCART(label=label, samples=n_samples, gini=gini, class_distribution=class_dist)
        
        if self.max_depth is not None and depth >= self.max_depth:
            most_common_label = Counter(y).most_common(1)[0][0]
            return TreeNodeCART(label=most_common_label, samples=n_samples, gini=gini, class_distribution=class_dist)
        
        if len(available_features) == 0 or n_samples < self.min_samples_split:
            most_common_label = Counter(y).most_common(1)[0][0]
            return TreeNodeCART(label=most_common_label, samples=n_samples, gini=gini, class_distribution=class_dist)
        
        best_feature, best_threshold, is_continuous = self.select_best_feature(X, y, available_features)
        
        if best_feature is None:
            most_common_label = Counter(y).most_common(1)[0][0]
            return TreeNodeCART(label=most_common_label, samples=n_samples, gini=gini, class_distribution=class_dist)
        
        node = TreeNodeCART(
            feature=best_feature,
            threshold=best_threshold,
            is_continuous=is_continuous,
            samples=n_samples,
            gini=gini,
            class_distribution=class_dist
        )
        
        # CART二叉分裂（连续特征可以重复使用，离散特征也可以）
        if is_continuous:
            left_indices = X[best_feature] <= best_threshold
            left_X = X[left_indices]
            left_y = y[left_indices]
            
            if len(left_y) > 0:
                node.left = self.build_tree(left_X, left_y, available_features, depth + 1)
            right_indices = X[best_feature] > best_threshold
            right_X = X[right_indices]
            right_y = y[right_indices]
            
            if len(right_y) > 0:
                node.right = self.build_tree(right_X, right_y, available_features, depth + 1)
        else:
            # 离散特征：按值分裂（= vs ≠）
            left_indices = X[best_feature] == best_threshold
            left_X = X[left_indices]
            left_y = y[left_indices]
            
            if len(left_y) > 0:
                node.left = self.build_tree(left_X, left_y, available_features, depth + 1)
            right_indices = X[best_feature] != best_threshold
            right_X = X[right_indices]
            right_y = y[right_indices]
            
            if len(right_y) > 0:
                node.right = self.build_tree(right_X, right_y, available_features, depth + 1)
        
        return node
    
    def fit(self, X, y):
        """
        训练CART决策树
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
        if node is None or node.label is not None:
            return node.label if node else None
        
        feature_value = x[node.feature]
        
        if node.is_continuous:
            # 连续特征：比较阈值
            if feature_value <= node.threshold:
                return self.predict_sample(x, node.left)
            else:
                return self.predict_sample(x, node.right)
        else:
            # 离散特征：比较值
            if feature_value == node.threshold:
                return self.predict_sample(x, node.left)
            else:
                return self.predict_sample(x, node.right)
    
    def predict(self, X):
        """
        预测多个样本
        """
        predictions = []
        for idx in range(len(X)):
            pred = self.predict_sample(X.iloc[idx], self.root)
            predictions.append(pred)
        return np.array(predictions)
    
    def print_tree(self, node=None, depth=0, position="root"):
        """
        打印CART决策树结构（二叉树）
        """
        if node is None:
            if depth == 0:
                node = self.root
            else:
                return
        
        indent = "│   " * depth
        
        if node.label is not None:
            print(f"{indent}└─→ 预测: 【{node.label}】 (样本数={node.samples}, 基尼={node.gini:.4f})")
            return
        
        if depth == 0:
            print(f"\n决策树结构 (CART - 二叉树):")
            print("=" * 50)
            feature_type = "连续" if node.is_continuous else "离散"
            print(f"根节点: {node.feature} ({feature_type}), 基尼={node.gini:.4f}")
        
        # 左子树
        if node.is_continuous:
            print(f"{indent}├── [左] {node.feature} <= {node.threshold:.4f}")
        else:
            print(f"{indent}├── [左] {node.feature} = {node.threshold}")
        
        if node.left and node.left.label is None:
            ft = "连续" if node.left.is_continuous else "离散"
            print(f"{indent}│   → 特征: {node.left.feature} ({ft}), 基尼={node.left.gini:.4f}")
        
        self.print_tree(node.left, depth + 1, "left")
        
        # 右子树
        if node.is_continuous:
            print(f"{indent}└── [右] {node.feature} > {node.threshold:.4f}")
        else:
            print(f"{indent}└── [右] {node.feature} ≠ {node.threshold}")
        
        if node.right and node.right.label is None:
            ft = "连续" if node.right.is_continuous else "离散"
            print(f"{indent}    → 特征: {node.right.feature} ({ft}), 基尼={node.right.gini:.4f}")
        
        self.print_tree(node.right, depth + 1, "right")


# ========== 测试代码 ==========
if __name__ == "__main__":
    # 加载数据
    train_data = pd.read_csv('Watermelon-train2.csv')
    test_data = pd.read_csv('Watermelon-test2.csv')
    
    X_train = train_data.drop(['编号', '好瓜'], axis=1)
    y_train = train_data['好瓜']
    
    X_test = test_data.drop(['编号', '好瓜'], axis=1)
    y_test = test_data['好瓜']
    
    # 训练CART决策树
    tree = DecisionTreeCART(
        min_samples_split=2,
        min_samples_leaf=1,
        max_depth=5,
        continuous_features=['密度']
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

