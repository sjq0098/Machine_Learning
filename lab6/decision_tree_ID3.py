import numpy as np
import pandas as pd
from collections import Counter

class TreeNode:
    """
    决策树节点类
    
    属性:
        feature: 当前节点用于分裂的特征名称
        label: 叶子节点的类别标签
        children: 子节点字典，key为特征值，value为子节点
    """
    def __init__(self, feature=None, label=None):
        self.feature = feature  # 用于分裂的特征
        self.label = label      # 叶子节点的标签
        self.children = {}      # 子节点字典

class DecisionTreeID3:
    """
    基于ID3算法的决策树分类器
    使用信息增益作为特征选择标准，支持离散特征
    """
    
    def __init__(self, min_samples_split=2):
        """
        初始化决策树
        
        参数:
            min_samples_split: 分裂所需的最小样本数
        """
        self.root = None
        self.min_samples_split = min_samples_split
        self.feature_names = None
    
    def calculate_entropy(self, y):
        """
        计算信息熵
        
        参数:
            y: 标签数组
        
        返回:
            entropy: 信息熵值
            
        公式: H(D) = -Σ(p_k * log2(p_k))
        其中 p_k 是第k类样本所占的比例
        """
        if len(y) == 0:
            return 0
        
        # 统计每个类别的频次
        counter = Counter(y)
        entropy = 0.0
        
        # 计算每个类别的概率并累加熵
        for count in counter.values():
            probability = count / len(y)
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def calculate_information_gain(self, X, y, feature):
        """
        计算某个特征的信息增益
        
        参数:
            X: 特征DataFrame
            y: 标签数组
            feature: 特征名称
        
        返回:
            info_gain: 信息增益值
            
        公式: Gain(D, A) = H(D) - Σ(|D_v|/|D| * H(D_v))
        """
        # 计算原始熵（整个数据集的熵）
        parent_entropy = self.calculate_entropy(y)
        
        # 获取该特征的所有唯一值
        feature_values = X[feature].unique()
        
        # 计算条件熵（按特征分裂后的加权平均熵）
        weighted_entropy = 0.0
        for value in feature_values:
            # 获取该特征值对应的样本索引
            indices = X[feature] == value
            subset_y = y[indices]
            
            # 计算权重（该分支样本数占总样本数的比例）
            weight = len(subset_y) / len(y)
            
            # 累加权重熵
            weighted_entropy += weight * self.calculate_entropy(subset_y)
        
        # 信息增益 = 原始熵 - 条件熵
        info_gain = parent_entropy - weighted_entropy
        
        return info_gain
    
    def select_best_feature(self, X, y, available_features):
        """
        选择信息增益最大的特征
        
        参数:
            X: 特征DataFrame
            y: 标签数组
            available_features: 可用特征的名称列表
        
        返回:
            best_feature: 最佳特征的名称
        """
        best_gain = -1
        best_feature = None
        
        # 遍历所有可用特征，计算信息增益
        for feature in available_features:
            gain = self.calculate_information_gain(X, y, feature)
            
            # 打印每个特征的信息增益（便于调试和理解）
            print(f"  特征 '{feature}' 的信息增益: {gain:.4f}")
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        
        print(f"  → 选择特征: {best_feature} (增益={best_gain:.4f})\n")
        return best_feature
    
    def build_tree(self, X, y, available_features, depth=0):
        """
        递归构建决策树
        
        参数:
            X: 特征DataFrame
            y: 标签数组
            available_features: 可用特征的名称列表
            depth: 当前深度（用于打印调试信息）
        
        返回:
            node: 决策树节点
        """
        indent = "  " * depth
        print(f"{indent}构建节点 [样本数={len(y)}, 类别分布={dict(Counter(y))}]")
        
        # 终止条件1：所有样本属于同一类别
        if len(np.unique(y)) == 1:
            print(f"{indent}→ 叶子节点: {y.iloc[0]}\n")
            return TreeNode(label=y.iloc[0])
        
        # 终止条件2：没有可用特征或样本数太少
        if len(available_features) == 0 or len(y) < self.min_samples_split:
            # 返回样本中最多的类别
            most_common_label = Counter(y).most_common(1)[0][0]
            print(f"{indent}→ 叶子节点（无可用特征）: {most_common_label}\n")
            return TreeNode(label=most_common_label)
        
        # 选择最佳分裂特征
        best_feature = self.select_best_feature(X, y, available_features)
        
        # 如果没有找到有效特征
        if best_feature is None:
            most_common_label = Counter(y).most_common(1)[0][0]
            print(f"{indent}→ 叶子节点（无有效特征）: {most_common_label}\n")
            return TreeNode(label=most_common_label)
        
        # 创建当前节点
        node = TreeNode(feature=best_feature)
        
        # 获取该特征的所有唯一值
        feature_values = X[best_feature].unique()
        
        # 更新可用特征列表（移除当前使用的特征）
        remaining_features = [f for f in available_features if f != best_feature]
        
        # 对每个特征值递归构建子树
        for value in feature_values:
            print(f"{indent}分支: {best_feature} = {value}")
            
            # 获取该特征值对应的子集
            indices = X[best_feature] == value
            subset_X = X[indices]
            subset_y = y[indices]
            
            # 递归构建子树
            if len(subset_y) > 0:
                node.children[value] = self.build_tree(
                    subset_X, subset_y, remaining_features, depth + 1
                )
            else:
                # 如果子集为空，创建叶子节点
                most_common_label = Counter(y).most_common(1)[0][0]
                print(f"{indent}  → 叶子节点（空子集）: {most_common_label}\n")
                node.children[value] = TreeNode(label=most_common_label)
        
        return node
    
    def fit(self, X, y):
        """
        训练决策树
        
        参数:
            X: 训练特征DataFrame
            y: 训练标签Series
        """
        # 保存特征名称
        self.feature_names = X.columns.tolist()
        
        # 重置索引以便后续操作
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        print("=" * 60)
        print("开始构建决策树...")
        print("=" * 60)
        
        # 构建决策树
        self.root = self.build_tree(X, y, self.feature_names)
        
        print("=" * 60)
        print(f"决策树训练完成！使用了 {len(self.feature_names)} 个特征，{len(y)} 个训练样本。")
        print("=" * 60)
    
    def predict_sample(self, x, node):
        """
        对单个样本进行预测
        
        参数:
            x: 单个样本的特征Series
            node: 当前节点
        
        返回:
            label: 预测的类别标签
        """
        # 如果是叶子节点，返回标签
        if node.label is not None:
            return node.label
        
        # 获取当前节点的特征值
        feature_value = x[node.feature]
        
        # 如果该特征值在训练时见过，递归预测
        if feature_value in node.children:
            return self.predict_sample(x, node.children[feature_value])
        else:
            # 如果是未见过的特征值，返回第一个子节点的预测
            # （这是一种简化处理，实际中可以用更复杂的策略）
            first_child = list(node.children.values())[0]
            return self.predict_sample(x, first_child)
    
    def predict(self, X):
        """
        对多个样本进行预测
        
        参数:
            X: 特征DataFrame
        
        返回:
            predictions: 预测结果数组
        """
        predictions = []
        for idx in range(len(X)):
            pred = self.predict_sample(X.iloc[idx], self.root)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def print_tree(self, node=None, depth=0, feature_value=None):
        """
        打印决策树结构（可视化）
        
        参数:
            node: 当前节点
            depth: 当前深度
            feature_value: 父节点的特征值
        """
        if node is None:
            node = self.root
        
        indent = "│   " * depth
        
        # 如果是叶子节点
        if node.label is not None:
            print(f"{indent}└─→ 预测: 【{node.label}】")
            return
        
        # 如果是内部节点
        if depth == 0:
            print(f"\n决策树结构:")
            print("=" * 50)
            print(f"特征: {node.feature}")
        
        for i, (value, child) in enumerate(node.children.items()):
            is_last = (i == len(node.children) - 1)
            branch = "└──" if is_last else "├──"
            
            if child.label is None:
                # 内部节点
                print(f"{indent}{branch} [{node.feature} = {value}] → 特征: {child.feature}")
                self.print_tree(child, depth + 1, value)
            else:
                # 叶子节点
                print(f"{indent}{branch} [{node.feature} = {value}]", end=" ")
                self.print_tree(child, depth + 1, value)

# ========== 扩展TreeNode以支持连续特征 ==========
class TreeNodeContinuous(TreeNode):
    """
    扩展的决策树节点类，支持连续特征分裂
    
    新增属性:
        threshold: 连续特征的分裂阈值
        is_continuous: 标记该节点是否使用连续特征分裂
    """
    def __init__(self, feature=None, label=None, threshold=None, is_continuous=False):
        super().__init__(feature, label)
        self.threshold = threshold  # 分裂阈值（用于连续特征）
        self.is_continuous = is_continuous  # 是否为连续特征分裂


# ========== 支持连续特征的决策树类 ==========
class DecisionTreeContinuous(DecisionTreeID3):
    """
    支持连续特征的决策树分类器
    继承自ID3，扩展了连续特征处理能力
    
    对于连续特征：使用二分法（阈值分裂）
    对于离散特征：使用多路分裂（ID3原始方法）
    """
    
    def __init__(self, min_samples_split=2, continuous_features=None):
        """
        初始化决策树
        
        参数:
            min_samples_split: 分裂所需的最小样本数
            continuous_features: 连续特征名称列表（如果为None则自动判断）
        """
        super().__init__(min_samples_split)
        self.continuous_features = continuous_features
        self.feature_types = {}  # 存储每个特征的类型
    
    def identify_feature_types(self, X):
        """
        自动识别特征类型（离散/连续）
        
        参数:
            X: 特征DataFrame
        
        策略:
            - 如果特征是数值型且唯一值个数 > 10，视为连续特征
            - 否则视为离散特征
        """
        if self.continuous_features is not None:
            # 用户指定了连续特征
            for col in X.columns:
                self.feature_types[col] = 'continuous' if col in self.continuous_features else 'discrete'
        else:
            # 自动判断
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > 10:
                    self.feature_types[col] = 'continuous'
                else:
                    self.feature_types[col] = 'discrete'
        
        print("特征类型识别结果:")
        for feature, ftype in self.feature_types.items():
            print(f"  {feature}: {ftype}")
    
    def find_best_threshold(self, X, y, feature):
        """
        为连续特征找到最佳分裂阈值
        
        参数:
            X: 特征DataFrame
            y: 标签Series
            feature: 特征名称
        
        返回:
            best_threshold: 最佳阈值
            best_gain: 最佳信息增益
        
        策略:
            尝试所有可能的分裂点（相邻不同值的中点）
        """
        # 获取该特征的所有唯一值并排序
        unique_values = sorted(X[feature].unique())
        
        # 如果只有一个值，无法分裂
        if len(unique_values) <= 1:
            return None, -1
        
        best_threshold = None
        best_gain = -1
        
        # 尝试所有相邻值的中点作为候选阈值
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            
            # 计算使用该阈值的信息增益
            gain = self.calculate_information_gain_continuous(X, y, feature, threshold)
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        
        return best_threshold, best_gain
    
    def calculate_information_gain_continuous(self, X, y, feature, threshold):
        """
        计算连续特征在给定阈值下的信息增益
        
        参数:
            X: 特征DataFrame
            y: 标签Series
            feature: 特征名称
            threshold: 分裂阈值
        
        返回:
            info_gain: 信息增益值
        
        公式: Gain(D, A, threshold) = H(D) - [|D_left|/|D| * H(D_left) + |D_right|/|D| * H(D_right)]
        """
        # 计算原始熵
        parent_entropy = self.calculate_entropy(y)
        
        # 根据阈值分裂数据
        left_indices = X[feature] <= threshold
        right_indices = X[feature] > threshold
        
        left_y = y[left_indices]
        right_y = y[right_indices]
        
        # 如果某一侧为空，返回0增益
        if len(left_y) == 0 or len(right_y) == 0:
            return 0
        
        # 计算加权熵
        left_weight = len(left_y) / len(y)
        right_weight = len(right_y) / len(y)
        
        weighted_entropy = (left_weight * self.calculate_entropy(left_y) + 
                           right_weight * self.calculate_entropy(right_y))
        
        # 信息增益
        info_gain = parent_entropy - weighted_entropy
        
        return info_gain
    
    def select_best_feature(self, X, y, available_features):
        """
        选择最佳特征（支持连续特征）
        
        重写父类方法以支持连续特征
        
        参数:
            X: 特征DataFrame
            y: 标签Series
            available_features: 可用特征的名称列表
        
        返回:
            best_feature: 最佳特征名称
            best_threshold: 最佳阈值（仅连续特征，离散特征返回None）
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        # 遍历所有可用特征
        for feature in available_features:
            if self.feature_types[feature] == 'discrete':
                # 离散特征：使用原始ID3方法
                gain = self.calculate_information_gain(X, y, feature)
                threshold = None
                print(f"  特征 '{feature}' (离散) 的信息增益: {gain:.4f}")
            else:
                # 连续特征：找最佳阈值
                threshold, gain = self.find_best_threshold(X, y, feature)
                if threshold is not None:
                    print(f"  特征 '{feature}' (连续, 阈值={threshold:.4f}) 的信息增益: {gain:.4f}")
            
            # 更新最佳特征
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
        
        if best_feature:
            if best_threshold is not None:
                print(f"  → 选择特征: {best_feature} (连续, 阈值={best_threshold:.4f}, 增益={best_gain:.4f})\n")
            else:
                print(f"  → 选择特征: {best_feature} (离散, 增益={best_gain:.4f})\n")
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, available_features, depth=0):
        """
        递归构建决策树（支持连续特征）
        
        重写父类方法以支持连续特征
        """
        indent = "  " * depth
        print(f"{indent}构建节点 [样本数={len(y)}, 类别分布={dict(Counter(y))}]")
        
        # 终止条件1：所有样本属于同一类别
        if len(np.unique(y)) == 1:
            print(f"{indent}→ 叶子节点: {y.iloc[0]}\n")
            return TreeNodeContinuous(label=y.iloc[0])
        
        # 终止条件2：没有可用特征或样本数太少
        if len(available_features) == 0 or len(y) < self.min_samples_split:
            most_common_label = Counter(y).most_common(1)[0][0]
            print(f"{indent}→ 叶子节点（终止条件）: {most_common_label}\n")
            return TreeNodeContinuous(label=most_common_label)
        
        # 选择最佳分裂特征
        best_feature, best_threshold = self.select_best_feature(X, y, available_features)
        
        # 如果没有找到有效特征
        if best_feature is None:
            most_common_label = Counter(y).most_common(1)[0][0]
            print(f"{indent}→ 叶子节点（无有效特征）: {most_common_label}\n")
            return TreeNodeContinuous(label=most_common_label)
        
        # 判断是连续特征还是离散特征
        is_continuous = (best_threshold is not None)
        
        # 创建当前节点
        node = TreeNodeContinuous(
            feature=best_feature, 
            threshold=best_threshold,
            is_continuous=is_continuous
        )
        
        if is_continuous:
            # 连续特征：二分法分裂
            print(f"{indent}分支: {best_feature} <= {best_threshold:.4f}")
            left_indices = X[best_feature] <= best_threshold
            left_X = X[left_indices]
            left_y = y[left_indices]
            
            if len(left_y) > 0:
                # 连续特征可以重复使用，不从available_features中移除
                node.children['left'] = self.build_tree(
                    left_X, left_y, available_features, depth + 1
                )
            else:
                most_common_label = Counter(y).most_common(1)[0][0]
                node.children['left'] = TreeNodeContinuous(label=most_common_label)
            
            print(f"{indent}分支: {best_feature} > {best_threshold:.4f}")
            right_indices = X[best_feature] > best_threshold
            right_X = X[right_indices]
            right_y = y[right_indices]
            
            if len(right_y) > 0:
                node.children['right'] = self.build_tree(
                    right_X, right_y, available_features, depth + 1
                )
            else:
                most_common_label = Counter(y).most_common(1)[0][0]
                node.children['right'] = TreeNodeContinuous(label=most_common_label)
        
        else:
            # 离散特征：多路分裂（ID3原始方法）
            feature_values = X[best_feature].unique()
            
            # 离散特征使用后需要移除
            remaining_features = [f for f in available_features if f != best_feature]
            
            for value in feature_values:
                print(f"{indent}分支: {best_feature} = {value}")
                indices = X[best_feature] == value
                subset_X = X[indices]
                subset_y = y[indices]
                
                if len(subset_y) > 0:
                    node.children[value] = self.build_tree(
                        subset_X, subset_y, remaining_features, depth + 1
                    )
                else:
                    most_common_label = Counter(y).most_common(1)[0][0]
                    node.children[value] = TreeNodeContinuous(label=most_common_label)
        
        return node
    
    def fit(self, X, y):
        """
        训练决策树（支持连续特征）
        
        重写父类方法
        """
        # 保存特征名称
        self.feature_names = X.columns.tolist()
        
        # 识别特征类型
        self.identify_feature_types(X)
        
        # 重置索引
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        print("\n" + "=" * 60)
        print("开始构建决策树（支持连续特征）...")
        print("=" * 60)
        
        # 构建决策树
        self.root = self.build_tree(X, y, self.feature_names)
        
        print("=" * 60)
        print(f"决策树训练完成！使用了 {len(self.feature_names)} 个特征，{len(y)} 个训练样本。")
        print("=" * 60)
    
    def predict_sample(self, x, node):
        """
        对单个样本进行预测（支持连续特征）
        
        重写父类方法
        """
        # 如果是叶子节点，返回标签
        if node.label is not None:
            return node.label
        
        # 获取当前节点的特征值
        feature_value = x[node.feature]
        
        if node.is_continuous:
            # 连续特征：根据阈值选择左右分支
            if feature_value <= node.threshold:
                return self.predict_sample(x, node.children['left'])
            else:
                return self.predict_sample(x, node.children['right'])
        else:
            # 离散特征：根据特征值选择分支
            if feature_value in node.children:
                return self.predict_sample(x, node.children[feature_value])
            else:
                # 未见过的特征值，返回第一个子节点的预测
                first_child = list(node.children.values())[0]
                return self.predict_sample(x, first_child)
    
    def print_tree(self, node=None, depth=0, branch_info=""):
        """
        打印决策树结构（支持连续特征）
        
        重写父类方法
        """
        if node is None:
            node = self.root
        
        indent = "│   " * depth
        
        # 如果是叶子节点
        if node.label is not None:
            print(f"{indent}└─→ 预测: 【{node.label}】")
            return
        
        # 如果是内部节点
        if depth == 0:
            print(f"\n决策树结构:")
            print("=" * 50)
            feature_type = "连续" if node.is_continuous else "离散"
            print(f"特征: {node.feature} ({feature_type})")
        
        if node.is_continuous:
            # 连续特征：显示阈值
            print(f"{indent}├── [{node.feature} <= {node.threshold:.4f}]")
            if node.children['left'].label is None:
                feature_type = "连续" if node.children['left'].is_continuous else "离散"
                print(f"{indent}│   → 特征: {node.children['left'].feature} ({feature_type})")
            self.print_tree(node.children['left'], depth + 1, "left")
            
            print(f"{indent}└── [{node.feature} > {node.threshold:.4f}]")
            if node.children['right'].label is None:
                feature_type = "连续" if node.children['right'].is_continuous else "离散"
                print(f"{indent}    → 特征: {node.children['right'].feature} ({feature_type})")
            self.print_tree(node.children['right'], depth + 1, "right")
        else:
            # 离散特征：显示所有取值
            for i, (value, child) in enumerate(node.children.items()):
                is_last = (i == len(node.children) - 1)
                branch = "└──" if is_last else "├──"
                
                print(f"{indent}{branch} [{node.feature} = {value}]", end="")
                if child.label is None:
                    feature_type = "连续" if child.is_continuous else "离散"
                    print(f" → 特征: {child.feature} ({feature_type})")
                    self.print_tree(child, depth + 1, value)
                else:
                    print(f" → 预测: 【{child.label}】")


