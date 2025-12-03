import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns

#支持中文画图：
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
class HierarchicalClustering:
    """
    层次聚类算法实现
    支持：single-linkage, complete-linkage, average-linkage
    """
    
    def __init__(self, method='single', metric='euclidean'):
        """
        初始化
        
        Args:
            method: 链接方法 ('single', 'complete', 'average')
            metric: 距离度量方法
        """
        self.method = method
        self.metric = metric
        self.Z = None  # 链接矩阵
        self.labels_ = None
        self.n_clusters_ = None
        
    def fit(self, X):
        """
        执行层次聚类
        """
        n_samples = X.shape[0]

        # 计算点对距离矩阵（原始样本之间）
        distances = pdist(X, metric=self.metric)
        distance_matrix = squareform(distances)

        # 初始化簇和相关数据结构
        clusters = {i: [i] for i in range(n_samples)}  # 簇中的样本索引
        active_nodes = list(range(n_samples))          # 当前还存在的“簇节点”

        Z = []

        # 关键修改：预留足够大的距离矩阵空间
        # 层次聚类会产生最多 2*n_samples-1 个节点，这里简单开到 2*n_samples
        max_nodes = 2 * n_samples
        D = np.full((max_nodes, max_nodes), np.inf)    # 先全部设为无穷大
        D[:n_samples, :n_samples] = distance_matrix    # 把原始样本间距离填进去

        node_id = n_samples  # 新簇的ID（从n_samples开始）

        # 逐步合并簇
        while len(active_nodes) > 1:
            # 找到距离最小的两个簇
            min_dist = np.inf
            merge_i, merge_j = -1, -1

            for idx_i, i in enumerate(active_nodes):
                for idx_j, j in enumerate(active_nodes):
                    if i < j and D[i, j] < min_dist:
                        min_dist = D[i, j]
                        merge_i = i
                        merge_j = j

            # 记录合并信息
            Z.append([
                merge_i,
                merge_j,
                float(min_dist),
                len(clusters[merge_i]) + len(clusters[merge_j])
            ])

            # 合并两个簇
            new_cluster = clusters[merge_i] + clusters[merge_j]
            clusters[node_id] = new_cluster

            # 计算新簇到其他簇的距离
            new_active_nodes = []
            for k in active_nodes:
                if k != merge_i and k != merge_j:
                    if self.method == 'single':
                        d = self._single_linkage(X, clusters[node_id], clusters[k])
                    elif self.method == 'complete':
                        d = self._complete_linkage(X, clusters[node_id], clusters[k])
                    elif self.method == 'average':
                        d = self._average_linkage(X, clusters[node_id], clusters[k])
                    else:
                        raise ValueError(f"未知链接方法: {self.method}")

                    # 更新距离矩阵（不再越界）
                    D[node_id, k] = d
                    D[k, node_id] = d
                    new_active_nodes.append(k)

            new_active_nodes.append(node_id)
            active_nodes = new_active_nodes

            # 删除已合并的簇
            del clusters[merge_i]
            del clusters[merge_j]

            node_id += 1

        self.Z = np.array(Z)
        return self

    
    def _single_linkage(self, X, cluster1, cluster2):
        """计算single-linkage距离"""
        min_dist = np.inf
        for i in cluster1:
            for j in cluster2:
                dist = np.linalg.norm(X[i] - X[j])
                min_dist = min(min_dist, dist)
        return min_dist
    
    def _complete_linkage(self, X, cluster1, cluster2):
        """计算complete-linkage距离"""
        max_dist = -np.inf
        for i in cluster1:
            for j in cluster2:
                dist = np.linalg.norm(X[i] - X[j])
                max_dist = max(max_dist, dist)
        return max_dist
    
    def _average_linkage(self, X, cluster1, cluster2):
        """计算average-linkage距离"""
        total_dist = 0
        count = 0
        for i in cluster1:
            for j in cluster2:
                dist = np.linalg.norm(X[i] - X[j])
                total_dist += dist
                count += 1
        return total_dist / count
    
    def predict(self, n_clusters=3):
        """
        根据指定的簇数进行预测
        
        Args:
            n_clusters: 簇的数量
        
        Returns:
            labels: 聚类标签
        """
        from scipy.cluster.hierarchy import fcluster
        self.labels_ = fcluster(self.Z, n_clusters, criterion='maxclust') - 1
        self.n_clusters_ = n_clusters
        return self.labels_


def load_and_prepare_wine_data():
    """加载和预处理wine数据集"""
    wine = load_wine()
    X = wine.data
    y = wine.target
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, wine.feature_names


def plot_comparison(X, y, n_clusters=3):
    """绘制三种方法的聚类结果对比"""
    
    # 初始化三个聚类对象
    methods = ['single', 'complete', 'average']
    clusterings = {}
    labels_dict = {}
    
    for method in methods:
        hc = HierarchicalClustering(method=method)
        hc.fit(X)
        clusterings[method] = hc
        labels_dict[method] = hc.predict(n_clusters=n_clusters)
    
    # 绘制聚类结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('层次聚类三种链接方法对比 (Wine数据集)', fontsize=16, fontweight='bold')
    
    # 第一行：树状图
    for idx, method in enumerate(methods):
        ax = axes[0, idx]
        dendrogram(clusterings[method].Z, ax=ax, no_labels=True)
        ax.set_title(f'{method.capitalize()}-Linkage 树状图', fontweight='bold')
        ax.set_xlabel('样本索引')
        ax.set_ylabel('距离')
    
    # 第二行：聚类结果散点图（前两个特征）
    for idx, method in enumerate(methods):
        ax = axes[1, idx]
        labels = labels_dict[method]
        
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                            s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.set_title(f'{method.capitalize()}-Linkage 聚类结果', fontweight='bold')
        ax.set_xlabel('特征1')
        ax.set_ylabel('特征2')
        plt.colorbar(scatter, ax=ax, label='簇')
    
    plt.tight_layout()
    plt.savefig('clustering_comparison_basic.png', dpi=300, bbox_inches='tight')
    print("✓ 基本对比图已保存: clustering_comparison_basic.png")
    plt.show()
    
    return clusterings, labels_dict


def evaluate_clustering(X, labels, method_name):
    """评估聚类质量"""
    silhouette = silhouette_score(X, labels)
    print(f"{method_name:20} - Silhouette Score: {silhouette:.4f}")
    return silhouette


if __name__ == "__main__":
    print("="*60)
    print("层次聚类实验 - Wine数据集")
    print("="*60)
    
    # 加载数据
    X, y_true, feature_names = load_and_prepare_wine_data()
    print(f"\n数据集形状: {X.shape}")
    print(f"真实聚类数: {len(np.unique(y_true))}")
    
    # 执行聚类和对比
    n_clusters = 3
    clusterings, labels_dict = plot_comparison(X, y_true, n_clusters=n_clusters)
    
    # 评估
    print(f"\n{'方法':<20} - 聚类质量评估 (Silhouette Score)")
    print("-"*50)
    for method in ['single', 'complete', 'average']:
        evaluate_clustering(X, labels_dict[method], method.capitalize())
    
    print("\n✓ 实验完成！")
