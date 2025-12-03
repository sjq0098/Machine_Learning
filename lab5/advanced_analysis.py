import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine, make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram
import time
import seaborn as sns
from hierarchical_clustering import HierarchicalClustering

#支持中文画图：
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class DetailedClusteringAnalysis:
    """
    详细的层次聚类对比分析
    包括：复杂度分析、准确度评估、数据集特征影响
    """
    
    def __init__(self):
        self.results = {}
        
    def generate_datasets(self):
        """生成多种特征的测试数据集"""
        datasets = {}
        
        # 1. Wine数据集
        wine = load_wine()
        scaler = StandardScaler()
        datasets['Wine'] = {
            'data': scaler.fit_transform(wine.data),
            'target': wine.target,
            'n_clusters': 3,
            'description': '真实数据集 - 葡萄酒分类',
            'shape': '高维(13特征), 178样本'
        }
        
        # 2. 高斯球形簇
        X_blob, y_blob = make_blobs(n_samples=200, centers=4, 
                                     n_features=2, random_state=42)
        datasets['Blobs'] = {
            'data': StandardScaler().fit_transform(X_blob),
            'target': y_blob,
            'n_clusters': 4,
            'description': '合成数据集 - 球形簇',
            'shape': '2D, 200样本, 4个簇'
        }
        
        # 3. 非球形簇（月牙形）
        X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)
        datasets['Moons'] = {
            'data': StandardScaler().fit_transform(X_moons),
            'target': y_moons,
            'n_clusters': 2,
            'description': '合成数据集 - 非球形簇',
            'shape': '2D, 300样本, 2个簇'
        }
        
        # 4. 高维数据
        X_high, y_high = make_blobs(n_samples=150, centers=5, 
                                     n_features=10, random_state=42)
        datasets['HighDim'] = {
            'data': StandardScaler().fit_transform(X_high),
            'target': y_high,
            'n_clusters': 5,
            'description': '高维合成数据集',
            'shape': '10D, 150样本, 5个簇'
        }
        
        # 5. 带异常点的数据
        X_blob2, y_blob2 = make_blobs(n_samples=150, centers=3, 
                                       n_features=2, random_state=42)
        # 添加异常点
        outliers = np.random.uniform(-10, 10, (10, 2))
        X_outlier = np.vstack([X_blob2, outliers])
        y_outlier = np.hstack([y_blob2, np.full(10, -1)])
        
        datasets['Outliers'] = {
            'data': StandardScaler().fit_transform(X_outlier),
            'target': y_outlier,
            'n_clusters': 3,
            'description': '含异常点的数据集',
            'shape': '2D, 160样本, 3个簇+10个异常点'
        }
        
        return datasets
    
    def measure_complexity(self, X, method_name, n_runs=3):
        """测量时间和空间复杂度"""
        times = []
        for _ in range(n_runs):
            start = time.time()
            hc = HierarchicalClustering(method=method_name)
            hc.fit(X)
            times.append(time.time() - start)

        avg_time = np.mean(times)
        space = X.nbytes  # 粗略估计

        return avg_time, space
    
    def analyze_single_dataset(self, name, dataset):
        """分析单个数据集的三种方法"""
        X = dataset['data']
        n_clusters = dataset['n_clusters']
        
        print(f"\n{'='*70}")
        print(f"数据集: {name}")
        print(f"描述: {dataset['description']}")
        print(f"形状: {dataset['shape']}")
        print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
        print(f"{'='*70}")
        
        results = {
            'dataset': name,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
        }
        
        methods = ['single', 'complete', 'average']
        
        for method in methods:
            print(f"\n{method.upper()}-LINKAGE 分析:")
            print("-" * 50)

            # 测量时间复杂度
            avg_time, _ = self.measure_complexity(X, method)

            # 执行聚类（使用自己的实现）
            hc = HierarchicalClustering(method=method)
            hc.fit(X)
            labels = hc.predict(n_clusters)
            Z = hc.Z

            # 计算评估指标
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            
            print(f"  执行时间: {avg_time*1000:.2f} ms")
            print(f"  Silhouette Score: {silhouette:.4f}  ↑越好")
            print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}  ↓越好")
            print(f"  Calinski-Harabasz Index: {calinski_harabasz:.2f}  ↑越好")
            
            results[f'{method}_time'] = avg_time
            results[f'{method}_silhouette'] = silhouette
            results[f'{method}_davies_bouldin'] = davies_bouldin
            results[f'{method}_calinski_harabasz'] = calinski_harabasz
            results[f'{method}_Z'] = Z
            results[f'{method}_labels'] = labels
        
        return results
    
    def run_comprehensive_analysis(self):
        """运行全面的对比分析"""
        print("\n" + "="*70)
        print("层次聚类详细对比分析")
        print("="*70)
        
        datasets = self.generate_datasets()
        all_results = []
        
        for name, dataset in datasets.items():
            result = self.analyze_single_dataset(name, dataset)
            all_results.append(result)
            self.results[name] = result
        
        return all_results, datasets
    
    def plot_comprehensive_results(self, all_results, datasets):
        """绘制全面的对比图"""
        
        # 提取数据用于绘图
        df_data = []
        for result in all_results:
            dataset_name = result['dataset']
            for method in ['single', 'complete', 'average']:
                df_data.append({
                    'Dataset': dataset_name,
                    'Method': method.capitalize(),
                    'Silhouette': result[f'{method}_silhouette'],
                    'Davies-Bouldin': result[f'{method}_davies_bouldin'],
                    'Calinski-Harabasz': result[f'{method}_calinski_harabasz'],
                    'Time (ms)': result[f'{method}_time'] * 1000
                })
        
        df = pd.DataFrame(df_data)
        
        # 绘制四个子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('层次聚类三种方法详细对比分析', fontsize=16, fontweight='bold')
        
        # 1. Silhouette Score对比
        ax = axes[0, 0]
        sns.barplot(data=df, x='Dataset', y='Silhouette', hue='Method', ax=ax)
        ax.set_title('Silhouette Score对比 (越高越好)', fontweight='bold')
        ax.set_ylabel('Silhouette Score')
        ax.legend(loc='best')
        
        # 2. Davies-Bouldin Index对比
        ax = axes[0, 1]
        sns.barplot(data=df, x='Dataset', y='Davies-Bouldin', hue='Method', ax=ax)
        ax.set_title('Davies-Bouldin Index对比 (越低越好)', fontweight='bold')
        ax.set_ylabel('Davies-Bouldin Index')
        ax.legend(loc='best')
        
        # 3. Calinski-Harabasz Index对比
        ax = axes[1, 0]
        sns.barplot(data=df, x='Dataset', y='Calinski-Harabasz', hue='Method', ax=ax)
        ax.set_title('Calinski-Harabasz Index对比 (越高越好)', fontweight='bold')
        ax.set_ylabel('Calinski-Harabasz Index')
        ax.legend(loc='best')
        
        # 4. 时间复杂度对比
        ax = axes[1, 1]
        sns.barplot(data=df, x='Dataset', y='Time (ms)', hue='Method', ax=ax)
        ax.set_title('执行时间对比 (ms)', fontweight='bold')
        ax.set_ylabel('时间 (毫秒)')
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ 全面对比图已保存: comprehensive_analysis.png")
        plt.show()
    
    def plot_dataset_visualizations(self, all_results, datasets):
        """绘制各数据集及其聚类结果"""
        
        # 只绘制2D数据集
        dataset_names = ['Blobs', 'Moons', 'Outliers']
        
        fig, axes = plt.subplots(len(dataset_names), 4, figsize=(16, 12))
        fig.suptitle('不同数据集上的聚类结果对比', fontsize=16, fontweight='bold')
        
        for row, name in enumerate(dataset_names):
            if name not in datasets:
                continue
            
            X = datasets[name]['data']
            
            # 第1列：原始数据
            ax = axes[row, 0]
            ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=30)
            ax.set_title(f'{name} - 原始数据')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 后3列：三种方法的聚类结果
            result = next(r for r in all_results if r['dataset'] == name)
            
            for col, method in enumerate(['single', 'complete', 'average']):
                ax = axes[row, col + 1]
                labels = result[f'{method}_labels']
                scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, 
                                    cmap='viridis', s=30, alpha=0.6, edgecolors='k', linewidth=0.5)
                ax.set_title(f'{name} - {method.capitalize()}')
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig('dataset_visualizations.png', dpi=300, bbox_inches='tight')
        print("✓ 数据集可视化已保存: dataset_visualizations.png")
        plt.show()
    


if __name__ == "__main__":
    analyzer = DetailedClusteringAnalysis()
    all_results, datasets = analyzer.run_comprehensive_analysis()
    
    # 绘制结果
    analyzer.plot_comprehensive_results(all_results, datasets)
    analyzer.plot_dataset_visualizations(all_results, datasets)
    
