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
        
        wine = load_wine()
        scaler = StandardScaler()
        datasets['Wine'] = {
            'data': scaler.fit_transform(wine.data),
            'target': wine.target,
            'n_clusters': 3,
            'description': '真实数据集 - 葡萄酒分类',
            'shape': '高维(13特征), 178样本'
        }
        
        X_blob, y_blob = make_blobs(n_samples=200, centers=4, 
                                     n_features=2, random_state=42)
        datasets['Blobs'] = {
            'data': StandardScaler().fit_transform(X_blob),
            'target': y_blob,
            'n_clusters': 4,
            'description': '合成数据集 - 球形簇',
            'shape': '2D, 200样本, 4个簇'
        }
        
        X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)
        datasets['Moons'] = {
            'data': StandardScaler().fit_transform(X_moons),
            'target': y_moons,
            'n_clusters': 2,
            'description': '合成数据集 - 非球形簇',
            'shape': '2D, 300样本, 2个簇'
        }
        
        X_high, y_high = make_blobs(n_samples=150, centers=5, 
                                     n_features=10, random_state=42)
        datasets['HighDim'] = {
            'data': StandardScaler().fit_transform(X_high),
            'target': y_high,
            'n_clusters': 5,
            'description': '高维合成数据集',
            'shape': '10D, 150样本, 5个簇'
        }
        
        X_blob2, y_blob2 = make_blobs(n_samples=150, centers=3, 
                                       n_features=2, random_state=42)
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
        space = X.nbytes

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

            avg_time, _ = self.measure_complexity(X, method)

            hc = HierarchicalClustering(method=method)
            hc.fit(X)
            labels = hc.predict(n_clusters)
            Z = hc.Z

            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            
            print(f"  执行时间: {avg_time*1000:.2f} ms")
            print(f"  Silhouette Score: {silhouette:.4f}")
            print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
            print(f"  Calinski-Harabasz Index: {calinski_harabasz:.2f}")
            
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
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('层次聚类三种方法详细对比分析', fontsize=16, fontweight='bold')
        
        ax = axes[0, 0]
        sns.barplot(data=df, x='Dataset', y='Silhouette', hue='Method', ax=ax)
        ax.set_title('Silhouette Score对比', fontweight='bold')
        ax.set_ylabel('Silhouette Score')
        ax.legend(loc='best')
        
        ax = axes[0, 1]
        sns.barplot(data=df, x='Dataset', y='Davies-Bouldin', hue='Method', ax=ax)
        ax.set_title('Davies-Bouldin Index对比', fontweight='bold')
        ax.set_ylabel('Davies-Bouldin Index')
        ax.legend(loc='best')
        
        ax = axes[1, 0]
        sns.barplot(data=df, x='Dataset', y='Calinski-Harabasz', hue='Method', ax=ax)
        ax.set_title('Calinski-Harabasz Index对比', fontweight='bold')
        ax.set_ylabel('Calinski-Harabasz Index')
        ax.legend(loc='best')
        
        ax = axes[1, 1]
        sns.barplot(data=df, x='Dataset', y='Time (ms)', hue='Method', ax=ax)
        ax.set_title('执行时间对比', fontweight='bold')
        ax.set_ylabel('时间 (毫秒)')
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        print("\n全面对比图已保存: comprehensive_analysis.png")
        plt.show()
    
    def plot_dataset_visualizations(self, all_results, datasets):
        """绘制各数据集及其聚类结果"""
        
        dataset_names = ['Blobs', 'Moons', 'Outliers']
        
        fig, axes = plt.subplots(len(dataset_names), 4, figsize=(16, 12))
        fig.suptitle('不同数据集上的聚类结果对比', fontsize=16, fontweight='bold')
        
        for row, name in enumerate(dataset_names):
            if name not in datasets:
                continue
            
            X = datasets[name]['data']
            
            ax = axes[row, 0]
            ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=30)
            ax.set_title(f'{name} - 原始数据')
            ax.set_xticks([])
            ax.set_yticks([])
            
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
        print("数据集可视化已保存: dataset_visualizations.png")
        plt.show()
    
    def plot_cluster_distributions(self, all_results, datasets, method='average'):
        """查看每个聚类的分布情况"""
        for name, dataset in datasets.items():
            result = next(r for r in all_results if r['dataset'] == name)
            labels = result[f'{method}_labels']
            X = dataset['data']
            
            unique, counts = np.unique(labels, return_counts=True)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.barplot(x=unique, y=counts, ax=axes[0], palette='viridis')
            axes[0].set_title(f'{name} - {method.capitalize()} 聚类大小')
            axes[0].set_xlabel('Cluster')
            axes[0].set_ylabel('Count')
            
            max_features = min(4, X.shape[1])
            feature_cols = [f'feat_{i}' for i in range(max_features)]
            df = pd.DataFrame(X[:, :max_features], columns=feature_cols)
            df['cluster'] = labels
            df_melt = df.melt(id_vars='cluster', var_name='feature', value_name='value')
            sns.boxplot(data=df_melt, x='feature', y='value', hue='cluster', ax=axes[1])
            axes[1].set_title(f'{name} - {method.capitalize()} 聚类特征分布')
            axes[1].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            outfile = f'{name}_{method}_cluster_distribution.png'
            plt.savefig(outfile, dpi=300, bbox_inches='tight')
            print(f"聚类分布图已保存: {outfile}")
            plt.show()
    


if __name__ == "__main__":
    analyzer = DetailedClusteringAnalysis()
    all_results, datasets = analyzer.run_comprehensive_analysis()
    
    analyzer.plot_comprehensive_results(all_results, datasets)
    analyzer.plot_dataset_visualizations(all_results, datasets)
    analyzer.plot_cluster_distributions(all_results, datasets, method='average')
    
