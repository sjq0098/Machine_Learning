import numpy as np
from collections import Counter,defaultdict

from sklearn.metrics import  normalized_mutual_info_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")


#读取数据semeion.data，并将数据变成数字

def load_data(file_path):
    # 文件中是类似 0.0000/1.0000 的浮点数，用 numpy 读取为浮点
    data = np.loadtxt(file_path, dtype=np.float32)
    return data

data = load_data('semeion.data')

#对于每一行数据，将前面的向量与后续十位标签分开

def preprocessing(data):
    vectors = data[:, :-10]
    labels = data[:, -10:]
    labels = np.argmax(labels, axis=1)
    return vectors, labels

vectors, labels = preprocessing(data)

#实现自己手写算法的CEN与NMI计算
def cen(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / len(y_true)


def nmi(y_true,y_pred):
    return normalized_mutual_info_score(y_true, y_pred)

#留一法knn检验
def knn(vectors,labels,k):
    n=vectors.shape[0]
    correct=0
    preds=[]
    for i in range(n):
        test_vec=vectors[i]
        test_label=labels[i]

        train_vecs=np.delete(vectors,i,axis=0)
        train_labels=np.delete(labels,i,axis=0)

        distances = np.linalg.norm(train_vecs - test_vec, axis=1)
        sorted_idx = np.argsort(distances)
        k_nearest = train_labels[sorted_idx[:k]]
        weights = 1 / (distances[sorted_idx[:k]] + 1e-12)  # 防止除零

        # 对每个类别累计权重
        class_weights = defaultdict(float)
        for lbl, w in zip(k_nearest, weights):
            class_weights[lbl] += w

        # 选权重最大的类别
        pred = max(class_weights.items(), key=lambda x: x[1])[0]

        #pred=Counter(k_nearest).most_common(1)[0][0]
        preds.append(pred)
        if pred==test_label:
            correct+=1
    
    acc=correct/n
    nmi_val=nmi(labels,preds)
    cen_val=cen(labels,preds)
    
    return acc,nmi_val,cen_val

for k in [3,5,7,9,11,13]:
    acc,nmi_val,cen_val=knn(vectors,labels,k)
    print("KNN (k=",k,") ACC =", acc, "| NMI =", nmi_val, "| CEN =", cen_val)