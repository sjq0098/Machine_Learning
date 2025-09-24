import numpy as np
from collections import Counter

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

#留一法knn检验
def knn(vectors,labels,k):
    n=vectors.shape[0]
    correct=0
    for i in range(n):
        test_vec=vectors[i]
        test_label=labels[i]

        train_vecs=np.delete(vectors,i,axis=0)
        train_labels=np.delete(labels,i,axis=0)

        distances = np.linalg.norm(train_vecs - test_vec, axis=1)
        sorted_idx = np.argsort(distances)
        k_nearest = train_labels[sorted_idx[:k]]

        pred=Counter(k_nearest).most_common(1)[0][0]

        if pred==test_label:
            correct+=1
    
    acc=correct/n
    return acc



print('LOOCV KNN (k=9) ACC =', knn(vectors,labels,9))

#将我们的实现与weka/sklearn的knn算法对比，主参考指标为acc，副指标为归一化互信息NMI、混淆熵CEN


#实现自己手写算法的CEN与NMI计算


# 使用 sklearn 的 KNN，对比 ACC 与 NMI
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, normalized_mutual_info_score

# 划分训练/测试集（分层抽样以保持类别比例）
X_train, X_test, y_train, y_test = train_test_split(
    vectors, labels, test_size=0.2, random_state=42, stratify=labels
)

# 训练并评估
sk_clf = KNeighborsClassifier(n_neighbors=9)
sk_clf.fit(X_train, y_train)
y_pred = sk_clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
nmi = normalized_mutual_info_score(y_test, y_pred)

print('sklearn KNN (k=9) | ACC =', acc, '| NMI =', nmi)




