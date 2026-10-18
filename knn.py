import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
def knn_from_scratch(X, y, Xt, k=5, weights='uniform'):
    """
    手写KNN预测函数
    :param X: 训练集特征 (n_train, n_features)
    :param y: 训练集标签 (n_train,)
    :param Xt: 测试集特征 (n_test, n_features)
    :param k: 近邻数量
    :param weights: 'uniform' 均等投票; 'distance' 距离加权投票
    :return: preds: 预测标签数组 (n_test,)
    """
    n_train = X.shape[0]
    n_test = Xt.shape[0]
    
    D = np.sqrt(np.sum((Xt[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1))
    preds = np.empty(n_test, dtype=int)

    for i in range(n_test):
        dists = D[i]
        
        topk_idx = np.argsort(dists)[:k]
        neighbor_dists = dists[topk_idx]
        neighbor_labels = y[topk_idx]

        if weights == 'uniform':
            
            labels, counts = np.unique(neighbor_labels, return_counts=True)
            pred = labels[np.argmax(counts)]
        elif weights == 'distance':
            
            w = 1.0 / (neighbor_dists + 1e-12)
            class_weights = {}
            for lab, wi in zip(neighbor_labels, w):
                class_weights[lab] = class_weights.get(lab, 0) + wi
            pred = max(class_weights, key=class_weights.get)
        else:
            raise ValueError("weights 只能是 'uniform' 或 'distance'")
        preds[i] = pred
    return preds

def shuffle_split(X, y, test_size=0.3, seed=0):
    """先打乱样本顺序，再切分，返回 (X_train, y_train, X_test, y_test)"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    X = np.asarray(X)[idx]
    y = np.asarray(y)[idx]
    split_point = int(len(y) * (1 - test_size))
    X_train = X[:split_point]
    y_train = y[:split_point]
    X_test = X[split_point:]
    y_test = y[split_point:]
    return X_train, y_train, X_test, y_test

def scan_k(X, y, ks=range(1, 16), test_size=0.3, seed=0):
    """扫描不同k，返回ks列表、训练准确率列表、测试准确率列表"""
    Xtr, ytr, Xte, yte = shuffle_split(X, y, test_size, seed)
    train_acc = []
    test_acc = []
    for k in ks:
        y_train_pred = knn_from_scratch(Xtr, ytr, Xtr, k=k)
        y_test_pred = knn_from_scratch(Xtr, ytr, Xte, k=k)
        tr_acc = accuracy_score(ytr, y_train_pred)
        te_acc = accuracy_score(yte, y_test_pred)
        train_acc.append(tr_acc)
        test_acc.append(te_acc)
    return list(ks), train_acc, test_acc

def plot_knn_boundary(Xtr, ytr, X, y, k, title, save_path=None):
    """绘制KNN二维决策边界，仅支持二维特征输入"""
    xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
    ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200),
                         np.linspace(ymin, ymax, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # 预测网格点
    Z = knn_from_scratch(Xtr, ytr, grid, k=k).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", s=15, edgecolors="k", linewidths=0.3)
    plt.title(title)
    plt.xlabel("feature 0")
    plt.ylabel("feature 1")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    
    data = load_iris()
    X_full = data.data
    y_full = data.target
    X_2d = X_full[:, :2]  

    
    ks, train_acc_list, test_acc_list = scan_k(X_full, y_full, ks=range(1,16), seed=42)
    plt.figure(figsize=(7,4))
    plt.plot(ks, train_acc_list, marker='o', label="Train Acc")
    plt.plot(ks, test_acc_list, marker='s', label="Test Acc")
    plt.xlabel("k value of KNN")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs k")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("figs/knn_k_curve.png", dpi=300)
    plt.show()
    print("=== k扫描结果 ===")
    for k,tr,te in zip(ks, train_acc_list, test_acc_list):
        print(f"k={k:2d} | train acc:{tr:.3f}, test acc:{te:.3f}")

    
    Xtr_2d, ytr_2d, Xte_2d, yte_2d = shuffle_split(X_2d, y_full, seed=42)
    plot_knn_boundary(Xtr_2d, ytr_2d, X_2d, y_full, k=5, title="KNN Decision Boundary k=5", save_path="knn_boundary.png")