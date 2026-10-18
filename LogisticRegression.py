import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

def sigmoid(z):
    """Sigmoid激活函数
    :param z: 输入，线性预测值 X@theta
    :return: 概率，范围(0,1)
    """
    # 防止数值溢出，限制z范围
    z = np.clip(z, -500, 500)
    return 1.0 / (1 + np.exp(-z))

class LogisticRegressionGD:
    """
    手写逻辑回归：批量梯度下降 BGD
    目标：最小化交叉熵损失（二元交叉熵）
    """
    def __init__(self, lr=0.1, n_iter=1000, seed=42):
        self.lr = lr                # 学习率
        self.n_iter = n_iter        # 迭代轮数
        self.seed = seed
        self.theta = None           # 参数向量 [w0, w1, w2,...] w0是偏置
        self.loss_history = []      # 保存每一轮损失，用于绘图

    def fit(self, X, y):
        """
        训练：批量梯度下降
        :param X: 特征矩阵 (n_samples, n_features)
        :param y: 标签 0/1 (n_samples,)
        """
        n_samples, n_feats = X.shape
        # 在特征最前面增加一列全1，对应偏置项 w0
        X_b = np.hstack([np.ones((n_samples, 1)), X])

        # 初始化参数
        rng = np.random.default_rng(self.seed)
        self.theta = rng.normal(0, 0.01, size=X_b.shape[1])

        for _ in range(self.n_iter):
            # 1. 线性预测
            z = X_b @ self.theta
            y_hat = sigmoid(z)

            # 2. 二元交叉熵损失（目标函数）
            # loss = -1/N * sum(y*log(y_hat)+(1-y)*log(1-y_hat))
            loss = -np.mean(y * np.log(y_hat + 1e-12) + (1 - y) * np.log(1 - y_hat + 1e-12))
            self.loss_history.append(loss)

            # 3. 计算梯度
            # 梯度公式：∂L/∂theta = (1/N) * X_b.T @ (y_hat - y)
            grad = (1 / n_samples) * X_b.T @ (y_hat - y)

            # 4. 更新参数：梯度下降
            self.theta -= self.lr * grad

    def predict_proba(self, X):
        """预测为类别1的概率"""
        n_samples = X.shape[0]
        X_b = np.hstack([np.ones((n_samples, 1)), X])
        z = X_b @ self.theta
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        """预测类别，大于0.5输出1，否则0"""
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(int)

def shuffle_split(X, y, test_size=0.3, seed=0):
    """打乱划分训练集测试集，和KNN代码保持一致，统一项目接口"""
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

if __name__ == "__main__":
    # 加载鸢尾花数据集，取前两类做二分类：0=setosa,1=versicolor
    data = load_iris()
    X_all = data.data
    y_all = data.target

    # 筛选标签0和1，去掉第三类，转为二分类任务
    mask = (y_all == 0) | (y_all == 1)
    X = X_all[mask][:, :2]   # 取前2维特征，方便可视化
    y = y_all[mask]

    # 划分训练/测试集
    X_train, y_train, X_test, y_test = shuffle_split(X, y, test_size=0.3, seed=42)

    # 初始化并训练逻辑回归
    model = LogisticRegressionGD(lr=0.3, n_iter=1200, seed=42)
    model.fit(X_train, y_train)

    # 预测+评估
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("===== 手写逻辑回归（梯度下降）结果 =====")
    print(f"训练集准确率: {train_acc:.3f}")
    print(f"测试集准确率: {test_acc:.3f}")
    print(f"最终参数 theta [bias,w1,w2]: {np.round(model.theta, 4)}")

    # 绘制损失下降曲线
    plt.figure(figsize=(7, 4))
    plt.plot(model.loss_history, label="Cross Entropy Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Binary Cross Entropy Loss")
    plt.title("Loss Curve of Logistic Regression (Gradient Descent)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("logreg_loss_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 绘制逻辑回归决策边界（二维）
    plt.figure(figsize=(6,5))
    xmin, xmax = X[:,0].min()-0.5, X[:,0].max()+0.5
    ymin, ymax = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(xmin,xmax,200), np.linspace(ymin,ymax,200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    plt.scatter(X[:,0], X[:,1], c=y, cmap="tab10", s=25, edgecolors="k", linewidths=0.4)
    plt.title("Logistic Regression Decision Boundary")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.savefig("logreg_decision_boundary.png", dpi=300, bbox_inches="tight")
    plt.show()