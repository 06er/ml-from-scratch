# ml-from-scratch
> 从零手写经典机器学习算法，不直接调用sklearn的算法API，仅使用numpy/matplotlib做底层实现；项目用于学习、面试演示。
## 项目简介
本项目手动实现K近邻(KNN)、逻辑回归(Logistic Regression)两大基础分类算法。
1. KNN：手动实现距离计算、数据集划分、K值扫描寻优、决策边界可视化
2. Logistic Regression：基于梯度下降实现，包含二分类交叉熵损失、梯度求解，绘制损失收敛曲线，评估模型准确率

数据集：Iris鸢尾花数据集
依赖库：`numpy` `matplotlib` `sklearn.datasets`（仅加载数据集，不调用模型）

## 目录结构
ml-from-scratch/
├── KNN.py                 # KNN 完整代码
├── LogisticRegression.py  # 手写逻辑回归（梯度下降）
├── figs/                  # 存放生成实验图片
│   ├── knn_decision_boundary.png   # KNN 决策边界图
│   ├── knn_k_acc_curve.png         # K 值 - 准确率曲线
│   └── lr_loss_curve.png           # 逻辑回归 loss 收敛曲线
└── README.md