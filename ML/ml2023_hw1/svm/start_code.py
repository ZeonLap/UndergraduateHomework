import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import trange
import matplotlib.pyplot as plt


def load_text_dataset(filename, positive='joy', negative='sadness'):
    """
    从文件filename读入文本数据集
    """
    data = pd.read_csv(filename)
    is_positive = data.Emotion == positive
    is_negative = data.Emotion == negative
    data = data[is_positive | is_negative]
    X = data.Text  # 输入文本
    y = np.array(data.Emotion == positive) * 2 - 1  # 1: positive, -1: negative
    return X, y


def vectorize(train, test):
    """
    将训练集和验证集中的文本转成向量表示

    Args：
        train - 训练集，大小为 num_instances 的文本数组
        test - 测试集，大小为 num_instances 的文本数组
    Return：
        train_normalized - 向量化的训练集 (num_instances, num_features)
        test_normalized - 向量化的测试集 (num_instances, num_features)
    """
    tfidf = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
    train_normalized = tfidf.fit_transform(train).toarray()
    test_normalized = tfidf.transform(test).toarray()
    return train_normalized, test_normalized


def compute_loss(X, y, theta, lambda_reg):
    reg = lambda_reg * np.sum(theta[:-1] ** 2) / 2
    hinge = np.sum(np.maximum(1 - y * np.dot(X, theta), 0)) / X.shape[0]
    return reg + hinge


def compute_subgrad(X, y, theta, lambda_reg):
    num_instances, num_features = X.shape[0], X.shape[1]
    subgrad_w = np.zeros(num_features - 1)
    subgrad_b = np.zeros(1)
    for i in range(num_instances):
        if y[i] * np.dot(X[i], theta) < 1:
            subgrad_w += (lambda_reg * theta - y[i] * X[i])[:-1]
            subgrad_b += -y[i]
        else:
            subgrad_w += (lambda_reg * theta)[:-1]
            subgrad_b += 0

    subgrad_w /= num_instances
    subgrad_b /= num_instances

    return np.concatenate([subgrad_w, subgrad_b])


def batch_data(X, y, _batch_size):
    """
    从训练集中随机抽取 batch_size 个样本
    """
    assert len(X) >= _batch_size, "batch_size must be no more than num_instances"
    indx = list(range(len(X)))
    np.random.shuffle(indx)
    
    start_idx = 0
    end_idx = min(start_idx + _batch_size, len(X))
    return X[indx[start_idx: end_idx]], y[indx[start_idx: end_idx]]


def kernel_batch_data(X, y, _batch_size):
    assert len(X) >= _batch_size, "batch_size must be no more than num_instances"
    indx = list(range(len(X)))
    np.random.shuffle(indx)
    
    start_idx = 0
    end_idx = min(start_idx + _batch_size, len(X))
    return X[indx[start_idx: end_idx]], y[indx[start_idx: end_idx]], indx[start_idx: end_idx]


def grad_checker(X, y, theta, lambda_reg, epsilon=1e-2, tolerance=1e-4):
    """梯度检查
    如果实际梯度和近似梯度的欧几里得距离超过容差，则梯度计算不正确。

    Args：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小（num_features）
        lambda_reg - 正则化系数
        epsilon - 步长
        tolerance - 容差

    Return：
        梯度是否正确

    """
    true_gradient = compute_subgrad(X, y, theta, lambda_reg)  # the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features)  # Initialize the gradient we approximate

    # TODO 2.2.5 (optional)
    Ident = np.eye(num_features)
    for i in range(num_features):
        J_plus = compute_loss(X, y, theta + epsilon * Ident[i], lambda_reg)
        J_minus = compute_loss(X, y, theta - epsilon * Ident[i], lambda_reg)
        approx_grad[i] = (J_plus - J_minus) / (2 * epsilon)

    return np.all(np.abs(true_gradient - approx_grad) < tolerance)  # 逐元素比较，每个theta_i的梯度误差都小于tolerance



def linear_svm_subgrad_descent(X, y, alpha=0.005, lambda_reg=0.0001, num_iter=100000, batch_size=1):
    """
    线性SVM的随机次梯度下降

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        alpha - 浮点数。梯度下降步长，可自行调整为默认值以外的值或扩展为步长策略
        lambda_reg - 正则化系数，可自行调整为默认值以外的值
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        batch_size - 批大小，可自行调整为默认值以外的值

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter+1, num_features)
        loss hist - 小批量损失函数的历史，数组大小(num_iter)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist

    # TODO 3.4.1
    for i in trange(num_iter):
        X_batch, y_batch = batch_data(X, y, batch_size)
        grad = compute_subgrad(X_batch, y_batch, theta, lambda_reg)
        theta -= alpha * grad
        theta_hist[i + 1] = theta
        loss_hist[i] = compute_loss(X_batch, y_batch, theta, lambda_reg)

    return theta_hist, loss_hist
    

def linear_svm_subgrad_descent_lambda(X, y, lambda_reg=0.0001, num_iter=10000, batch_size=1, mu=1e-3):
    """
    线性SVM的随机次梯度下降;在lambda-强凸条件下有理论更快收敛速度的算法
    该函数每次迭代的梯度下降步长已由算法给出，无需自行调整

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        lambda_reg - 正则化系数，可自行调整为默认值以外的值
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        batch_size - 批大小，可自行调整为默认值以外的值

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter+1, num_features)
        loss hist - 小批量损失函数的历史，数组大小(num_iter)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist

    # TODO 3.4.3
    for i in trange(num_iter):
        X_batch, y_batch = batch_data(X, y, batch_size)
        grad = compute_subgrad(X_batch, y_batch, theta, lambda_reg)
        theta -= 1 / (mu * (i + 1)) * grad
        theta_hist[i + 1] = theta
        loss_hist[i] = compute_loss(X, y, theta, lambda_reg)

    return theta_hist, loss_hist


def compute_kernel_loss(X, y, theta, lambda_reg, K, indx):
    K_indx = K[indx]
    reg = lambda_reg * np.dot(theta, np.dot(K, theta)) / 2
    hinge = np.sum(np.maximum(1 - y * np.dot(K_indx, theta), 0)) / X.shape[0]
    return reg + hinge


def compute_kernel_subgrad(X, y, theta, lambda_reg, K, indx):
    assert(len(indx) == len(y))
    K_indx = K[indx]
    num_instances, num_features = X.shape[0], K.shape[0]
    subgrad = np.zeros(num_features)
    for i in range(num_instances):
        if y[i] * np.dot(K_indx[i], theta) >= 1:
            subgrad += lambda_reg * np.dot(K, theta)
        else:
            subgrad += lambda_reg * np.dot(K, theta) - y[i] * K_indx[i]

    return subgrad / num_instances
    

def kernel_svm_subgrad_descent(X, y, alpha=0.01, lambda_reg=0, num_iter=100000, batch_size=1):
    """
    Kernel SVM的随机次梯度下降

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        alpha - 浮点数。初始梯度下降步长
        lambda_reg - 正则化系数
        num_iter - 遍历整个训练集的次数（即次数）
        batch_size - 批大小

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter, num_features)
        loss hist - 正则化损失函数向量的历史，数组大小(num_iter,)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_instances)  # Initialize theta
    theta_hist = np.zeros((num_iter+1, num_instances))  # Initialize theta_hist
    loss_hist = np.zeros((num_iter,))  # Initialize loss_hist

    # TODO 3.4.4
    # K = np.array([[np.tanh(beta * np.dot(X[i], X[j]) + c) for j in range(num_instances)] for i in range(num_instances)])  # Kernel Matrix
    K = np.array([[np.dot(X[i], X[j]) for j in range(num_instances)] for i in range(num_instances)])  # Kernel Matrix
    for i in trange(num_iter):
        X_batch, y_batch, indx = kernel_batch_data(X, y, batch_size)
        grad = compute_kernel_subgrad(X_batch, y_batch, theta, lambda_reg, K, indx)
        theta -= alpha * grad
        theta_hist[i + 1] = theta
        loss_hist[i] = compute_kernel_loss(X_batch, y_batch, theta, lambda_reg, K, indx)

    return theta_hist, loss_hist


def main():
    # 加载所有数据
    X_train, y_train = load_text_dataset("data_train.csv", "joy", "sadness")
    X_val, y_val = load_text_dataset("data_test.csv")
    print("Training Set Size: {} Validation Set Size: {}".format(len(X_train), len(X_val)))
    print("Training Set Text:", X_train, sep='\n')

    # 将训练集和验证集中的文本转成向量表示
    X_train_vect, X_val_vect = vectorize(X_train, X_val)
    X_train_vect = np.hstack((X_train_vect, np.ones((X_train_vect.shape[0], 1))))  # 增加偏置项
    X_val_vect = np.hstack((X_val_vect, np.ones((X_val_vect.shape[0], 1))))  # 增加偏置项

    # SVM的随机次梯度下降训练
    # TODO
    print("Running SSGD_Lambda Method")
    theta_hist, loss_hist = linear_svm_subgrad_descent_lambda(X_train_vect, y_train, lambda_reg=0.001, num_iter=10000)

    # plt.xlabel("Iteration")
    # plt.plot(loss_hist)
    # plt.legend()
    # plt.show()

    # 计算SVM模型在验证集上的准确率，F1-Score以及混淆矩阵
    # TODO

    print("Evaluating on Validation Set")
    theta = theta_hist[-1]
    
    y_pred = np.where(np.dot(X_val_vect, theta) > 0, 1, -1)
    # # Kernel Method
    # def K(x, y):
    #     return np.dot(x, y)

    # y_pred = []
    # for i in range(X_val_vect.shape[0]):
    #     y_pred_current = 0
    #     for j in range(X_train_vect.shape[0]):
    #         y_pred_current += theta[j] * K(X_val_vect[i], X_train_vect[j])
    #     y_pred.append(1 if y_pred_current > 0 else -1)

    print(f'验证集的准确率为{accuracy_score(y_val, y_pred)}')
    print(f'验证集的F1-Score为{f1_score(y_val, y_pred)}')
    print(f'验证集的混淆矩阵为{confusion_matrix(y_val, y_pred)}')


if __name__ == '__main__':
    main()
