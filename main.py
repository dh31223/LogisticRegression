#导入包

#数据处理包
import numpy as np
import pandas as pd
#线性模型里的逻辑回归包
from sklearn.linear_model import LogisticRegression
#数据标准化包
from sklearn.preprocessing import StandardScaler
#模型评分包
from sklearn.metrics import accuracy_score
#数据划分包
from sklearn.model_selection import train_test_split
#导出模型
import joblib

#将txt文件转化为csv文件
def trans_data(path : str = 'cancer.csv', save : bool = False):

    #定义列名
    column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
                    'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
                    'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

    data = pd.read_csv('cancer.txt', names = column_names)

    if save:
        data.to_csv(path_or_buf = path, index = False)



#训练模型
def train_model(path : str = 'Model', save : bool = False):
    #读入数据
    data = pd.read_csv('cancer.csv')

    #处理空值
    data.dropna(inplace = True)

    #确定特征值和标签
    x = data.iloc[:, 1:-1]
    y = data['Class']

    #数据划分
    #random_state传入随机数种子
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 10)


    #数据标准化
    transform = StandardScaler()
    x_train = transform.fit_transform(x_train)
    x_test = transform.transform(x_test)

    #训练模型
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)

    #模型预测
    y_predict = estimator.predict(x_test)


    #模型评分
    print('模型准确率-->', accuracy_score(y_test, y_predict))


    #模型导出
    if save:
        joblib.dump(value = estimator, filename = path)


if __name__ == '__main__':
    train_model()
