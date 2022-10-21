# README

#Mine本地工程名为：“BERT_BiLSTM”，对应的pycharm依赖文件均以此为名

## environment：

python——3.6

torch——1.5.0

transformers——4.4.1



## More:

dataset——DuEE1.0

训练好模型："./model"

数据处理："./Data_Pro.py"

配置文件："./config.py"

相关函数文件："./utils.py"

**训练："./train.py"**

预测："./predict.py"



## Run:

1. 将Bert模型下载保存到**"./BERT_MODEL"**文件夹下
2. 将数据用**"./Data_pro.py"**处理成需要的标注形式
3. 直接运行**"./train.py"**文件开始训练，训练好的模型保存在**"./model"**文件夹下
4. 运行**"./predict.py"**可查看模型预测效果

