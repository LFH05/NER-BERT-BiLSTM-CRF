#!/usr/bin/env python3
# coding: utf-8

"""
@File   : train.py
@Author : FuHuang Liu
@Date   : 2022/9/23
@Desc   : 训练
"""
import numpy as np
import torch

from utils import *
from model import *
from config import *
from seqeval.metrics import f1_score, precision_score, recall_score
import os
import matplotlib.pyplot as plt


def train():
    # 读取训练文件夹
    train_filename = os.path.join('data', 'fina_data', 'role_train.tsv')
    # 返回训练数据的文本和标签
    train_text, train_label = read_data2(train_filename)

    # 验证集
    dev_filename = os.path.join('data', 'fina_data', 'role_dev.tsv')
    dev_text, dev_label = read_data2(dev_filename)
    # print(train_filename)

    # 得到label2index, index2label
    label2index, index2label = build_label_2_index(train_label)

    # 数据迭代器 DataSet
    train_data = MyData(train_text, train_label, tokenizer, label2index, MAX_LEN)
    # train_data = train_data.remove_columns_(train_data["train"].column_names)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=False)

    dev_data = MyData(dev_text, dev_label, tokenizer, label2index, MAX_LEN)
    dev_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    # 模型
    model = MyModel(len(label2index)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 记录每个loss值和f1值的变化
    index = np.linspace(0, 99, 100)
    loss_val = []
    f1_val = []
    max_f1 = 0
    # 训练

    for epoch in range(EPOCHS):
        print("======================第{0}epoch=====================".format(epoch))
        model.train()
        for batch_idx, data in enumerate(train_loader):
            min_loss = 2000
            batch_text, batch_label, batch_len, mask = data

            # print(batch_idx)
            # print(batch_text.shape)
            # print(batch_label.shape)
            # 将数据放到GPU上
            # loss = model(batch_text.to(DEVICE), batch_label.to(DEVICE), mask)
            loss = model.loss_fn(batch_text.to(DEVICE), batch_label.to(DEVICE), mask.to(DEVICE))
            if loss < min_loss:
                min_loss = loss.item()
                # print("save model")
                # torch.save(model, MODEL_DIR + f'model.pth')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'Epoch: {epoch}, min_training_loss: {min_loss}')
        loss_val.append(min_loss)

        # 保存模型
        # if epoch == 99:
        #     torch.save(model, MODEL_DIR + f'model_{epoch}.pth')

        model.eval()

        # 用来存放预测标签和真实标签
        all_pre = []
        all_tag = []

        for batch_text, batch_label, batch_len, mask in dev_loader:
            # batch_text, batch_label, batch_len = batch_text.to(DEVICE), batch_label.to(DEVICE), batch_len.to(DEVICE)
            # 因为是预测，所以在模型输入的地方，没有加入batch_label
            pre = model(batch_text.to(DEVICE), mask.to(DEVICE))

            # 将pre从GPU上读下来，转成list
            # pre = pre.cpu().numpy().tolist()
            batch_label = batch_label.cpu().numpy().tolist()

            # 还有一点要注意， from seqeval.metrics import f1_score
            # 在使用 f1_score的时候，所需要的标签应该是完整的，而不是经过填充过的
            # 所以我们需要将填充过的标签信息进行拆分怎么做呢？
            # 就需要将最开始没有填充过的文本长度记录下来，在__getitem__的返回量中增加一个长度量，那样我们就能知道文本真实长度
            # 然后就此进行切分，因为左边增加了一个开始符，需要去掉一个即可；右边按照长度来切分

            for p, t, l in zip(pre, batch_label, batch_len):
                p = p[1: l + 1]
                t = t[1: l + 1]

                pre = [index2label[j] for j in p]
                tag = [index2label[j] for j in t]
                all_pre.append(pre)
                all_tag.append(tag)
        f1_score_ = f1_score(all_pre, all_tag)
        if f1_score_ > max_f1:
            max_f1 = f1_score_
            print("保存模型")
            torch.save(model, MODEL_DIR + f'model_all.pth')
        p_score = precision_score(all_pre, all_tag)
        r_score = recall_score(all_pre, all_tag)
        # f1_score(batch_label_index, pre)
        print(f'p值={p_score}, r值={r_score}, f1={f1_score_}')
        # print(2*p_score*r_score/(p_score+r_score))
        f1_val.append(f1_score_)
    print("最大的f1值为：{0}，第{1}个EPOCH".format(max(f1_val), f1_val.index(max(f1_val))))
    # 运行完了，看一下Loss和f1值的变化
    # plt.plot(index, f1_val, ls="-", lw="2", label="plot figure")
    # plt.legend()
    # # plt.show()
    # plt.savefig("ss.png")


if __name__ == '__main__':
    train()