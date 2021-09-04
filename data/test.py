import csv
import pandas as pd
data = pd.read_csv("/home/math-tr/xq/dbrl_1/bigdata2021-rl-recsys/train_user_consumed.csv")
print(data)

# def read_csv(path):
#
#     rows = open(path, 'r', encoding='utf-8').readlines()
#     lines = [x.rstrip() for x in rows] # 去掉每行数据的/n转义字符
# if __name__ == '__main__':
#     path = r"/home/math-tr/xq/dbrl_1/bigdata2021-rl-recsys/train_user_consumed.csv"
#     read_csv(path)
# # with open("/home/math-tr/xq/dbrl_1/bigdata2021-rl-recsys/train_user_consumed.csv",'r',encoding="utf-8") as f:
# #     reader = csv.reader(f)
#     print("text:", read_csv(path))