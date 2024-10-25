import numpy as np

# 加载npz文件
data = np.load('./reddit.npz')

# 遍历每个数组并打印形状和大小
for name, array in data.items():
    print(f"Name: {name}, Shape: {array.shape}, Size: {array.nbytes} bytes")

train_index = data['train_index']
y_train = data['y_train']
feats_shape = data['feats'].shape  # 获取 feats 的形状

# 组成元组对
paired_data = list(zip(train_index, y_train))

# 按照 train_index 排序
sorted_data = sorted(paired_data, key=lambda x: x[0])

# 写入txt文件
with open('reddit.txt', 'w') as f:
    # 写入矩阵的长和宽，以及 train_index 的大小
    f.write(f"{feats_shape[0]}\t{feats_shape[0]}\t{len(train_index)}\n")

    for index, value in sorted_data:
        f.write(f"{index}\t{value}\n")