import torch
import numpy as np
import os
def same_seed(seed):
    '''随机种子'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_modelfolder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 创建成功！")
    else:
        print(f"文件夹 '{folder_path}' 已存在！")