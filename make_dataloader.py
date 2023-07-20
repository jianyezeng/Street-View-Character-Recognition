import os
import torch
import json
from torchvision import transforms
from PIL import Image
from config import config
import numpy as np
from torch.utils.data import dataloader,Dataset
def make_dataloader(opt):
    def json_data_processing(read_path,save_path):
        with open(read_path, 'r') as f:
            read_data = json.load(f)
        #linl = []
        #for key, item in read_data.items():
        #    linl.append(len(item['label']))
        #max_len = max(linl)
        for key,item in read_data.items():
            while len(item['label']) < 5:
                item['height'].append(0)
                item['label'].append(0)
                item['left'].append(0)
                item['top'].append(0)
                item['width'].append(0)
            while len(item['label']) > 5:
                item['height'].pop(-1)
                item['label'].pop(-1)
                item['left'].pop(-1)
                item['top'].pop(-1)
                item['width'].pop(-1)

        # 转换为 JSON 格式的字符串
        json_str = json.dumps(read_data, indent=2)

        # 将 JSON 字符串写入文件
        with open(save_path, 'w') as f:
            f.write(json_str)

    class CustomDataset(Dataset):
        def __init__(self, root, labels_file, transform):
            self.root = root
            self.labels_file = labels_file
            self.transform = transform
            self.labels = {}
            with open(self.labels_file) as f:
                labels = json.load(f)

            for filename in os.listdir(root):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    label = labels.get(filename, -1)
                    self.labels[filename] = label
            self.filenames = list(self.labels.keys())

        def __len__(self):
            return len(self.filenames)

        def __getitem__(self, index):
            filename = self.filenames[index]
            image = Image.open(os.path.join(self.root, filename)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)

            label = self.labels[filename]['label']
            label = np.array(list(map(int, label)))
            return image, label

    # 使用示例
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    if opt == 'train':
        json_data_processing(config['train_labels'],config['new_train_labels'])
        path = config['train_path']
        labels = config['new_train_labels']
    elif opt == 'valid':
        json_data_processing(config['valid_labels'],config['new_valid_labels'])
        path = config['valid_path']
        labels = config['new_valid_labels']
    else:
        path = config['test_path']
        labels = None
    dataset = CustomDataset(path,labels , transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # 打印数据集的长度和一个样本的形状和标签
    return dataset,dataloader

if __name__ == '__main__':
    opt = 'valid'
    dataset,dataloader = make_dataloader(opt)
    for i,j in dataloader:
        print(j.size())