import torch
from config import device
from config import config
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

def predict(model):
    class ImageDataset(Dataset):
        def __init__(self, folder_path,transform):
            self.transform = transform
            self.folder_path = folder_path
            self.image_list = os.listdir(folder_path)

        def __len__(self):
            return len(self.image_list)

        def __getitem__(self, index):
            image_name = self.image_list[index]
            image_path = os.path.join(self.folder_path, image_name)
            image = Image.open(image_path)
            image = self.transform(image)
            label = image_name
            return image, label

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = ImageDataset(config['test_path'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    model = model().to(device)
    model.load_state_dict(torch.load(config['save_path']))
    model.eval()
    test_pred = []
    for input,name in dataloader:
        if device == 'cuda':
            input = input.cuda()
        c0,c1,c2,c3,c4 = model(input)
        output = np.concatenate([
            c0.data.cpu().numpy(),
            c1.data.cpu().numpy(),
            c2.data.cpu().numpy(),
            c3.data.cpu().numpy(),
            c4.data.cpu().numpy()], axis=1)
        test_pred.append(output)
    test_pred = np.vstack(test_pred)
    test_predict_label = np.vstack([
        test_pred[:, :11].argmax(1),
        test_pred[:, 11:22].argmax(1),
        test_pred[:, 22:33].argmax(1),
        test_pred[:, 33:44].argmax(1),
        test_pred[:, 44:55].argmax(1),
    ]).T

    test_label_pred = []
    for x in test_predict_label:
        test_label_pred.append(''.join(map(str, x[x != 10])))

    result_list = []
    for number in test_label_pred:
        number_str = str(number)
        number_without_zeros = number_str.replace('0', '')
        result_list.append(int(number_without_zeros))

    df_submit = pd.read_csv(config['sample_path'])
    df_submit['file_code'] = result_list
    df_submit.to_csv(config['submit_name'], index=None)

if __name__ == '__main__':
    from network import svhn
    predict(svhn)