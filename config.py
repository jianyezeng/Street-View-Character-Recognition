import torch
config ={
    'train_path': './data/mchar_train/mchar_train',
    'valid_path':'./data/mchar_val/mchar_val',
    'train_labels':'./data/mchar_train.json',
    'valid_labels':'./data/mchar_val.json',
    'new_train_labels':'./data/new_mchar_train.json',
    'new_valid_labels':'./data/new_mchar_val.json',
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'