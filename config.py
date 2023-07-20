import torch
config ={
    'train_path':'./data/mchar_train/mchar_train',
    'valid_path':'./data/mchar_val/mchar_val',
    'train_labels':'./data/mchar_train.json',
    'valid_labels':'./data/mchar_val.json',
    'new_train_labels':'./data/new_mchar_train.json',
    'new_valid_labels':'./data/new_mchar_val.json',
    'test_path':'./data/mchar_test/mchar_test',
    'seed':512,
    'n_epochs':2,
    'batch_size':32,
    'early_stop':20,
    'save_path': './models/model.ckpt',
    'learning_rate':0.001,
    'sample_path':'./data/sample.csv',
    'submit_name':'submit.csv',
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)