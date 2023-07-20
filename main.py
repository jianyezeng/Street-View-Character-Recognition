from make_dataloader import make_dataloader
from network import svhn
from train import training
from pred import predict
from utils import create_modelfolder
def main():
    create_modelfolder('./models')
    train_set,train_dataloader = make_dataloader('train')
    valid_set,valid_dataloader = make_dataloader('valid')
    training(train_dataloader,valid_dataloader,svhn)
    predict(svhn)
if __name__ == '__main__':
    main()