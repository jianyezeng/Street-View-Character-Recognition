from make_dataloader import make_dataloader
from network import svhn

def main():
    train_dataloader = make_dataloader('train')
    valid_dataloader = make_dataloader('valid')
if __name__ == '__main__':
    main()