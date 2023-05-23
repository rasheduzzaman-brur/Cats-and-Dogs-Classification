import os.path

from torch.utils.data import Dataset,DataLoader
from torchvision import datasets

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from model import Network






def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.001,
                        help='learning rate ')
    parser.add_argument('--train_path', default='', help="train data path")
    parser.add_argument('--test_path', default='', help="test data path")
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Learning rate step gamma')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='CUDA ues for training')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the training and validation sets
    train_dataset = datasets.ImageFolder(os.path.join(args.train_path, 'train'), transform)
    test_dataset = datasets.ImageFolder(os.path.join(args.test_path, 'test'), transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=args.shuffle,num_workers=args.num_workers,pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,shuffle=args.shuffle,num_workers=args.num_workers,pin_memory=args.pin_memory)
    model = Network(args.num_class).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)


    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "cats_dogs.pt")


if __name__ == '__main__':
    main()

