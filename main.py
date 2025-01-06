import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, RandomCrop
from torchvision import datasets
from torch.utils.data import random_split
import numpy as np

class LeNet5V2(nn.Module):
    def __init__(self, args, num_classes=100):
        super(LeNet5V2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)  # Added padding
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adjusted size calculation based on padding
        self.fc1 = nn.Linear(64 * 8 * 8, 384)  # Updated size
        self.bn3 = nn.BatchNorm1d(384)
        self.dropout1 = nn.Dropout(args.dropout1)

        self.fc2 = nn.Linear(384, 192)
        self.bn4 = nn.BatchNorm1d(192)
        self.dropout2 = nn.Dropout(args.dropout2)

        self.fc3 = nn.Linear(192, num_classes)

        # Proper initialization
        if args.init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)  # Updated size
        x = self.dropout1(F.relu(self.bn3(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn4(self.fc2(x))))
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


class LeNet5(nn.Module):
    def __init__(self, num_classes=100):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 6 * 6, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#TODO: deletable
# def setup_distributed():
#     if not dist.is_initialized():
#         os.environ['RANK'] = '0'
#         os.environ['WORLD_SIZE'] = '1'
#         os.environ['MASTER_ADDR'] = '127.0.0.1'
#         os.environ['MASTER_PORT'] = '29500'
#         dist.init_process_group(backend='nccl', init_method='env://')

def load_data(batch_size, hardTransform):
    if hardTransform:
        transform = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            RandomRotation(10),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader

def create_optimizer(opt_name, model, lr, weight_decay=0.0001):
    if opt_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_name.lower() == 'adam':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name.lower() == 'lars':
        return optim.LARS(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, eps=1e-8, trust_coef=0.001)
    elif opt_name.lower() == 'lamb':
        return optim.LAMB(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-6)
    else:
        raise ValueError(f"Optimizer {opt_name} not supported")

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100 * correct / total

def train(args):
    # Setup device and distributed training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # setup_distributed() #DELETABLE!

    # Load data
    train_dataset, test_dataset, train_loader, test_loader = load_data(args.batch_size, args.transform)

    # Initialize model
    if args.model_version == 1:
        model = LeNet5()
    else:
        model = LeNet5V2(args)
    if args.starting_model:
        model.load_state_dict(torch.load(args.starting_model))
    model.to(device)

    # Setup distributed model
    if dist.is_initialized():
        model = DDP(model)

    # Split dataset for LocalSGD
    total_size = len(train_dataset)
    split_size = total_size // args.K
    split_sizes = [split_size] * (args.K-1) + [total_size - (args.K-1) * split_size]
    train_subsets = random_split(train_dataset, split_sizes)
    train_loaders_splitted = [torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=True)
                             for subset in train_subsets]

    # Setup training
    criterion = nn.CrossEntropyLoss()
    local_models = [model.to(device) for _ in range(args.K)]
    best_loss = float('inf')
    epochs_done_until_avg = 0
    is_avg_epoch = False

    print(f"Training with {args.K} splits and averaging every {args.J} epochs")

    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        local_losses = []

        for i, local_model in enumerate(local_models):
            local_optimizer = create_optimizer(args.optimizer, local_model, args.learning_rate)
            local_loss = 0.0

            if is_avg_epoch:
                local_model.load_state_dict(model.state_dict())

            for inputs, targets in train_loaders_splitted[i]:
                inputs, targets = inputs.to(device), targets.to(device)
                local_optimizer.zero_grad()
                outputs = local_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                local_optimizer.step()

            local_losses.append(loss.item())

        epochs_done_until_avg += 1
        is_avg_epoch = False

        # Model averaging
        if epochs_done_until_avg == args.J:
            global_dict = model.state_dict()
            for key in global_dict:
                global_dict[key] = torch.mean(torch.stack([local_models[i].state_dict()[key].float()
                                                         for i in range(args.K)]), dim=0)
            if args.J > 1:
                print(f"Model updated at epoch {epoch + 1}")
            model.load_state_dict(global_dict)
            epochs_done_until_avg = 0
            is_avg_epoch = True
            torch.save(model.state_dict(), args.last_model)

        epoch_loss = np.mean(local_losses)
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), args.best_model)
            print(f"Best model saved with loss = {best_loss}")

        print(f"Epoch {epoch+1}: Loss = {loss.item()}")

        # Evaluate on test set periodically
        if (epoch + 1) % args.eval_interval == 0:
            test_acc = evaluate(model, test_loader, device)
            train_acc = evaluate(model, train_loader, device)
            print(f"Epoch {epoch+1} - Test Accuracy: {test_acc:.2f}% - Train Accuracy: {train_acc:.2f}%")

    print("Training completed. Best model loaded.")
    return model, train_loader, test_loader, device

def main():
    parser = argparse.ArgumentParser(description='Distributed Learning with LocalSGD')
    parser.add_argument('--K', type=int, default=1, help='Number of splits for training dataset')
    parser.add_argument('--J', type=int, default=1, help='Interval for weight averaging')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam', 'lars', 'lamb'], help='Optimizer to use')
    parser.add_argument('--best-model', type=str, default='best_model.pth',
                       help='Path to save best model')
    parser.add_argument('--last-model', type=str, default='last_model.pth',
                       help='Path to save last model')
    parser.add_argument('--starting-model', type=str, default=None,
                       help='Path to starting model (optional)')
    parser.add_argument('--start-epoch', type=str, default=0,
                       help='If you started with a trained model, set the epoch the process got stuck')

    parser.add_argument('--eval-interval', type=int, default=5,
                       help='Interval for evaluation during training')
    parser.add_argument('--transform', type=bool, default=False,
                       help='Say if we want to add harder transformations or not')
    parser.add_argument('--model-version', type=int, default=1,
                       help='Say what version of the model we want to use (1 or 2)')
    parser.add_argument('--dropout1', type=float, default=0,
                       help='Says the first dropout. If 0 it is not applied. 0.5 is a default.')
    parser.add_argument('--dropout2', type=float, default=0,
                       help='Says the second dropout. If 0 it is not applied. 0.5 is a default.')
    parser.add_argument('--init-weights', type=bool, default=False,
                       help='Says if weights must be initialized or not.')

    args, uknown = parser.parse_known_args()

    if args.epochs % args.J != 0:
        raise ValueError("Number of epochs must be a multiple of J")

    model, train_loader, test_loader, device = train(args)
    evaluate(model, train_loader, device)
    evaluate(model, test_loader, device)

# if __name__ == '__main__':
#     main()
