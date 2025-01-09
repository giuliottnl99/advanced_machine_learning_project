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
import matplotlib.pyplot as plt
import csv
import torch_optimizer as optim_torch
import math
from torch.cuda.amp import autocast, GradScaler




class LeNet5(nn.Module):
    def __init__(self, num_classes=100):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adjusting fc1 input size to match the output of conv/pool layers
        self.fc1 = nn.Linear(64 * 6 * 6, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Output: (batch_size, 64, 32, 32)
        x = self.pool1(x)          # Output: (batch_size, 64, 16, 16)
        x = F.relu(self.conv2(x))  # Output: (batch_size, 64, 12, 12)
        x = self.pool2(x)          # Output: (batch_size, 64, 6, 6)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_data(batch_size, hardTransform):
    if hardTransform:
        transform = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.5071, 0.4867, 0.4408), (0.2673, 0.2564, 0.2762))
        ])
    else:
        transform = Compose([
            ToTensor(),
            Normalize((0.5071, 0.4867, 0.4408), (0.2673, 0.2564, 0.2762))
        ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader

def create_optimizer(opt_name, model, lr, weight_decay=0.0001, momentum=0.9):
    if opt_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name.lower() == 'adam':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name.lower() == 'lars':
        return optim_torch.LARS(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, eps=1e-8, trust_coefficient=0.001)
    elif opt_name.lower() == 'lamb':
        return optim_torch.LAMB(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-6)
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

#TODO: manage the plot!
def plotEpochs(epochs, data_arr_1, data_arr_2, label1, label2, figName): 
    #inputArray format: {epoch: 'number', var1: 'trainAcc', var2: 'testAcc'}
    #labelsFormat: {lab1: 'label', lab2: 'label'}
    plt.figure()
    plt.plot(epochs, data_arr_1, label=label1)
    plt.plot(epochs, data_arr_2, label=label2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.savefig(figName)

def compute_test_loss(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the device

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Accumulate loss
            test_loss += loss.item() * inputs.size(0)  # Multiply by batch size
            total_samples += inputs.size(0)
    average_loss = test_loss / total_samples
    return average_loss



def train(args):
    # Setup device and distributed training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    train_dataset, test_dataset, train_loader, test_loader = load_data(args.batch_size, args.transform)

    # Initialize model
    if args.model_version == 1:
        model = LeNet5()
    if args.starting_model:
        model.load_state_dict(torch.load(args.starting_model))
    model.to(device)


    # Setup training
    criterion = nn.CrossEntropyLoss()
    local_models = [model.to(device) for _ in range(args.K)]

    print(f"Training with {args.K} splits and averaging every {args.J} epochs, woth {args.warmup} warmup epochs")

    epochs = []
    trainAccuracies = []
    testAccuracies = []
    epoch_training_losses = []
    train_losses = []
    test_losses = []
    if args.start_epoch==0: #create the file if we are starting a new model
        with open('accuraciesAndLosses.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'train-accuracy', 'test-accuracy', 'train_loss', 'test-loss'])  # Write headers


    tot_epochs = args.warmup + args.epochs
    # Training loop
    #for each epoch
    for epoch in range(args.start_epoch, tot_epochs):
        #change lr by applying cosine annealing:
        lr = args.lr_min + 0.5 * (args.lr_max - args.lr_min) * (1 + math.cos(math.pi * epoch / args.epochs))

        model.train() #activate model for training
        local_losses = []
        iterator = iter(train_loader)
        remaining_iterations = len(train_loader)
        steps_before_avg = min(args.J, remaining_iterations/args.K) #just in case we are in the last step and we to not have enough batches, we apply less steps
        while remaining_iterations>=args.K: #I discard the last few iterations that does not have enough local models
            for i, local_model in enumerate(local_models): #for each local model
                if remaining_iterations<0:
                    break
                test_local_total_loss, total_samples = 0.0, 0
                if args.K>1 or epoch==1: #load global model
                    local_model.load_state_dict(model.state_dict())
                local_model.train()
                local_optimizer = create_optimizer(args.optimizer, local_model, lr, momentum=args.momentum, weight_decay=args.weight_decay)
                for nTime in range(steps_before_avg): #for J times
                    inputs, targets = next(iterator) #get a piece of the dataset.
                    remaining_iterations -=1 #1 less iteration after getting piece of the dataset
                    inputs, targets = inputs.to(device), targets.to(device)
                    local_optimizer.zero_grad()
                    outputs = local_model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    local_optimizer.step() 
                    test_local_total_loss += loss.item() * inputs.size(0)  # Multiply by batch size
                    total_samples += inputs.size(0)
                    if remaining_iterations<0:
                        break


                local_losses.append(test_local_total_loss / total_samples) #append avg local losses
            if args.K>1:                
                global_dict = model.state_dict() #After all local_models have been trained, avg
                for key in global_dict:
                    global_dict[key] = torch.mean(torch.stack([local_models[i].state_dict()[key].float()
                                                            for i in range(args.K)]), dim=0)
                model.load_state_dict(global_dict)

        epoch_train_loss = np.mean(local_losses)
        print(f"Epoch {epoch+1}: Training Loss = {epoch_train_loss}")

        # Evaluate on test set periodically
        if (epoch + 1) % args.eval_interval == 0:
            test_acc = evaluate(model, test_loader, device)
            train_acc = evaluate(model, train_loader, device)
            epochs.append(epoch+1)
            trainAccuracies.append(train_acc)
            testAccuracies.append(test_acc)
            epoch_training_losses.append(epoch_train_loss)
            train_losses.append(epoch_train_loss)
            test_loss = compute_test_loss(model, test_loader, criterion, device)
            test_losses.append(test_loss)
            print(f"Epoch {epoch+1} - Test Accuracy: {test_acc:.2f}% - Train Accuracy: {train_acc:.2f}% - Test Loss: {test_loss} - Current LR: {lr}")
            with open('accuraciesAndLosses.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch+1, train_acc, test_acc, epoch_train_loss, test_loss])  # Write headers
                torch.save(model.state_dict(), args.last_model) #save model with the csv

    plotEpochs(epochs, trainAccuracies, testAccuracies, 'Train accuracy', 'Test accuracy', 'accuracy_fig.png')
    plotEpochs(epochs, train_losses, test_losses, 'Train losses', 'Test losses', 'losses_fig.png')
    print("Training completed. Best model loaded.")
    return model, train_loader, test_loader, device

def main():
    parser = argparse.ArgumentParser(description='Distributed Learning with LocalSGD')
    parser.add_argument('--K', type=int, default=1, help='Number of splits for training dataset')
    parser.add_argument('--J', type=int, default=1, help='Interval for weight averaging')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for training')
    parser.add_argument('--lr-max', type=float, default=0.01, help='Learning rate max of cosine annealing')
    parser.add_argument('--lr-min', type=float, default=0.001, help='Learning rate min of cosine annealing')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam', 'lars', 'lamb'], help='Optimizer to use')
    parser.add_argument('--best-model', type=str, default='best_model.pth',
                       help='Path to save best model')
    parser.add_argument('--last-model', type=str, default='last_model.pth',
                       help='Path to save last model')
    parser.add_argument('--starting-model', type=str, default=None,
                       help='Path to starting model (optional)')
    parser.add_argument('--start-epoch', type=int, default=0,
                       help='If you started with a trained model, set the epoch the process got stuck')

    parser.add_argument('--eval-interval', type=int, default=5,
                       help='Interval for evaluation during training')
    parser.add_argument('--transform', type=bool, default=True,
                       help='Say if we want to add harder transformations or not')
    parser.add_argument('--model-version', type=int, default=1,
                       help='Say what version of the model we want to use (1 or 2)')
    parser.add_argument('--warmup', type=int, default=0,
                       help='If 0, no warmup is applied')
    parser.add_argument('--slowMMO-value', type=float, default=0.0,
                       help='Used to apply slowMMO. If 0, it applies simple FedAVG to average the weights')

    #must be removed: model 2 is not authorized!
    # parser.add_argument('--dropout1', type=float, default=0,
    #                    help='Says the first dropout. If 0 it is not applied. 0.5 is a default.')
    # parser.add_argument('--dropout2', type=float, default=0,
    #                    help='Says the second dropout. If 0 it is not applied. 0.5 is a default.')
    # parser.add_argument('--init-weights', type=bool, default=False,
    #                    help='Says if weights must be initialized or not.')

    args, uknown = parser.parse_known_args()

    if args.epochs % args.J != 0:
        raise ValueError("Number of epochs must be a multiple of J")
    if args.eval_interval % args.J != 0:
        raise ValueError("Eval interval must be a multiple of J")

    model, train_loader, test_loader, device = train(args)
    plt.show()


# if __name__ == '__main__':
#     main()
