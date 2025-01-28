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


global first_epoch
first_epoch = True

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
        x = F.relu(self.conv1(x))  # Output: (batch_size, 64, 32, 32) Add norm
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
            Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2673, 0.2564, 0.2762))
        ])
    else:
        transform = Compose([
            ToTensor(),
            Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2673, 0.2564, 0.2762))
        ])

    full_train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_size = int(0.9 * len(full_train_dataset))  # 90% for training
    test_size = len(full_train_dataset) - train_size  # Remaining 10% for validation
    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(full_train_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, val_dataset, train_loader, test_loader, val_loader

def create_optimizer(args, model, lr):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'lars':
        return optim_torch.LARS(model.parameters(), lr=lr, weight_decay=args.weight_decay, momentum=args.momentum, eps=1e-8, trust_coefficient=args.trust_coefficient)
    elif args.optimizer.lower() == 'lamb':
        # return LAMB(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-6)
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        # return LAMB(learning_rate=lr, beta_1=args.momentum, beta_2=args.beta_2, weight_decay=args.weight_decay, epsilon=1e-6, exclude_from_weight_decay=["bias", "LayerNorm"])
        return LAMB(learning_rate=lr, beta_1=args.momentum, beta_2=args.beta_2, weight_decay=args.weight_decay, epsilon=1e-6)
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

def compute_loss(model, loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the device

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Accumulate loss
            test_loss += loss.item() * inputs.size(0)  # Multiply by batch size
            total_samples += inputs.size(0)
    average_loss = test_loss / total_samples
    return average_loss


def initializeLrs(args):
    lr_max_after_warmup = args.lr_max * ((float(args.batch_size)/64.0)**0.5) #Used for large batches. es: lr_max=0.1 and batch_size=128 => lr_max_after_warmup=1.412

    lr_max_before_warmup = args.lr_max
    if args.from_zero_warmup:
        lr_max_before_warmup = 0.0
    lr = lr_max_after_warmup
    return lr_max_after_warmup, lr_max_before_warmup, lr

def computeCurrentLr(args, epoch, lr_max_after_warmup, lr_max_before_warmup):
    warmed_up_epoch = epoch - args.warmup #this is the epoch when the effective training starts
    if (args.optimizer=='sgd' or args.optimizer=='adam') and epoch>=args.warmup:
        lr = args.lr_min + 0.5 * (lr_max_after_warmup - args.lr_min) * (1 + math.cos(math.pi * warmed_up_epoch / (args.epochs))) #change lr by applying cosine annealing:
    elif (args.optimizer=='lars' or args.optimizer=='lamb') and epoch>=args.warmup:
        decay_factor = (1 - (warmed_up_epoch / args.epochs))
        lr = lr_max_after_warmup * decay_factor #change lr by applying polynomial decay:
    elif epoch<args.warmup:         #if we are in warmup
        lr = lr_max_before_warmup + (lr_max_after_warmup - lr_max_before_warmup)*(float(epoch)/float(args.warmup))
    if epoch==args.warmup and args.warmup != 0:
        print("warmup terminated")
    return lr

def runBatchTrain(args, iterator, device, local_optimizer, local_model, criterion, scaler, test_local_total_loss, total_samples):
    inputs, targets = next(iterator) #get a piece of the dataset. TODO: fix the bug in LARS where batch_size=64
    inputs, targets = inputs.to(device), targets.to(device)
    if args.optimizer != 'lamb':
        local_optimizer.zero_grad()
    outputs = local_model(inputs)
    loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    if args.optimizer != 'lamb':
        scaler.step(local_optimizer)
        scaler.update()

 
    test_local_total_loss += loss.item() * inputs.size(0)  # Multiply by batch size
    total_samples += inputs.size(0)
    return test_local_total_loss, total_samples

def updateLocalModel(args, model, epoch, local_model, local_optimizer, lr):
    if args.K>1 or epoch==1: #load global model
        local_model.load_state_dict(model.state_dict())
        local_model.train()
        local_optimizer = create_optimizer(args, local_model, lr) #initialize in case there is only one. This saves computational power.
    else:
        local_model.train()
    return local_model, local_optimizer

def trainLocalModelForJSteps(args, remaining_iterations, local_losses, local_model, local_optimizer, criterion, iterator, device, scaler, steps_before_avg, local_model_num):
    test_local_total_loss, total_samples = 0.0, 0

    for nTime in range(steps_before_avg): #for J times
        test_local_total_loss, total_samples = runBatchTrain(args, iterator, device, local_optimizer, local_model, criterion, scaler, test_local_total_loss, total_samples)
        remaining_iterations -= 1
        if remaining_iterations <=0:
            global first_epoch
            if first_epoch:
                print(f"Warning: not enough iterations to complete the final iteration. nTime/stepsBeforeAvg={nTime}/{steps_before_avg} and remaining_iterations={remaining_iterations}, local model num {local_model_num}")
                first_epoch=False #this output is always the same for each epoch, no need to print more than once!
            break
    local_losses.append(test_local_total_loss / total_samples) #append avg local losses
    if remaining_iterations<=0:
        return local_losses, remaining_iterations, True
    else: 
        return local_losses, remaining_iterations, False

def updateGlobalModel(args, model, local_models):
    global_dict = model.state_dict() #After all local_models have been trained, avg
    for key in global_dict:
        global_dict[key] = global_dict[key] * args.slowMMO + (1-args.slowMMO) * torch.mean(torch.stack([local_models[i].state_dict()[key].float()
                                                for i in range(args.K)]), dim=0)
    model.load_state_dict(global_dict)

def updateCsvAndLastModel(model, device, csv_name, last_model_path, epoch, train_loader, val_loader, epoch_train_loss, criterion):
    val_acc = evaluate(model, val_loader, device)
    train_acc = evaluate(model, train_loader, device)
    val_loss = compute_loss(model, val_loader, criterion, device)
    with open(csv_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, train_acc, val_acc, epoch_train_loss, val_loss])  # Write headers
        torch.save(model.state_dict(), last_model_path) #save model with the csv
        print("Model and csv updated")
    return val_acc, train_acc, val_loss

def initCsvFile(csv_name):
    with open(csv_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train-accuracy', 'val-accuracy', 'train_loss', 'val-loss'])  # Write headers


def train(args):
    # Setup device and distributed training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    train_dataset, test_dataset, val_dataset, train_loader, test_loader, val_loader = load_data(args.batch_size, args.transform)

    # Initialize model
    model = LeNet5() 
    if args.starting_model:
        model.load_state_dict(torch.load(args.starting_model))
    model.to(device)
    finalStr = f'K={args.K}_J={args.J}_epochs={args.epochs}_batch_size={args.batch_size}_momentum={args.momentum}_lr_max={args.lr_max}_lr_min={args.lr_min}_weight_decay={args.weight_decay}_optimizer={args.optimizer}_warmup={args.warmup}_slowMMO={args.slowMMO}_from_zero_warmup={args.from_zero_warmup}_trust_coefficient={args.trust_coefficient}'
    last_model_path = args.save_path + 'last_model' + finalStr + '.pth'
    csv_name = args.save_path + 'accuracies_and_losses' + finalStr + '.csv'

    # Setup training
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) #correct to change?
    local_models = [model.to(device) for _ in range(args.K)]

    print(f"Training with {args.K} splits and averaging every {args.J} epochs, {args.warmup} warmup epochs")

    if args.start_epoch==0: #create the file if we are starting a new model
        initCsvFile(csv_name)

    tot_epochs = args.warmup + args.epochs
    lr_max_after_warmup, lr_max_before_warmup, lr = initializeLrs(args)

    for epoch in range(args.start_epoch, tot_epochs):
        lr = computeCurrentLr(args, epoch, lr_max_after_warmup, lr_max_before_warmup)
        model.train() #activate model for training
        local_losses, iterator, remaining_iterations, scaler, breakNext = [], iter(train_loader), len(train_loader), GradScaler(), False
        local_optimizer = create_optimizer(args, model, lr) #initialize in case there is only one. This saves computational power. 
        while remaining_iterations>=args.K and breakNext==False: #I discard the last few iterations that does not have enough local models
            if args.J>=float(remaining_iterations)/float(args.K): #
                breakNext=True
            steps_before_avg = min(args.J, math.floor(float(remaining_iterations)/float(args.K))) #just in case we are in the last step and we to not have enough batches, we apply less steps
            if first_epoch and steps_before_avg<args.J:
                print(f"last batch of the epoch has {steps_before_avg} steps before avg.")

            for i, local_model in enumerate(local_models): #for each local model
                local_model, local_optimizer = updateLocalModel(args, model, epoch, local_model, local_optimizer, lr) #TODO: change this!
                local_losses, remaining_iterations, breakNext = trainLocalModelForJSteps(args, remaining_iterations, local_losses, local_model, local_optimizer, criterion, iterator, device, scaler, steps_before_avg, i)
                if breakNext:
                    break
            if args.K>1:                
                updateGlobalModel(args, model, local_models)
        epoch_train_loss = np.mean(local_losses)
        print(f"Epoch {epoch+1}: Training Loss = {epoch_train_loss} lr={lr}")

        # Evaluate on test set periodically
        if (epoch + 1) % args.eval_interval == 0:
            val_acc, train_acc, val_loss = updateCsvAndLastModel(model, device, csv_name, last_model_path, epoch, train_loader, val_loader, epoch_train_loss, criterion)
            print(f"Epoch {epoch+1} - Validation Accuracy: {val_acc:.2f}% - Train Accuracy: {train_acc:.2f}% - Validation Loss: {val_loss} - Current LR: {lr}")

    test_acc = evaluate(model, test_loader, device)
    test_loss = compute_loss(model, test_loader, criterion, device)

    print(f"Training completed - Test Accuracy: {test_acc:.2f}% - Test Loss: {test_loss:.2f}")
    return model, train_loader, val_loader, device

def main():
    parser = argparse.ArgumentParser(description='Distributed Learning with LocalSGD')
    parser.add_argument('--K', type=int, default=1, help='Number of splits for training dataset')
    parser.add_argument('--J', type=int, default=1, help='Interval for weight averaging')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for training. In LAMB is Beta-1 value')
    parser.add_argument('--lr-max', type=float, default=0.01, help='Learning rate max of cosine annealing')
    parser.add_argument('--lr-min', type=float, default=0.001, help='Learning rate min of cosine annealing')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam', 'lars', 'lamb'], help='Optimizer to use')
    parser.add_argument('--save-path', type=str, default='',
                       help='Path where to save csv and last model')

    parser.add_argument('--starting-model', type=str, default=None,
                       help='Path to starting model (optional)')
    parser.add_argument('--start-epoch', type=int, default=0,
                       help='If you started with a trained model, set the epoch the process got stuck')

    parser.add_argument('--eval-interval', type=int, default=5,
                       help='Interval for evaluation during training')
    parser.add_argument('--transform', type=bool, default=True,
                       help='Say if we want to add harder transformations or not')
    parser.add_argument('--warmup', type=int, default=0,
                       help='If 0, no warmup is applied')
    parser.add_argument('--slowMMO', type=float, default=0.0,
                       help='Used to apply slowMMO. If 0, it applies simple FedAVG to average the weights')
    parser.add_argument('--from-zero-warmup', type=bool, default=False,
                       help='In case you want a special warmup, set to 0')
    parser.add_argument('--trust-coefficient', type=float, default=0.001,
                       help='Trust coefficient in LARS')
    parser.add_argument('--beta-2', type=float, default=0.999,
                       help='Beta-2 in LAMB')

    valid_args = {action.option_strings[0] for action in parser._actions if action.option_strings}
    unrecognized_args = {arg for arg in sys.argv[1:] if arg.startswith("--") and arg not in valid_args}
    if unrecognized_args:
        raise ValueError(f"Unrecognized arguments found: {unrecognized_args}")


    args, uknown = parser.parse_known_args()

    model, train_loader, test_loader, device = train(args)

