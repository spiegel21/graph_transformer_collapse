import argparse
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
from tqdm.auto import tqdm
from collections import OrderedDict
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')

def expand_channels(x):
    return x.repeat(3, 1, 1) if x.size(0) == 1 else x

def main(mode, cont=False):
    if(mode == 'vit_small'):
        transform = transforms.Compose([
            transforms.ToTensor(),
            expand_channels
        ])
        batch_size=128
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        batch_size = 128
    n_epochs = 200
    trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='../data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    if(mode == 'vit_small'):
        model = torchvision.models.VisionTransformer(
            image_size=28,
            patch_size=4,
            num_layers=12,
            num_heads=8,
            hidden_dim=288,
            mlp_dim=1152,
            num_classes=10
        )
    else:
        model = torchvision.models.resnet18(pretrained=False)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        model = model.to(device)
    

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=n_epochs)
    criterion = torch.nn.CrossEntropyLoss()
    
    write_mode = 'w'
    start_epoch = 0

    if(cont):
        log_lines = open(f'logs/{mode}_train_log.txt').readlines()
        for line in np.flip(log_lines):
            if('Epoch' in line):
                start_epoch = int(line.split()[-1])
                if(start_epoch % 5 == 0):
                    break
        loaded_state = torch.load(f'models/{mode}/epoch_{start_epoch}.pth')
        model.load_state_dict(loaded_state)
        write_mode = 'a'

        start_epoch += 1

        for i in range(start_epoch):
            scheduler.step()

        print(f'Starting from epoch {start_epoch}')

    model = model.to(device)

    sys.stdout = open(f'logs/{mode}_train_log.txt', write_mode)
    for epoch in (range(start_epoch, start_epoch +n_epochs)):
        tqdm.write(f'Epoch {epoch}')
        model.train()
        loss_list = []
        acc_list = []
        for batch_idx, (data, target) in enumerate(tqdm(trainloader)):
            data = data.to(device)
            out = model(data)
            out_preds = torch.argmax(out, dim=1)
            target = target.to(device)

            loss = criterion(out, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            train_acc = (torch.mean((out_preds == target).float()).item())
            acc_list.append(train_acc)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss_list = []
            val_acc_list = []
            for batch_idx, (data, target) in enumerate(tqdm(testloader)):
                data = data.to(device).float()
                out = model(data)
                out_preds = torch.argmax(out, dim=1)
                target = target.to(device)
                loss = criterion(out, target)

                val_loss_list.append(loss.item())
                val_acc_list.append(torch.mean((out_preds == target).float()).item())

        tqdm.write(f'Train Loss: {np.mean(loss_list)}\nTrain Accuracy: {np.mean(acc_list)}\nVal Loss: {np.mean(val_loss_list)}\nVal Accuracy: {np.mean(val_acc_list)}\n')

        if((epoch < 20) or ((epoch % 5) == 0)):
            torch.save(model.state_dict(), f'models/{mode}/epoch_{epoch}.pth')

    sys.stdout.close()
    sys.stdout = sys.__stdout__

if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description="Run the model with specified mode.")
    parser.add_argument('mode', type=str, choices=['resnet18', 'vit_small'],
                        help='Model mode to run (either "resnet18" or "vit_small").')
    args = parser.parse_args()

    main(mode=args.mode)