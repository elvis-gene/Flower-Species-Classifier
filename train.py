# Author: Elvis Gene
# Date: 23 - Jan - 2020

import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session
import time


parser= argparse.ArgumentParser(description='Train a network')

parser.add_argument('--arch',  dest="arch", action="store", default="vgg16", type = str, help = 'Options: vgg16, resnet18')
parser.add_argument('--data_loc', dest="data_loc", type=str, action="store", default="./flowers/")
parser.add_argument('--checkpoints_loc', dest="checkpoints_loc", action="store", default="./", help="Directory to save checkpoints")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001, help='Learning rate')
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1, help='Number of epochs')
parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.3, help='Dropout value')
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=2304, help='Number of hidden units')
parser.add_argument('--use_gpu', dest="use_gpu", action="store_true", default=False, help='Include GPU for training?')

args = parser.parse_args()


structure = args.arch
data_loc = args.data_loc
checkpoints_dir = args.checkpoints_loc
learning_rate = args.learning_rate
dropout = args.dropout
epochs = args.epochs
hidden_u = args.hidden_units
use_gpu = args.use_gpu


def data_elements(data_loc):

    data_dir = data_loc
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {'train_transforms': transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]),

                          'test_transforms': transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),

                          'valid_transforms': transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])}

    # Done: Load the datasets with ImageFolder
    image_datasets = {'train_data': datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
                         'test_data': datasets.ImageFolder(test_dir, transform=data_transforms['test_transforms']),
                         'valid_data': datasets.ImageFolder(valid_dir, transform=data_transforms['valid_transforms'])}

    # Done: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'train_loader': torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=32, shuffle=False),
                    'test_loader': torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=32, shuffle=False),
                    'valid_loader':torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=32, shuffle=False)}

    return dataloaders, image_datasets


def def_network(structure, dropout, hidden_u, learing_rate, use_gpu):

    if structure == 'resnet18':
        model = models.resnet18(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict ([
                    ('fc1', nn.Linear (9216, hidden_u)),
                    ('relu1', nn.ReLU ()),
                    ('dropout1', nn.Dropout(dropout)),
                    ('fc2', nn.Linear (hidden_u, 1854)),
                    ('relu2', nn.ReLU()),
                    ('dropout2', nn.Dropout (dropout)),
                    ('fc3', nn.Linear (1854, 102)),
                    ('output', nn.LogSoftmax (dim =1))
                ]))

        model.classifier = classifier

    else:
        model = models.vgg16(pretrained=True)

        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False

            # Build a feed-forward network
        classifier = nn.Sequential(
            nn.Linear(25088, hidden_u),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_u, 1224),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1224, 684),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(684, 302),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(302, 102),
            nn.LogSoftmax(dim=1))

        model.classifier = classifier
    print(model)

    optimizer = optim.Adam(model.classifier.parameters(), learing_rate)
    criterion = criterion = nn.NLLLoss()

    return model, criterion, optimizer



# Train your network
def train_network(epochs, model, use_gpu, dataloaders, criterion, optimizer):


    print("Training...")
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)
    criterion = criterion = nn.NLLLoss()
    device = 'cuda'
    model.to(device)
    epochs = 6
    print_every = 5
    steps = 0
    valid_loss = 0
    accuracy=0

    with active_session():
        # do long-running work here
        start = time.time()

        for e in range(epochs):
            running_loss = 0
            for i, (inputs, labels) in enumerate(dataloaders['train_loader']):
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if steps % print_every == 0:
                    model.eval()
                    #  Training Loss and Training accuracy
                    with torch.no_grad():

                            outputs = model.forward(inputs)
                            train_loss = criterion(outputs, labels)
                            p = torch.exp(outputs).data
                            equal = (labels.data == p.max(dim=1)[1])
                            t_accuracy += equal.type(torch.FloatTensor).mean()

                    train_loss = train_loss / len(dataloaders['train_loader'])
                    t_accuracy = t_accuracy / len(dataloaders['train_loader'])

                    print("Epoch: {}/{}..".format(e + 1, epochs),
                          "Validation Loss: {:.3f}.. ".format(train_loss),
                          "Validation Accuracy: {:.3f}".format(t_accuracy))


                    # Validation Loss and Validation accuracy
            for iii, (inputs_v, labels_v) in enumerate(dataloaders['valid_loader']):

                inputs_v, labels_v = inputs_v.to('cuda:0'), labels_v.to('cuda:0')
                model.to('cuda:0')

                if steps % print_every == 0:
                    model.eval()

                    with torch.no_grad():
                        #optimizer.zero_grad() #no backward pass
                        outputs_v = model.forward(inputs_v)
                        valid_loss = criterion(outputs_v, labels_v)
                        ps = torch.exp(outputs_v).data
                        equality = (labels_v.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()

                    valid_loss = valid_loss / len(dataloaders['valid_loader'])
                    accuracy = accuracy / len(dataloaders['valid_loader'])

                    print("Epoch: {}/{}..".format(e + 1, epochs),
                          "Validation Loss: {:.3f}.. ".format(valid_loss),
                          "Validation Accuracy: {:.3f}".format(accuracy))

                    model.train() #To check latter if it should also be put on top...

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Done training and Testing!")
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))



# path='checkpoint.pth',structure ='densenet121', hidden_layer1=120,dropout=0.5,lr=0.001,epochs=12
def save_checkpoints(hidden_u, checkpoints_dir, image_datasets, model):
    # Done: Save the checkpoint
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'hidden_layers': [hidden_u, 1224, 684, 302],
                  'model_class_to_idx': image_datasets['train_data'].class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, checkpoints_dir + 'part2_checkpoint.pth')

    print("Checkpoint saved!")

def main():
    dataloaders, image_datasets = data_elements(data_loc)
    model, criterion, optimizer = def_network(structure, dropout, hidden_u, learning_rate, use_gpu)
    train_network(epochs, model, use_gpu, dataloaders, criterion, optimizer)
    save_checkpoints(hidden_u, checkpoints_dir, image_datasets, model)


if __name__ == '__main__':
    main()
