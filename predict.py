# Author: Elvis Gene
# Date: 23 - Jan - 2019

import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session
import json
from PIL import Image
import numpy as np
from torch.autograd import Variable

import train


in_parser= argparse.ArgumentParser(description='Classify flowers')

in_parser.add_argument('--checkpoints_file', dest="checkpoints_file", action="store", default="./part2_checkpoint.pth", type=str,
                    help="Directory to retrieve the checkpoint")
in_parser.add_argument('--topk', default=5, type=int, dest="topk", action="store")
in_parser.add_argument('--image', dest="image" default='./flowers/test/10/image_07090.jpg', action="store", type = str)
in_parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
in_parser.add_argument('--u_gpu', dest="u_gpu", action="store_true", default=False, help='Include GPU for training?')

in_parser = in_parser.parse_args()

k = in_parser.topk
checkpoint_dir = in_parser.checkpoints_file
util_gpu = in_parser.u_gpu
img = (in_parser.image)
cat_to_name = in_parser.category_names


with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    hidden_layers = checkpoint['hidden_layers']

    # Load a pretrained network.
    if train.structure is 'resnet18':
        model = models.resnet18(pretrained=True)

    # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(
             nn.Linear(checkpoint['input_size'], hidden_layers[0]),
             nn.ReLU(),
             nn.Dropout(train.dropout),
             nn.Linear(hidden_layers[0], hidden_layers[1]),
             nn.ReLU(),
             nn.Dropout(train.dropout),
             nn.Linear(hidden_layers[1], hidden_layers[2]),
             nn.LogSoftmax(dim =1))

        model.classifier = classifier

    else:
        model = models.vgg16(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        # Build a feed-forward network
        classifier = nn.Sequential(
            nn.Linear(checkpoint['input_size'], hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(train.dropout),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(train.dropout),
            nn.Linear(hidden_layers[1], hidden_layers[2]),
            nn.ReLU(),
            nn.Dropout(train.dropout),
            nn.Linear(hidden_layers[2], hidden_layers[3]),
            nn.ReLU(),
            nn.Dropout(train.dropout),
            nn.Linear(hidden_layers[3], checkpoint['output_size']),
            nn.LogSoftmax(dim=1))

        model.classifier = classifier

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['model_class_to_idx']

    print(model)
    return model


def prediction(image, model, topk=5):

    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # DONE: Implement the code to predict the class from an image file
    model.to('cuda')
    image =  Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0)
    image = image.cuda()
    result = model(image).topk(topk)
    probs = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
    classes = result[1].data.cpu().numpy()[0]
    idx_to_class = { v : k for k,v in model.class_to_idx.items()}
    topk_class = [idx_to_class[x] for x in classes]

    return probs, topk_class


def get_topk(image_path, model, topk=5):

   image = process_image(image_path)
   probs, topk_classes  = prediction(image, model, topk)

   labels = [cat_to_name [name] for name in topk_classes]

   j = 0
   while j < topk:
        print("{} - {} : {:.2f}%".format(j+1, labels[j], probs[j] * 100))
        j += 1


def process_image(image_path):
    pil_img = Image.open(image_path)

    img_mod = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])

    np_image = img_mod(pil_img)
    return np_image


def main():
    model = load_checkpoint('part2_checkpoint.pth')
    print(model)

    get_topk(img, model, 5)
    print('')
    print('Only One Image This Time')
    get_topk(img, model, 1)


if __name__ == '__main__':
    main()
