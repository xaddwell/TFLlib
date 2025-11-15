


import os
import sys
import json
import inspect
import torch
import logging
from os import path
import transformers

from .basehead import BaseHeadSplit

from .resnet import ResNet18,ResNet34,ResNet50,ResNet101,ResNet152
from .vgg import VGG11,VGG11BN,VGG13,VGG13BN,VGG9,VGG9BN

from .harcnn import HARCNN

from .deepspeech import DeepSpecch
from .inception3 import Inceptionv3

from .lenet import LeNet
from .simplecnn import SimpleCNN
from .rnn import RNN
from .sent140lstm import Sent140LSTM
from .stackedlstm import StackedLSTM
from .logreg import LogReg
from .shufflenet import ShuffleNet
from .squeezenet import SqueezeNet
from .squeezenext import SqueezeNeXt

from .m5 import M5
from .mobilenet import MobileNet
from .mobilenext import MobileNeXt
from .mobilebert import MobileBert
from .albert import ALBERT
from .electra import ELECTRA
from .tinybert import TinyBert
from .minilm import MiniLM



from flcore.fedatasets import dataset_dict
logger = logging.getLogger(__name__)



model_dict = {
    'resnet18':ResNet18,'resnet34':ResNet34,'resnet50':ResNet50,'resnet101':ResNet101,'resnet152':ResNet152,'vgg11':VGG11,'vgg11bn':VGG11BN,'vgg13':VGG13,'vgg13bn':VGG13BN,'vgg9':VGG9,'vgg9bn':VGG9BN,'lenet':LeNet,'simplecnn':SimpleCNN,'inceptionv3':Inceptionv3,'shufflenet':ShuffleNet,'squeezenet':SqueezeNet,'squeezenext':SqueezeNeXt,'mobilenet':MobileNet,'mobilenext':MobileNeXt,
    'harcnn':HARCNN,'logreg':LogReg,'deepspecch':DeepSpecch,
    'rnn':RNN,'sent140lstm':Sent140LSTM, 'stackedlstm':StackedLSTM,'m5':M5,
    'mobilebert':MobileBert,'tinybert':TinyBert, 'albert':ALBERT,'electra':ELECTRA,'minilm':MiniLM
    }


cv_model = {'resnet18','resnet34','resnet50','resnet101','resnet152','vgg11','vgg11bn','vgg13','vgg13bn','vgg9','vgg9bn','inceptionv3','lenet','simplecnn','shufflenet','squeezenet','squeezenext','mobilenet','mobilenext'}

domain_model = {"nlp":{'sent140lstm','stackedlstm',"rnn","albert","mobilebert","tinybert","electra",'minilm'},"tabular":{'logreg','harcnn'},"cv":cv_model}




def load_model(args):
    # retrieve model skeleton
    model_class = model_dict[args.model_name]
    dataset_args = dataset_dict[args.data_name]
    dataset_domain = dataset_args["domain"]
    args = check_args(args,dataset_domain)
    # get required model arguments
    required_args = inspect.getargspec(model_class)[0]
    # collect eneterd model arguments
    model_args = {}
    for argument in required_args:
        if argument != 'self':
            if argument in dataset_args.keys():
                model_args[argument] = dataset_args[argument]
            else:
                try:
                    model_args[argument] = getattr(args, argument)
                except AttributeError:
                    print("can not get attribution {} from args".format(argument))
    # get model instance
    model = model_class(**model_args)
    # adjust arguments if needed
    args.start_round = 0
    if args.resume_ckpt:
        model.load_state_dict(torch.load(args.resume_ckpt))
        # model = torch.load(args.resume_ckpt)
        args.start_round = int(os.path.basename(args.resume_ckpt).split('.')[-2].split("_")[-1])
    
    return model, args


def check_args(args,dataset_domain):
    data_name = args.data_name
    model_name = args.model_name
    assert data_name in dataset_dict.keys(),"{} datasets not surport yet, we surport datasets {}".format(data_name,dataset_dict.keys())
    assert model_name in model_dict.keys(),"{} model not surport yet, we surport models {}".format(model_name, model_dict.keys())
    assert model_name in domain_model[dataset_domain], "{} data do not match with model {}, suggestion:{}".format(dataset_domain, model_name, domain_model[dataset_domain])

    if args.resume_ckpt:
        if not os.path.exists(args.resume_ckpt):
            #args.resume_ckpt=None
            raise ValueError((f"model {model_name} checkpoints path {args.resume_ckpt} do not exist"))
    return args

