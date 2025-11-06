

from torchvision.transforms import Compose,ToPILImage,ToTensor,Normalize,RandomHorizontalFlip,Resize

transform1 = Compose([ToTensor(),RandomHorizontalFlip()])
totensor = ToTensor()
resize = Resize([32,32])

transform_torch = {
    "CIFAR10":{
        "train":totensor,
        "test":totensor,
    },
    "CINIC10":{
        "train":totensor,
        "test":totensor,
    },
    "CIFAR100":{
        "train":transform1,
        "test":totensor,
    },
    "TinyImageNet":{
        "train":Compose([resize,totensor]),
        "test":Compose([resize,totensor]),
    },
    "FEMNIST":{
        "train":Compose([totensor]),
        "test":Compose([totensor]),
    },
    "CelebA":{
        "train":Compose([resize,totensor]),
        "test":Compose([resize,totensor]),
    },
    "Reddit":{
        "train":Compose([None]),
        "test":Compose([None]),
    },
    "Sent140":{
        "train":Compose([None]),
        "test":Compose([None]),
    },"Twitter":{

    },"IMDB":{
        "train":Compose([None]),
        "test":Compose([None]),
    },"Purchase100":{

    },"creditcard":{

    },"Loan":{
        
    },"Shakespeare":{
        "train":Compose([None]),
        "test":Compose([None]),
    }
}



