import os

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from robustbench.utils import load_model


def load_dataset(dataset, batch_size, shuffle_train=True, shuffle_test=False):
    if dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([transforms.ToTensor()])

        trainset = torchvision.datasets.CIFAR10(
            root="../dataset/", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root="../dataset/", train=False, download=True, transform=transform_test
        )

    elif dataset == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([transforms.ToTensor()])

        trainset = torchvision.datasets.CIFAR100(
            root="../dataset/", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root="../dataset/", train=False, download=True, transform=transform_test
        )

    elif dataset == "tinyimg":
        class TinyImageNet(Dataset):
            def __init__(self, split, transform=None):
                root = "../dataset/tiny-imagenet-200/"
                self.dataset = torchvision.datasets.ImageFolder(os.path.join(root, split))
                self.transform = transform

            def __getitem__(self, index):
                img, target = self.dataset[index]
                if self.transform is not None:
                    img = self.transform(img)
                return img, target

            def __len__(self):
                return len(self.dataset)

        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([transforms.ToTensor()])

        trainset = TinyImageNet("train", transform_train)
        testset = TinyImageNet("val", transform_test)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle_train, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=shuffle_test, num_workers=2
    )

    return trainloader, testloader


def load_student(student_name, dataset, depth=32, widen_factor=10):
    if dataset == "cifar10":
        from cifar10_models import mobilenet_v2, resnet18, wideresnet

        if student_name == "RES-18":
            student = resnet18()
        elif student_name == "MN-V2":
            student = mobilenet_v2()
        elif student_name == "WRN":
            student = wideresnet(
                depth=depth,
                num_classes=10,
                widen_factor=widen_factor,
                dropRate=0.0,
            )
        else:
            raise ValueError(f"Unknown student model: {student_name}")

    elif dataset == "cifar100":
        from cifar100_models import mobilenet_v2, resnet18
        from cifar10_models import wideresnet

        if student_name == "RES-18":
            student = resnet18()
        elif student_name == "MN-V2":
            student = mobilenet_v2()
        elif student_name == "WRN":
            student = wideresnet(
                depth=depth,
                num_classes=100,
                widen_factor=widen_factor,
                dropRate=0.0,
            )
        else:
            raise ValueError(f"Unknown student model: {student_name}")

    elif dataset == "tinyimg":
        from cifar100_models import pResNet18

        if student_name != "RES-18":
            raise ValueError("Only RES-18 is supported for Tiny-ImageNet.")
        student = pResNet18(num_classes=200)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    student = torch.nn.DataParallel(student)
    student = student.cuda()

    return student


def load_teacher(teacher_name, dataset):
    if dataset in ["cifar10", "cifar100"]:
        teacher = load_model(
            model_name=teacher_name,
            dataset=dataset,
            threat_model="Linf",
        )

    elif dataset == "tinyimg":
        ckpt_path = "models/tiny_linf_wrn28-10.pt"

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Tiny-ImageNet teacher checkpoint not found: {ckpt_path}\n"
                "Please download `tiny_linf_wrn28-10.pt` from the official DM-Improves-AT repository:\n"
                "  https://github.com/wzekai/DM-Improves-AT\n"
                "Direct checkpoint link:\n"
                "  https://huggingface.co/wzekai99/DM-Improves-AT1/resolve/main/checkpoint/tiny_linf_wrn28-10.pt\n"
                f"Then place it at: {ckpt_path}"
            )

        from cifar100_models import ti_wideresnetwithswish

        teacher = ti_wideresnetwithswish(num_classes=200)
        teacher = torch.nn.Sequential(teacher)
        teacher = torch.nn.DataParallel(teacher)

        checkpoint = torch.load(ckpt_path)
        teacher.load_state_dict(checkpoint["model_state_dict"])

    else:
        raise ValueError(f"Unknown dataset for teacher loading: {dataset}")

    return teacher


def get_student_name(args):
    if args.student == "WRN":
        return f"WRN-{args.depth}-{args.widen_factor}"
    return args.student