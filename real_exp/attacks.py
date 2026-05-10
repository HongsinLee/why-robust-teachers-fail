import torch
import torch.nn as nn
import torch.nn.functional as F


def _set_bn_dropout_eval(model):
    for _, m in model.named_modules():
        if "BatchNorm" in m.__class__.__name__:
            m.eval()
        if "Dropout" in m.__class__.__name__:
            m.eval()


def _linf_random_start(images, eps):
    adv_images = images + torch.empty_like(images).uniform_(-eps, eps)
    return torch.clamp(adv_images, min=0, max=1).detach()


def _linf_project(images, adv_images, eps):
    delta = torch.clamp(adv_images - images, min=-eps, max=eps)
    return torch.clamp(images + delta, min=0, max=1).detach()


def PGD(images, labels, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
    model.eval()

    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = _linf_random_start(images, eps)

    criterion = nn.CrossEntropyLoss()

    for _ in range(steps):
        adv_images.requires_grad = True
        logits = model(adv_images)
        loss = criterion(logits, labels)

        grad = torch.autograd.grad(
            loss, adv_images, retain_graph=False, create_graph=False
        )[0]

        adv_images = adv_images.detach() + alpha * grad.sign()
        adv_images = _linf_project(images, adv_images, eps)

    model.train()
    return adv_images


def TRADES(images, labels, model, eps=8 / 255, alpha=2 / 255, steps=10):
    model.train()
    _set_bn_dropout_eval(model)

    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    with torch.no_grad():
        logits_clean = model(images)

    adv_images = images + 0.001 * torch.randn_like(images)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    criterion_kl = nn.KLDivLoss(reduction="sum")

    for _ in range(steps):
        adv_images.requires_grad = True
        logits_adv = model(adv_images)

        loss = criterion_kl(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_clean, dim=1),
        )

        grad = torch.autograd.grad(
            loss, adv_images, retain_graph=False, create_graph=False
        )[0]

        adv_images = adv_images.detach() + alpha * grad.sign()
        adv_images = _linf_project(images, adv_images, eps)

    model.train()
    return adv_images


def rslad_inner_loss(images, labels, model, teacher_logits, eps=8 / 255, alpha=2 / 255,
                     steps=10, random_start=True):
    model.train()
    _set_bn_dropout_eval(model)

    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()
    teacher_logits = teacher_logits.detach()

    adv_images = images.clone().detach()
    if random_start:
        adv_images = _linf_random_start(images, eps)

    criterion_kl = nn.KLDivLoss(reduction="batchmean")

    for _ in range(steps):
        adv_images.requires_grad = True
        student_logits = model(adv_images)

        loss = criterion_kl(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
        )

        grad = torch.autograd.grad(
            loss, adv_images, retain_graph=False, create_graph=False
        )[0]

        adv_images = adv_images.detach() + alpha * grad.sign()
        adv_images = _linf_project(images, adv_images, eps)

    model.train()
    return adv_images


def adaad_inner_loss(images, labels, model, teacher, eps=8 / 255, alpha=2 / 255,
                     steps=10, random_start=True):
    model.train()
    _set_bn_dropout_eval(model)

    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    adv_images = images.clone().detach()
    if random_start:
        adv_images = _linf_random_start(images, eps)

    criterion_kl = nn.KLDivLoss(reduction="batchmean")

    for _ in range(steps):
        adv_images.requires_grad = True
        delta = adv_images - images

        with torch.no_grad():
            teacher_plus = teacher(images + delta)
            teacher_minus = teacher(images - delta)

        student_plus = model(adv_images)
        student_minus = model(images - delta)

        loss = criterion_kl(
            F.log_softmax(student_plus - student_minus, dim=1),
            F.softmax((teacher_plus - teacher_minus).detach(), dim=1),
        )

        grad = torch.autograd.grad(
            loss, adv_images, retain_graph=False, create_graph=False
        )[0]

        adv_images = adv_images.detach() + alpha * grad.sign()
        adv_images = _linf_project(images, adv_images, eps)

    model.train()
    return adv_images
