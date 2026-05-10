import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from args import create_parser
from attacks import PGD, TRADES, rslad_inner_loss, adaad_inner_loss
from status import ProgressBar
from utils import load_dataset, load_student, load_teacher, get_student_name

try:
    import wandb
except ImportError:
    wandb = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = create_parser()
args = parser.parse_args()
assert args.method in ["pgd", "trades", "ard", "rslad", "iad", "adaad", "igdm"]
print(args)

set_seed(args.seed)

ad_methods = ["ard", "rslad", "iad", "adaad", "igdm"]
student_name = get_student_name(args)
teacher_name = "tiny_linf_wrn28-10" if args.dataset == "tinyimg" else args.teacher

if args.method in ad_methods:
    save_dir = os.path.join(args.save_root, args.dataset, f"train_eps{int(args.eps)}",
                            student_name, args.method, teacher_name)
else:
    save_dir = os.path.join(args.save_root, args.dataset, f"train_eps{int(args.eps)}",
                            student_name, args.method)

os.makedirs(save_dir, exist_ok=True)
best_path = os.path.join(save_dir, f"seed{args.seed}_best.pt")
final_path = os.path.join(save_dir, f"seed{args.seed}.pt")

if not args.nowand:
    assert wandb is not None, "Wandb is not installed. Use --nowand 1 or install wandb."
    wandb_tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               config=vars(args), name=args.wandb_name, tags=wandb_tags)

trainloader, testloader = load_dataset(args.dataset, args.batch)

student = load_student(args.student, args.dataset, args.depth, args.widen_factor)
student.train()

teacher = None
if args.method in ad_methods:
    teacher = load_teacher(args.teacher, args.dataset).cuda()
    teacher.eval()

optimizer = torch.optim.SGD(student.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.wd)

criterion_ce = nn.CrossEntropyLoss()
criterion_kl = nn.KLDivLoss(reduction="batchmean")
criterion_kl_none = nn.KLDivLoss(reduction="none")
progress_bar = ProgressBar()

eps = args.eps / 255.0
attack_alpha = eps / 4.0
best_robust_acc = 0.0

for epoch in range(1, args.epochs + 1):
    student.train()
    train_clean_correct, train_robust_correct, train_total = 0, 0, 0
    epoch_loss_sum = 0.0

    for step, (X, y) in enumerate(trainloader):
        X, y = X.cuda().float(), y.cuda()

        if args.method in ["pgd", "ard", "iad"]:
            X_adv = PGD(X, y, student, eps=eps, alpha=attack_alpha,
                        steps=args.train_pgd_steps, random_start=True)

        elif args.method == "trades":
            X_adv = TRADES(X, y, student, eps=eps, alpha=attack_alpha,
                           steps=args.train_pgd_steps)

        elif args.method == "rslad":
            with torch.no_grad():
                teacher_logits_clean = teacher(X)
            X_adv = rslad_inner_loss(X, y, student, teacher_logits_clean,
                                     eps=eps, alpha=attack_alpha,
                                     steps=args.train_pgd_steps, random_start=True)

        elif args.method in ["adaad", "igdm"]:
            X_adv = adaad_inner_loss(X, y, student, teacher,
                                     eps=eps, alpha=attack_alpha,
                                     steps=args.train_pgd_steps, random_start=True)

        optimizer.zero_grad()

        if args.method == "pgd":
            student_logits = student(X_adv)
            loss = criterion_ce(student_logits, y)

        elif args.method == "trades":
            clean_logits = student(X)
            adv_logits = student(X_adv)
            loss_ce = criterion_ce(clean_logits, y)
            loss_kl = criterion_kl(F.log_softmax(adv_logits, dim=1),
                                   F.softmax(clean_logits.detach(), dim=1))
            loss = loss_ce + 6.0 * loss_kl
            student_logits = adv_logits

        elif args.method in ["ard", "adaad"]:
            with torch.no_grad():
                teacher_logits = teacher(X_adv)
            student_logits = student(X_adv)
            loss = criterion_kl(F.log_softmax(student_logits, dim=1),
                                F.softmax(teacher_logits, dim=1))

        elif args.method == "rslad":
            with torch.no_grad():
                teacher_clean = teacher(X)
            student_logits = student(X_adv)
            loss = criterion_kl(F.log_softmax(student_logits, dim=1),
                                F.softmax(teacher_clean, dim=1))

        elif args.method == "iad":
            warm_up_epoch = 60 if args.dataset == "cifar10" else 80

            with torch.no_grad():
                teacher_clean = teacher(X)
                guide = teacher(X_adv)

            student_adv = student(X_adv)
            loss = criterion_kl(F.log_softmax(student_adv, dim=1),
                                F.softmax(teacher_clean, dim=1))

            if epoch >= warm_up_epoch:
                alpha_weight = F.softmax(guide, dim=1).gather(1, y.view(-1, 1)).squeeze()
                alpha_weight = alpha_weight.pow(0.1)

                student_clean = student(X)
                self_kl = criterion_kl_none(
                    F.log_softmax(student_adv, dim=1),
                    F.softmax(student_clean, dim=1)
                ).sum(dim=1)

                loss = loss + (self_kl * (1.0 - alpha_weight)).mean()

            student_logits = student_adv

        elif args.method == "igdm":
            delta = X_adv - X

            with torch.no_grad():
                teacher_plus = teacher(X + delta)
                teacher_minus = teacher(X - delta)

            student_plus = student(X + delta)
            student_minus = student(X - delta)

            kl_loss = criterion_kl(F.log_softmax(student_plus, dim=1),
                                   F.softmax(teacher_plus, dim=1))
            diff_loss = criterion_kl(
                F.log_softmax(student_plus - student_minus, dim=1),
                F.softmax((teacher_plus - teacher_minus).detach(), dim=1)
            )

            loss = kl_loss + 1.0 * (epoch / args.epochs) * diff_loss
            student_logits = student_plus

        loss.backward()
        optimizer.step()
        epoch_loss_sum += loss.item()

        with torch.no_grad():
            clean_logits = student(X)
            train_clean_correct += (clean_logits.argmax(1) == y).sum().item()
            train_robust_correct += (student_logits.argmax(1) == y).sum().item()
            train_total += y.size(0)

        progress_bar.prog(step, len(trainloader), epoch, loss.item())

    avg_epoch_loss = epoch_loss_sum / len(trainloader)
    train_clean_acc = train_clean_correct / train_total
    train_robust_acc = train_robust_correct / train_total

    if epoch in [100, 150]:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.1

    student.eval()
    clean_correct, robust_correct, total = 0, 0, 0

    for X, y in testloader:
        X, y = X.cuda().float(), y.cuda()
        X_adv = PGD(X, y, student, eps=eps, alpha=attack_alpha,
                    steps=args.eval_pgd_steps, random_start=True)
        student.eval()
        with torch.no_grad():
            logits = student(X)
            logits_adv = student(X_adv)

        clean_correct += (logits.argmax(1) == y).sum().item()
        robust_correct += (logits_adv.argmax(1) == y).sum().item()
        total += y.size(0)

    clean_acc = clean_correct / total
    robust_acc = robust_correct / total

    print(f"\nEpoch {epoch} | Clean Acc: {clean_acc:.4f} | "
          f"PGD{args.eval_pgd_steps} Acc: {robust_acc:.4f} | "
          f"Train Clean: {train_clean_acc:.4f} | "
          f"Train Robust: {train_robust_acc:.4f} | Loss: {avg_epoch_loss:.4f}")

    if not args.nowand:
        wandb.log({
            "main_epoch": epoch,
            "clean_acc": clean_acc,
            "robust_acc": robust_acc,
            "train_clean_acc": train_clean_acc,
            "train_robust_acc": train_robust_acc,
            "avg_epoch_loss": avg_epoch_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })

    if robust_acc > best_robust_acc:
        best_robust_acc = robust_acc
        torch.save(student.state_dict(), best_path)
        print(f"--- New best model saved: {best_path} | robust acc: {best_robust_acc:.4f} ---")

torch.save(student.state_dict(), final_path)
print(f"Final model saved to: {final_path}")
print(f"Best PGD{args.eval_pgd_steps} robust acc: {best_robust_acc:.4f}")

if not args.nowand:
    wandb.finish()