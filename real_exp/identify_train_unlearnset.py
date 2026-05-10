import os
import json
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from args import create_parser
from attacks import PGD
from utils import load_student, get_student_name


SPLIT_GROUPS_BY_DATASET = {
    "cifar10": [
        ("pgd_at", "PGD-AT", "pgd", None),
        ("trades", "TRADES", "trades", None),
        ("chen", "Chen", "ard", "Chen2021LTD_WRN34_10"),
        ("rebuffi", "Rebuffi", "ard", "Rebuffi2021Fixing_70_16_cutmix_extra"),
        ("bartoldson", "Bartoldson", "ard", "Bartoldson2024Adversarial_WRN-94-16"),
        ("gowal", "Gowal", "ard", "Gowal2021Improving_28_10_ddpm_100m"),
    ],
    "cifar100": [
        ("pgd_at", "PGD-AT", "pgd", None),
        ("trades", "TRADES", "trades", None),
        ("chen", "Chen", "ard", "Chen2021LTD_WRN34_10"),
        ("wang28", "Wang28", "ard", "Wang2023Better_WRN-28-10"),
        ("wang70", "Wang70", "ard", "Wang2023Better_WRN-70-16"),
        ("gowal", "Gowal", "ard", "Gowal2020Uncovering_extra"),
    ],
    "tinyimg": [
        ("pgd_at", "PGD-AT", "pgd", None),
        ("trades", "TRADES", "trades", None),
        ("wang", "Wang", "ard", "tiny_linf_wrn28-10"),
    ],
}

SPLIT_GROUP_KEYS = sorted({
    group_key
    for groups in SPLIT_GROUPS_BY_DATASET.values()
    for group_key, _, _, _ in groups
})


def load_clean_trainset(dataset):
    transform = transforms.Compose([transforms.ToTensor()])

    if dataset == "cifar10":
        return torchvision.datasets.CIFAR10(root="../dataset/", train=True, download=True, transform=transform)

    if dataset == "cifar100":
        return torchvision.datasets.CIFAR100(root="../dataset/", train=True, download=True, transform=transform)

    if dataset == "tinyimg":
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

        return TinyImageNet("train", transform)

    raise ValueError(f"Unknown dataset: {dataset}")


def get_split_groups(dataset):
    if dataset not in SPLIT_GROUPS_BY_DATASET:
        raise ValueError(f"Unknown dataset: {dataset}")
    return SPLIT_GROUPS_BY_DATASET[dataset]


def get_group(group_key, dataset):
    for group in get_split_groups(dataset):
        if group[0] == group_key:
            return group
    raise ValueError(f"Unknown split group for {dataset}: {group_key}")


def checkpoint_path(args, method, seed, teacher):
    student_name = get_student_name(args)
    base_dir = os.path.join(args.save_root, args.dataset, f"train_eps{int(args.eps)}", student_name)

    if method in ["pgd", "trades"]:
        return os.path.join(base_dir, method, f"seed{seed}_best.pt")

    teacher_name = "tiny_linf_wrn28-10" if args.dataset == "tinyimg" else teacher
    return os.path.join(base_dir, method, teacher_name, f"seed{seed}_best.pt")


def output_dir(args):
    student_name = get_student_name(args)
    return os.path.join(args.unlearnset_out_root, args.dataset, f"eps{int(args.eps)}", student_name)


def score_to_indices(score, num_models):
    return {str(k): np.where(score == k)[0].tolist() for k in range(num_models + 1)}


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def evaluate_checkpoint(args, path, loader, num_samples):
    model = load_student(args.student, args.dataset, args.depth, args.widen_factor)
    state_dict = torch.load(path, map_location="cuda", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    robust_correct = np.zeros(num_samples, dtype=np.bool_)
    eps = args.eps / 255.0
    alpha = eps / 4.0
    cur = 0

    for x, y in tqdm(loader, desc=os.path.basename(path), leave=False):
        x, y = x.cuda().float(), y.cuda()
        batch_size = x.size(0)

        x_adv = PGD(x, y, model, eps=eps, alpha=alpha,
                    steps=args.eval_pgd_steps, random_start=True)

        model.eval()
        with torch.no_grad():
            pred = model(x_adv).argmax(1)

        robust_correct[cur:cur + batch_size] = (pred == y).cpu().numpy()
        cur += batch_size

    del model
    torch.cuda.empty_cache()
    return robust_correct


def eval_one(args):
    group_key, group_name, method, teacher = get_group(args.split_group, args.dataset)

    trainset = load_clean_trainset(args.dataset)
    loader = DataLoader(trainset, batch_size=args.batch, shuffle=False, num_workers=4)
    num_samples = len(trainset)

    out_dir = output_dir(args)
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    ckpt = checkpoint_path(args, method, args.split_seed, teacher)

    if not os.path.exists(ckpt):
        print(f"[Skip] checkpoint not found: {ckpt}")
        return

    print(f"[Eval] group={group_name}, seed={args.split_seed}")
    print(f"[Load] {ckpt}")

    correct = evaluate_checkpoint(args, ckpt, loader, num_samples)
    save_path = os.path.join(cache_dir, f"{group_key}_seed{args.split_seed}.npy")
    np.save(save_path, correct.astype(np.uint8))

    print(f"[Saved] {save_path}")


def merge_results(args):
    trainset = load_clean_trainset(args.dataset)
    num_samples = len(trainset)

    out_dir = output_dir(args)
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(out_dir, exist_ok=True)

    total_score = np.zeros(num_samples, dtype=np.int32)
    total_models = 0
    missing_cache = []

    summary = {
        "dataset": args.dataset,
        "student": get_student_name(args),
        "epsilon": args.eps,
        "eval_attack": f"PGD-{args.eval_pgd_steps}",
        "num_samples": num_samples,
        "requested_seeds": args.split_seeds,
        "groups": {},
    }

    for group_key, group_name, method, teacher in get_split_groups(args.dataset):
        print(f"\n========== Merge group: {group_name} ==========")

        group_score = np.zeros(num_samples, dtype=np.int32)
        used_seeds = []

        for seed in args.split_seeds:
            path = os.path.join(cache_dir, f"{group_key}_seed{seed}.npy")

            if not os.path.exists(path):
                print(f"[Skip] cached result not found: {path}")
                missing_cache.append(path)
                continue

            correct = np.load(path).astype(np.int32)
            group_score += correct
            total_score += correct
            used_seeds.append(seed)

        num_group_models = len(used_seeds)

        if num_group_models == 0:
            print(f"[Skip] no cached results for group: {group_name}")
            summary["groups"][group_name] = {
                "method": method,
                "teacher": teacher,
                "num_models": 0,
                "used_seeds": [],
                "learnable_count": None,
                "unlearnable_count": None,
            }
            continue

        total_models += num_group_models

        learnable = np.where(group_score == num_group_models)[0]
        unlearnable = np.where(group_score == 0)[0]

        group_result = {
            "group": group_name,
            "method": method,
            "teacher": teacher,
            "num_models": num_group_models,
            "used_seeds": used_seeds,
            "num_samples": num_samples,
            "learnable_count": int(len(learnable)),
            "unlearnable_count": int(len(unlearnable)),
            "learnable_indices": learnable.tolist(),
            "unlearnable_indices": unlearnable.tolist(),
            "robust_score_to_indices": score_to_indices(group_score, num_group_models),
        }

        group_file = os.path.join(out_dir, f"{group_key}.json")
        save_json(group_file, group_result)
        np.save(os.path.join(out_dir, f"{group_key}_score.npy"), group_score)

        summary["groups"][group_name] = {
            "method": method,
            "teacher": teacher,
            "num_models": num_group_models,
            "used_seeds": used_seeds,
            "learnable_count": int(len(learnable)),
            "unlearnable_count": int(len(unlearnable)),
            "json": group_file,
        }

        print(f"{group_name}: models={num_group_models}, learnable={len(learnable)}, unlearnable={len(unlearnable)}")

    if total_models == 0:
        raise RuntimeError("No cached evaluation results found. Run with --split_mode eval first.")

    learnable = np.where(total_score == total_models)[0]
    unlearnable = np.where(total_score == 0)[0]

    train_unlearnset = {
        "dataset": args.dataset,
        "student": get_student_name(args),
        "epsilon": args.eps,
        "eval_attack": f"PGD-{args.eval_pgd_steps}",
        "num_models": total_models,
        "num_samples": num_samples,
        "learnable_count": int(len(learnable)),
        "unlearnable_count": int(len(unlearnable)),
        "learnable_indices": learnable.tolist(),
        "unlearnable_indices": unlearnable.tolist(),
        "robust_score_to_indices": score_to_indices(total_score, total_models),
    }

    save_json(os.path.join(out_dir, "train_unlearnset.json"), train_unlearnset)
    np.save(os.path.join(out_dir, "learnable_indices.npy"), learnable)
    np.save(os.path.join(out_dir, "unlearnable_indices.npy"), unlearnable)
    np.save(os.path.join(out_dir, "total_score.npy"), total_score)

    summary["intersection"] = {
        "num_models": total_models,
        "learnable_count": int(len(learnable)),
        "unlearnable_count": int(len(unlearnable)),
        "json": os.path.join(out_dir, "train_unlearnset.json"),
    }
    summary["missing_cache"] = missing_cache

    save_json(os.path.join(out_dir, "summary.json"), summary)

    print("\n========== Final train unlearnset ==========")
    print(f"Total evaluated models: {total_models}")
    print(f"Learnable: {len(learnable)}")
    print(f"Unlearnable: {len(unlearnable)}")
    print(f"Missing cached results: {len(missing_cache)}")
    print(f"Saved to: {out_dir}")


def main():
    parser = create_parser()
    parser.add_argument("--split_mode", type=str, default="merge", choices=["eval", "merge"])
    parser.add_argument("--split_group", type=str, default="pgd_at", choices=SPLIT_GROUP_KEYS)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--split_seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--unlearnset_out_root", type=str, default="analysis_results/train_unlearnset")
    args = parser.parse_args()


    if args.split_mode == "eval":
        eval_one(args)
    elif args.split_mode == "merge":
        merge_results(args)


if __name__ == "__main__":
    main()
