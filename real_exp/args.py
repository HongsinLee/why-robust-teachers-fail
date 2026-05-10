from argparse import ArgumentParser


def create_parser():
    parser = ArgumentParser(description="Robust teacher failure experiments", allow_abbrev=False)

    parser.add_argument("--method", type=str, default="ard",
                        choices=["pgd", "trades", "ard", "rslad", "iad", "adaad", "igdm"])
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "tinyimg"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--student", type=str, default="RES-18",
                        choices=["RES-18", "MN-V2", "WRN"])
    parser.add_argument("--teacher", type=str, default="Chen2021LTD_WRN34_10",
                        help="RobustBench teacher model name")
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--widen_factor", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=2e-4)

    parser.add_argument("--eps", type=float, default=8,
                        help="epsilon numerator, e.g., eps=8 means 8/255")
    parser.add_argument("--train_pgd_steps", type=int, default=10)
    parser.add_argument("--eval_pgd_steps", type=int, default=20)

    parser.add_argument("--save_root", type=str, default="result_models")
    parser.add_argument("--nowand", type=int, default=1, choices=[0, 1])
    parser.add_argument("--wandb_entity", type=str, default="your_wandb_entity")
    parser.add_argument("--wandb_project", type=str, default="robust-teachers-fail")
    parser.add_argument("--wandb_name", type=str, default="debug")
    parser.add_argument("--wandb_tags", type=str, default="NoTag")

    return parser
