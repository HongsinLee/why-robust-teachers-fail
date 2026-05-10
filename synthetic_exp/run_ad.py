import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------- 1) Arguments ----------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic AD experiment.")

    # Data
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--p", type=int, default=4, help="Number of patches.")
    parser.add_argument(
        "--p_un",
        type=float,
        default=None,
        help=(
            "Fraction of unlearnable samples. "
            "If omitted, runs 0.00, 0.05, 0.10, 0.15, 0.20."
        ),
    )
    parser.add_argument(
        "--p_un_list",
        type=float,
        nargs="+",
        default=None,
        help="List of unlearnable fractions to run.",
    )
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--sigma_n", type=float, default=0.4)

    # Model
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--m", type=int, default=80)
    parser.add_argument("--q", type=int, default=3)
    parser.add_argument("--sigma0_scale", type=float, default=0.1)

    # Teacher
    parser.add_argument(
        "--teacher",
        type=str,
        default=None,
        choices=["good", "bad"],
        help="Teacher type. If omitted, runs both good and bad teachers.",
    )
    parser.add_argument(
        "--teacher_margin",
        type=float,
        default=10.0,
        help="Teacher logit margin for confident predictions.",
    )

    # Training
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # Evaluation
    parser.add_argument("--pgd_steps", type=int, default=20)

    # Output
    parser.add_argument("--out_dir", type=str, default="synthetic_exp/results/ad")
    parser.add_argument("--save_diagnostic_plot", action="store_true")

    return parser.parse_args()


def make_run_dir(out_dir, teacher, p_un, seed):
    run_name = f"pun_{p_un:.2f}_seed_{seed}"
    run_dir = Path(out_dir) / teacher / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------- 2) Data ----------------
def generate_features(d, u_idx, v_idx):
    """Return learnable feature u and unlearnable feature v."""
    u = np.zeros(d)
    v = np.zeros(d)
    u[u_idx] = 1.0
    v[v_idx] = 1.0
    return u, v


def generate_data(N, P, d, u, v, alpha, sigma_n, p_un, u_idx, v_idx):
    """
    Generate patch-based synthetic data.

    Learnable samples contain alpha * y * u.
    Unlearnable samples contain alpha * y * v.
    The student cannot use v because its v-coordinate is projected to zero.
    """
    X_data, y_data, is_unlearnable = [], [], []

    for _ in range(N):
        y = np.random.randint(2)
        sign = 1 if y == 1 else -1

        X = np.zeros((P, d))
        sample_unlearnable = np.random.rand() < p_un
        signal_patch = np.random.randint(P)

        for p in range(P):
            patch = np.zeros(d)

            if p == signal_patch:
                if sample_unlearnable:
                    patch += alpha * sign * v
                else:
                    patch += alpha * sign * u

            # Noise is orthogonal to both u and v.
            noise = np.zeros(d)
            noise[u_idx + 1 : v_idx] = np.random.randn(d - 2) * sigma_n
            patch += noise

            X[p] = patch

        X_data.append(X)
        y_data.append(y)
        is_unlearnable.append(sample_unlearnable)

    return X_data, np.array(y_data), np.array(is_unlearnable, dtype=bool)


# ---------------- 3) Model and teacher ----------------
def relu_power_q(z, q):
    """ReLU^q activation."""
    return np.maximum(0, z) ** q


def relu_power_q_grad(z, q):
    """Gradient of ReLU^q."""
    grad = np.zeros_like(z)
    mask = z > 0
    grad[mask] = q * (z[mask] ** (q - 1))
    return grad


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def teacher_probs(y, is_unlearnable, teacher, k, margin):
    """
    Return teacher soft label.

    Bad teacher:
        confident on both learnable and unlearnable samples.

    Good teacher:
        confident on learnable samples but uncertain on unlearnable samples.
    """
    if teacher == "good" and is_unlearnable:
        return np.ones(k) / k

    logits = np.zeros(k)
    logits[y] = margin
    return softmax(logits)


class ReLUNet:
    def __init__(self, k, m, d, q, sigma0, v_idx):
        self.k = k
        self.m = m
        self.d = d
        self.q = q
        self.v_idx = v_idx

        self.w = np.random.randn(k, m, d) * sigma0
        self.project_unlearnable_feature()

    def project_unlearnable_feature(self):
        """Enforce student weights to be orthogonal to v."""
        self.w[:, :, self.v_idx] = 0.0

    def forward(self, X):
        logits = np.zeros(self.k)
        XT = X.T

        for c in range(self.k):
            dots = self.w[c] @ XT
            logits[c] = np.sum(relu_power_q(dots, self.q))

        return logits

    def predict(self, X_batch):
        return np.array([np.argmax(self.forward(X)) for X in X_batch])

    def grad_distillation(self, X, target_probs):
        """
        Gradient of cross-entropy with teacher soft labels.
        Equivalent to the KL distillation gradient up to constants.
        """
        logits = self.forward(X)
        student_probs = softmax(logits)
        d_logits = student_probs - target_probs

        grad_w = np.zeros_like(self.w)
        XT = X.T

        for c in range(self.k):
            dots = self.w[c] @ XT
            d_relu = relu_power_q_grad(dots, self.q)
            grad_w[c] = d_logits[c] * (d_relu @ X)

        grad_w[:, :, self.v_idx] = 0.0
        return grad_w

    def grad_input(self, X, y):
        logits = self.forward(X)
        probs = softmax(logits)

        d_logits = probs.copy()
        d_logits[y] -= 1.0

        grad_X = np.zeros_like(X)
        XT = X.T

        for c in range(self.k):
            dots = self.w[c] @ XT
            d_relu = relu_power_q_grad(dots, self.q)
            grad_X += d_logits[c] * (self.w[c].T @ d_relu).T

        return grad_X

    def train_attack(self, X, y, eps, u_idx):
        """
        Simple training-time adversary used in the synthetic AD experiment.

        The perturbation shifts the learnable coordinate of all patches.
        Since the student is constrained to be orthogonal to v, the v-feature
        in unlearnable samples remains inaccessible to the student.
        """
        if eps == 0:
            return X.copy()

        X_adv = X.copy()
        attack_sign = -1.0 if y == 1 else 1.0
        X_adv[:, u_idx] += attack_sign * eps
        return X_adv

    def pgd_attack(self, X, y, eps, steps):
        """PGD-style test-time attack."""
        if eps == 0:
            return X.copy()

        step_size = eps / (steps / 2.5)

        X_adv = X + np.random.uniform(-eps, eps, X.shape)
        X_adv = X + np.clip(X_adv - X, -eps, eps)

        for _ in range(steps):
            grad_X = self.grad_input(X_adv, y)
            X_adv = X_adv + step_size * np.sign(grad_X)
            X_adv = X + np.clip(X_adv - X, -eps, eps)

        return X_adv


# ---------------- 4) Metrics ----------------
def clean_acc(model, X_list, y):
    preds = model.predict(X_list)
    return 100.0 * np.mean(preds == y)


def train_robust_acc(model, X_list, y, eps, u_idx):
    X_adv = [model.train_attack(X, yi, eps, u_idx) for X, yi in zip(X_list, y)]
    preds = model.predict(X_adv)
    return 100.0 * np.mean(preds == y)


def test_robust_acc(model, X_list, y, eps, pgd_steps):
    X_adv = [model.pgd_attack(X, yi, eps, pgd_steps) for X, yi in zip(X_list, y)]
    preds = model.predict(X_adv)
    return 100.0 * np.mean(preds == y)


def distillation_loss(model, X, target_probs):
    probs = softmax(model.forward(X))
    return -np.sum(target_probs * np.log(probs + 1e-12))


def weight_metrics(W, u_idx, v_idx, topk=5):
    Wf = W.reshape(-1, W.shape[-1])

    u_mag = np.abs(Wf[:, u_idx])
    noise = np.abs(Wf[:, u_idx + 1 : v_idx])

    noise_rms = np.linalg.norm(noise, axis=1) / np.sqrt(noise.shape[1])
    noise_max = np.max(noise, axis=1)
    noise_min = np.min(noise, axis=1)

    k_eff = min(topk, noise.shape[1])
    noise_topk = np.partition(noise, -k_eff, axis=1)[:, -k_eff:]
    noise_topk_mean = np.mean(noise_topk, axis=1)

    return {
        "u_mag": float(np.mean(u_mag)),
        "noise_rms": float(np.mean(noise_rms)),
        "noise_max": float(np.mean(noise_max)),
        "noise_topk": float(np.mean(noise_topk_mean)),
        "noise_min": float(np.mean(noise_min)),
    }


# ---------------- 5) Diagnostic plot ----------------
def save_diagnostic_plot(history, params, run_dir):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    title = (
        f"{params['teacher'].capitalize()} Teacher AD: "
        f"alpha={params['alpha']}, sigma_n={params['sigma_n']}, "
        f"p_un={params['p_un']}, P={params['p']}\n"
        f"steps={params['steps']}, lr={params['lr']}, eps={params['eps']}, "
        f"teacher_margin={params['teacher_margin']}"
    )
    fig.suptitle(title, fontsize=14)

    ax = axes[0]
    ax.plot(history["Step"], history["Robust_Train_Acc"], "r-", label="Robust Train")
    ax.plot(history["Step"], history["Clean_Train_Acc"], "b-", label="Clean Train")
    ax.plot(history["Step"], history["Robust_Test_Acc"], "k-", label="Robust Test")
    ax.plot(history["Step"], history["Clean_Test_Acc"], "g-", label="Clean Test")
    ax.set_title("Accuracy")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, linestyle=":")
    ax.set_ylim(-5, 105)

    ax = axes[1]
    ax.plot(history["Step"], history["Loss_All"], "k", label="Train CE (all)")
    ax.plot(history["Step"], history["Loss_Learnable"], "g", label="Train CE (learnable)")
    ax.plot(history["Step"], history["Loss_Unlearnable"], "m", label="Train CE (unlearnable)")
    ax.set_title("Distillation Loss")
    ax.set_xlabel("Steps")
    ax.set_ylabel("CE")
    ax.legend()
    ax.grid(True, linestyle=":")

    ax = axes[2]
    ax.plot(history["Step"], history["U_Mag"], "b", label="|w_u|")
    ax.plot(history["Step"], history["Noise_RMS"], "r", label="RMS noise")
    ax.plot(history["Step"], history["Noise_Max"], "orange", label="max noise coord")
    ax.plot(history["Step"], history["Noise_TopK"], "purple", label="top-5 noise mean")
    ax.plot(history["Step"], history["Noise_Min"], "cyan", label="min noise coord")
    ax.set_title("Weight statistics")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Magnitude")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig_path = run_dir / "diagnostic.pdf"
    plt.savefig(fig_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()


# ---------------- 6) Single run ----------------
def run_single(args, teacher, p_un):
    np.random.seed(args.seed)

    u_idx = 0
    v_idx = args.d - 1
    sigma0 = np.sqrt(1.0 / args.d) * args.sigma0_scale

    run_dir = make_run_dir(args.out_dir, teacher, p_un, args.seed)

    config = vars(args).copy()
    config["teacher"] = teacher
    config["p_un"] = p_un
    config["sigma0"] = sigma0
    config["u_idx"] = u_idx
    config["v_idx"] = v_idx

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    u, v = generate_features(args.d, u_idx, v_idx)

    X_train, y_train, train_unlearn = generate_data(
        N=args.n_train,
        P=args.p,
        d=args.d,
        u=u,
        v=v,
        alpha=args.alpha,
        sigma_n=args.sigma_n,
        p_un=p_un,
        u_idx=u_idx,
        v_idx=v_idx,
    )

    X_test, y_test, _ = generate_data(
        N=args.n_test,
        P=args.p,
        d=args.d,
        u=u,
        v=v,
        alpha=args.alpha,
        sigma_n=args.sigma_n,
        p_un=p_un,
        u_idx=u_idx,
        v_idx=v_idx,
    )

    model = ReLUNet(
        k=args.k,
        m=args.m,
        d=args.d,
        q=args.q,
        sigma0=sigma0,
        v_idx=v_idx,
    )

    history = {
        "Step": [],
        "Robust_Train_Acc": [],
        "Robust_Test_Acc": [],
        "Clean_Train_Acc": [],
        "Clean_Test_Acc": [],
        "Loss_All": [],
        "Loss_Learnable": [],
        "Loss_Unlearnable": [],
        "U_Mag": [],
        "Noise_RMS": [],
        "Noise_Max": [],
        "Noise_TopK": [],
        "Noise_Min": [],
        "P_Unlearnable": [],
        "Teacher": [],
        "Seed": [],
    }

    print(f"[Run] teacher={teacher}, p_un={p_un:.2f}, seed={args.seed}")
    print(f"[Save] {run_dir}")

    for t in tqdm(range(args.steps)):
        idx = np.random.randint(args.n_train)

        X_i = X_train[idx]
        y_i = y_train[idx]
        un_i = train_unlearn[idx]

        X_adv_i = model.train_attack(X_i, y_i, args.eps, u_idx)

        target_probs = teacher_probs(
            y=y_i,
            is_unlearnable=un_i,
            teacher=teacher,
            k=args.k,
            margin=args.teacher_margin,
        )

        grad = model.grad_distillation(X_adv_i, target_probs)
        model.w -= args.lr * grad
        model.project_unlearnable_feature()

        if t % args.eval_interval == 0 or t == args.steps - 1:
            eval_idx = np.random.choice(
                args.n_train,
                min(200, args.n_train),
                replace=False,
            )

            X_eval = [X_train[i] for i in eval_idx]
            y_eval = y_train[eval_idx]
            unlearn_eval = train_unlearn[eval_idx]

            robust_train = train_robust_acc(model, X_eval, y_eval, args.eps, u_idx)
            robust_test = test_robust_acc(
                model, X_test, y_test, args.eps, args.pgd_steps
            )
            clean_train = clean_acc(model, X_eval, y_eval)
            clean_test = clean_acc(model, X_test, y_test)

            idx_learn = np.where(~unlearn_eval)[0]
            idx_unlearn = np.where(unlearn_eval)[0]

            X_learn = [X_eval[j] for j in idx_learn]
            y_learn = y_eval[idx_learn]

            X_unlearn = [X_eval[j] for j in idx_unlearn]
            y_unlearn = y_eval[idx_unlearn]

            target_eval = [
                teacher_probs(y, un, teacher, args.k, args.teacher_margin)
                for y, un in zip(y_eval, unlearn_eval)
            ]

            loss_all = np.mean(
                [
                    distillation_loss(model, X, target)
                    for X, target in zip(X_eval, target_eval)
                ]
            )

            loss_learn = (
                np.mean(
                    [
                        distillation_loss(
                            model,
                            X,
                            teacher_probs(y, False, teacher, args.k, args.teacher_margin),
                        )
                        for X, y in zip(X_learn, y_learn)
                    ]
                )
                if len(X_learn) > 0
                else np.nan
            )

            loss_unlearn = (
                np.mean(
                    [
                        distillation_loss(
                            model,
                            X,
                            teacher_probs(y, True, teacher, args.k, args.teacher_margin),
                        )
                        for X, y in zip(X_unlearn, y_unlearn)
                    ]
                )
                if len(X_unlearn) > 0
                else np.nan
            )

            wm = weight_metrics(model.w, u_idx, v_idx, topk=5)

            history["Step"].append(t)
            history["Robust_Train_Acc"].append(robust_train)
            history["Robust_Test_Acc"].append(robust_test)
            history["Clean_Train_Acc"].append(clean_train)
            history["Clean_Test_Acc"].append(clean_test)
            history["Loss_All"].append(loss_all)
            history["Loss_Learnable"].append(loss_learn)
            history["Loss_Unlearnable"].append(loss_unlearn)
            history["U_Mag"].append(wm["u_mag"])
            history["Noise_RMS"].append(wm["noise_rms"])
            history["Noise_Max"].append(wm["noise_max"])
            history["Noise_TopK"].append(wm["noise_topk"])
            history["Noise_Min"].append(wm["noise_min"])
            history["P_Unlearnable"].append(p_un)
            history["Teacher"].append(teacher)
            history["Seed"].append(args.seed)

    df = pd.DataFrame(history)

    csv_path = run_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)

    if args.save_diagnostic_plot:
        save_diagnostic_plot(history, config, run_dir)

    print(f"[Done] metrics saved to {csv_path}")
    print(
        f"[Final] teacher={teacher}, "
        f"robust_train={df['Robust_Train_Acc'].iloc[-1]:.1f}, "
        f"robust_test={df['Robust_Test_Acc'].iloc[-1]:.1f}, "
        f"clean_train={df['Clean_Train_Acc'].iloc[-1]:.1f}, "
        f"clean_test={df['Clean_Test_Acc'].iloc[-1]:.1f}"
    )


# ---------------- 7) Main ----------------
def main():
    args = parse_args()

    if args.teacher is None:
        teachers = ["bad", "good"]
    else:
        teachers = [args.teacher]

    if args.p_un_list is not None:
        p_un_values = args.p_un_list
    elif args.p_un is not None:
        p_un_values = [args.p_un]
    else:
        p_un_values = [0.0, 0.05, 0.10, 0.15, 0.20]

    for teacher in teachers:
        for p_un in p_un_values:
            run_single(args, teacher, p_un)


if __name__ == "__main__":
    main()
