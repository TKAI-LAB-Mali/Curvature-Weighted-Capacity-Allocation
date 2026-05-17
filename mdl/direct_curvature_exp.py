import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DeepMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=96, depth=8, num_classes=10):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (depth - 1) + [num_classes]
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.linears):
            x = layer(x)
            if i < len(self.linears) - 1:
                x = F.relu(x)
        return x


def get_layer_param_groups(model):
    return {
        f"layer_{i + 1}": list(layer.parameters())
        for i, layer in enumerate(model.linears)
    }


def layer_param_counts(model):
    names = []
    counts = []
    for i, layer in enumerate(model.linears):
        names.append(f"layer_{i + 1}")
        counts.append(sum(p.numel() for p in layer.parameters()))
    return names, np.asarray(counts, dtype=np.float64)

def make_digits_loaders(batch_size=128):
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=0,
        stratify=y,
    )

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_model(model, train_loader, device, epochs=40, lr=1e-3):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d} | train loss = {total_loss / total:.4f}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    return correct / max(total, 1)

def flatten_tensors(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten_like(vec, tensors):
    outs = []
    idx = 0
    for t in tensors:
        n = t.numel()
        outs.append(vec[idx:idx + n].view_as(t))
        idx += n
    return outs


def replace_none_grads(grads, params):
    return [
        torch.zeros_like(p) if g is None else g
        for g, p in zip(grads, params)
    ]


def conjugate_gradient(Avp, b, max_iter=80, tol=1e-6):
    """
    Solve Ax=b using conjugate gradient.
    Assumes A is approximately SPD after damping.
    If negative curvature is detected, stops early and reports it.
    """

    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = torch.dot(r, r)

    converged = False
    negative_curvature = False

    for it in range(max_iter):
        Ap = Avp(p)
        denom = torch.dot(p, Ap)

        if denom <= 1e-12:
            negative_curvature = True
            break

        alpha = rs_old / (denom + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = torch.dot(r, r)

        if torch.sqrt(rs_new) < tol:
            converged = True
            rs_old = rs_new
            break

        beta = rs_new / (rs_old + 1e-12)
        p = r + beta * p
        rs_old = rs_new

    info = {
        "converged": converged,
        "negative_curvature": negative_curvature,
        "iterations": it + 1,
        "residual_norm": float(torch.sqrt(rs_old).detach().cpu()),
    }

    return x, info

def curvature_gain_hvp_cg(
    model,
    batch,
    layer_params,
    device,
    damping=1e-2,
    max_cg_iter=80,
    tol=1e-6,
):
    """
    Approximate

        q_curv = g_k^T (H_kk + damping I)^(-1) g_k

    using Hessian-vector products and conjugate gradient.
    """

    model.zero_grad(set_to_none=True)
    model.train()

    x, y = batch
    x, y = x.to(device), y.to(device)

    logits = model(x)
    loss = F.cross_entropy(logits, y)

    grads = torch.autograd.grad(
        loss,
        layer_params,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    grads = replace_none_grads(grads, layer_params)

    g_flat = flatten_tensors([g.detach() for g in grads])

    def Avp(v_flat):
        v_tensors = unflatten_like(v_flat, layer_params)

        gv = sum((g * v).sum() for g, v in zip(grads, v_tensors))

        hv = torch.autograd.grad(
            gv,
            layer_params,
            retain_graph=True,
            allow_unused=True,
        )
        hv = replace_none_grads(hv, layer_params)

        hv_flat = flatten_tensors([h.detach() for h in hv])
        return hv_flat + damping * v_flat

    x_sol, info = conjugate_gradient(
        Avp=Avp,
        b=g_flat,
        max_iter=max_cg_iter,
        tol=tol,
    )

    q_curv = torch.dot(g_flat, x_sol).item()
    q_grad = torch.dot(g_flat, g_flat).item()

    return max(q_curv, 0.0), max(q_grad, 0.0), info

def diagonal_fisher_proxy(
    model,
    batch,
    layer_params,
    device,
    damping=1e-3,
    max_samples=96,
):

    model.zero_grad(set_to_none=True)
    model.eval()

    x, y = batch
    x, y = x.to(device), y.to(device)
    n = min(x.size(0), max_samples)

    logits = model(x[:n])
    loss = F.cross_entropy(logits, y[:n])

    grads = torch.autograd.grad(
        loss,
        layer_params,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    grads = replace_none_grads(grads, layer_params)
    g_flat = flatten_tensors([g.detach() for g in grads])

    fisher_diag = torch.zeros_like(g_flat)

    for i in range(n):
        model.zero_grad(set_to_none=True)

        logits_i = model(x[i:i + 1])
        loss_i = F.cross_entropy(logits_i, y[i:i + 1])

        grads_i = torch.autograd.grad(
            loss_i,
            layer_params,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        grads_i = replace_none_grads(grads_i, layer_params)

        gi_flat = flatten_tensors([g.detach() for g in grads_i])
        fisher_diag += gi_flat ** 2

    fisher_diag /= float(n)

    q = torch.sum((g_flat ** 2) / (fisher_diag + damping)).item()
    return max(q, 0.0)


def datainf_style_proxy(
    model,
    batch,
    layer_params,
    device,
    damping=1e-2,
    max_samples=96,
):
    """
    DataInf/LayerIF-style empirical-Fisher influence proxy:

        q_IF = g^T (lambda I + (1/n) G^T G)^(-1) g

    where:
        g = batch gradient for the layer,
        G = per-sample gradient matrix for the layer.

    Uses Woodbury to avoid p x p inversion:

        A = lambda I + (1/n)G^T G

        A^{-1}g =
            (1/lambda)g
            - (1/lambda)G^T(n lambda I + GG^T)^(-1)Gg
    """

    model.zero_grad(set_to_none=True)
    model.eval()

    x, y = batch
    x, y = x.to(device), y.to(device)
    n = min(x.size(0), max_samples)

    # Batch gradient
    logits = model(x[:n])
    loss = F.cross_entropy(logits, y[:n])

    grads = torch.autograd.grad(
        loss,
        layer_params,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    grads = replace_none_grads(grads, layer_params)
    g_flat = flatten_tensors([g.detach() for g in grads])

    # Per-sample gradient matrix G
    per_sample_grads = []

    for i in range(n):
        model.zero_grad(set_to_none=True)

        logits_i = model(x[i:i + 1])
        loss_i = F.cross_entropy(logits_i, y[i:i + 1])

        grads_i = torch.autograd.grad(
            loss_i,
            layer_params,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        grads_i = replace_none_grads(grads_i, layer_params)

        gi_flat = flatten_tensors([g.detach() for g in grads_i])
        per_sample_grads.append(gi_flat)

    G = torch.stack(per_sample_grads, dim=0)  # [n, p]

    Gg = G @ g_flat
    Kmat = G @ G.T
    Kmat = Kmat + (n * damping) * torch.eye(
        n,
        device=device,
        dtype=Kmat.dtype,
    )

    try:
        alpha = torch.linalg.solve(Kmat, Gg)
        Ainv_g = (g_flat / damping) - (G.T @ alpha) / damping
        q = torch.dot(g_flat, Ainv_g).item()
    except RuntimeError:
        q = 0.0

    if not np.isfinite(q):
        q = 0.0

    return max(q, 0.0)


def layer_ablation_proxy(
    model,
    batch,
    layer_index,
    device,
    epsilon=0.10,
):

    model.eval()

    x, y = batch
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        base_loss = F.cross_entropy(model(x), y).item()

    def hook_fn(module, inp, out):
        return (1.0 - epsilon) * out

    handle = model.linears[layer_index].register_forward_hook(hook_fn)

    with torch.no_grad():
        pert_loss = F.cross_entropy(model(x), y).item()

    handle.remove()

    return max(pert_loss - base_loss, 0.0)

def normalize_sum_one(q, eps=1e-12):
    q = np.asarray(q, dtype=np.float64)
    q = np.maximum(q, 0.0)
    return q / (q.sum() + eps)


def normalize_mean_one(q, eps=1e-12):
    q = np.asarray(q, dtype=np.float64)
    return q / (q.mean() + eps)


def rankdata(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman_corr(x, y):
    rx = rankdata(x)
    ry = rankdata(y)

    rx -= rx.mean()
    ry -= ry.mean()

    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    return 0.0 if denom < 1e-12 else float(rx @ ry / denom)


def kendall_tau(x, y):
    n = len(x)
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            sx = np.sign(x[i] - x[j])
            sy = np.sign(y[i] - y[j])

            if sx == 0 or sy == 0:
                continue

            if sx == sy:
                concordant += 1
            else:
                discordant += 1

    denom = concordant + discordant
    return 0.0 if denom == 0 else float((concordant - discordant) / denom)


def top_m_overlap(x, y, m=3):
    top_x = set(np.argsort(-np.asarray(x))[:m])
    top_y = set(np.argsort(-np.asarray(y))[:m])
    return len(top_x.intersection(top_y)) / float(m)


def l1_relative(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sum(np.abs(a - b)) / (np.sum(np.abs(a)) + eps))


def calibrate_gamma_for_active_budget(q, costs, B, alpha=0.5, beta=1.0, target_multiplier=1.25):
    q = normalize_sum_one(q)
    costs = np.asarray(costs, dtype=np.float64)
    K = len(q)

    qbeta = np.maximum(q, 1e-12) ** beta

    # Want average unconstrained e roughly target_multiplier * B/K:
    # gamma*qbeta/(alpha*c) - 1 ~= target_multiplier*B/K
    target_e = target_multiplier * B / K
    gamma_values = (target_e + 1.0) * alpha * costs / qbeta

    # Use median for robustness.
    gamma = float(np.median(gamma_values))
    return gamma


def algorithm1_mdl_allocation_continuous(
    q,
    costs=None,
    B=20.0,
    alpha=0.5,
    gamma=None,
    beta=1.0,
    tol=1e-10,
    max_iter=200,
):

    q = normalize_sum_one(q)
    K = len(q)

    if costs is None:
        costs = np.ones(K, dtype=np.float64)
    else:
        costs = np.asarray(costs, dtype=np.float64)

    costs = np.maximum(costs, 1e-12)

    if gamma is None:
        gamma = calibrate_gamma_for_active_budget(
            q=q,
            costs=costs,
            B=B,
            alpha=alpha,
            beta=beta,
        )

    def e_of_lambda(lam):
        e = gamma * (q ** beta) / ((alpha + lam) * costs) - 1.0
        return np.maximum(e, 0.0)

    def budget_used(lam):
        return float(np.dot(costs, e_of_lambda(lam)))

    # If unconstrained solution is feasible.
    if budget_used(0.0) <= B:
        return e_of_lambda(0.0), 0.0, gamma

    lo = 0.0
    hi = 1.0

    while budget_used(hi) > B:
        hi *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)

        if budget_used(mid) > B:
            lo = mid
        else:
            hi = mid

        if hi - lo < tol:
            break

    lam_star = hi
    return e_of_lambda(lam_star), lam_star, gamma


def round_largest_remainder_equal_cost(e_cont, total_experts):
    """
    Converts continuous expert allocation to integer expert counts for equal costs, theo says this so use it, why not.
    """

    e_cont = np.asarray(e_cont, dtype=np.float64)
    e_cont = np.maximum(e_cont, 0.0)

    if e_cont.sum() <= 1e-12:
        return np.zeros_like(e_cont, dtype=int)

    e_scaled = total_experts * e_cont / e_cont.sum()
    floors = np.floor(e_scaled).astype(int)

    leftover = int(total_experts - floors.sum())

    if leftover > 0:
        frac = e_scaled - floors
        idx = np.argsort(-frac)[:leftover]
        floors[idx] += 1

    return floors


def compare_algorithm1_allocations(
    q_curv,
    proxy_scores,
    costs=None,
    total_budget=20.0,
    total_experts=20,
    alpha=0.5,
    gamma=None,
    beta=1.0,
):

    K = len(q_curv)

    if costs is None:
        costs = np.ones(K, dtype=np.float64)

    # Calibrate gamma on q_curv once if not supplied.
    if gamma is None:
        gamma = calibrate_gamma_for_active_budget(
            q=normalize_sum_one(q_curv),
            costs=costs,
            B=total_budget,
            alpha=alpha,
            beta=beta,
        )

    e_curv_cont, lam_curv, _ = algorithm1_mdl_allocation_continuous(
        q=q_curv,
        costs=costs,
        B=total_budget,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
    )

    e_curv_int = round_largest_remainder_equal_cost(
        e_curv_cont,
        total_experts=total_experts,
    )

    rows = []

    for name, q_proxy in proxy_scores.items():
        e_proxy_cont, lam_proxy, _ = algorithm1_mdl_allocation_continuous(
            q=q_proxy,
            costs=costs,
            B=total_budget,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
        )

        e_proxy_int = round_largest_remainder_equal_cost(
            e_proxy_cont,
            total_experts=total_experts,
        )

        rows.append({
            "proxy": name,
            "lambda_curv": lam_curv,
            "lambda_proxy": lam_proxy,
            "gamma": gamma,
            "alloc_l1_cont": l1_relative(e_curv_cont, e_proxy_cont),
            "alloc_l1_int": l1_relative(e_curv_int, e_proxy_int),
            "alloc_top3_cont": top_m_overlap(e_curv_cont, e_proxy_cont, m=3),
            "alloc_top3_int": top_m_overlap(e_curv_int, e_proxy_int, m=3),
            "e_curv_cont": e_curv_cont,
            "e_proxy_cont": e_proxy_cont,
            "e_curv_int": e_curv_int,
            "e_proxy_int": e_proxy_int,
        })

    return rows


def allocation_uniform(K, total_experts):
    base = np.ones(K) * (total_experts / K)
    return round_largest_remainder_equal_cost(base, total_experts)


def allocation_random(K, total_experts, rng):
    w = rng.random(K)
    w = w / w.sum()
    return round_largest_remainder_equal_cost(w, total_experts)


def allocation_parameter_proportional(param_counts, total_experts):
    w = np.asarray(param_counts, dtype=np.float64)
    w = w / w.sum()
    return round_largest_remainder_equal_cost(w, total_experts)


def exact_budget_pruning_quadratic(
    q,
    n_params,
    target_sparsity=0.5,
    eta=2.0,
    kappa=1.0,
    rho_max=0.95,
    tol=1e-10,
    max_iter=200,
):
    """
    Exact-budget pruning diagnostic:

        min sum_k eta q_k^kappa rho_k^2
        s.t. sum_k n_k rho_k = S
             0 <= rho_k <= rho_max

    KKT:
        rho_k(lambda) = clip(lambda n_k / (2 eta q_k^kappa), 0, rho_max)
    """

    q = normalize_sum_one(q)
    q = np.maximum(q, 1e-12)

    n = np.asarray(n_params, dtype=np.float64)
    S = target_sparsity * n.sum()

    def rho_of_lambda(lam):
        rho = lam * n / (2.0 * eta * (q ** kappa))
        return np.clip(rho, 0.0, rho_max)

    def sparsity_used(lam):
        return float(np.dot(n, rho_of_lambda(lam)))

    lo = 0.0
    hi = 1.0

    while sparsity_used(hi) < S:
        hi *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)

        if sparsity_used(mid) < S:
            lo = mid
        else:
            hi = mid

        if hi - lo < tol:
            break

    return rho_of_lambda(hi), hi


def exact_budget_pruning_logbarrier(
    q,
    n_params,
    target_sparsity=0.5,
    eta=2.0,
    kappa=1.0,
    rho_max=0.95,
    tol=1e-10,
    max_iter=200,
):
    """
    Exact-budget pruning diagnostic:

        min sum_k eta q_k^kappa [-log(1-rho_k)]
        s.t. sum_k n_k rho_k = S
             0 <= rho_k <= rho_max

    KKT:
        rho_k(lambda) =
            1 - eta q_k^kappa / (lambda n_k)
    """

    q = normalize_sum_one(q)
    q = np.maximum(q, 1e-12)

    n = np.asarray(n_params, dtype=np.float64)
    S = target_sparsity * n.sum()

    def rho_of_lambda(lam):
        rho = 1.0 - (eta * (q ** kappa)) / (lam * n + 1e-12)
        return np.clip(rho, 0.0, rho_max)

    def sparsity_used(lam):
        return float(np.dot(n, rho_of_lambda(lam)))

    lo = 1e-12
    hi = 1.0

    while sparsity_used(hi) < S:
        hi *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)

        if sparsity_used(mid) < S:
            lo = mid
        else:
            hi = mid

        if hi - lo < tol:
            break

    return rho_of_lambda(hi), hi


def pruning_uniform(n_params, target_sparsity):
    return np.ones_like(np.asarray(n_params, dtype=np.float64)) * target_sparsity


def apply_layerwise_magnitude_pruning(model, rho_by_layer):
    """
    Applies unstructured magnitude pruning to Linear weights only.
    """

    for layer, rho in zip(model.linears, rho_by_layer):
        rho = float(np.clip(rho, 0.0, 0.99))

        W = layer.weight.data
        numel = W.numel()
        prune_count = int(round(rho * numel))

        if prune_count <= 0:
            continue
        if prune_count >= numel:
            prune_count = numel - 1

        flat_abs = W.abs().view(-1)
        threshold = torch.kthvalue(flat_abs, prune_count).values
        mask = (W.abs() > threshold).float()
        layer.weight.data.mul_(mask)

    return model


def achieved_weight_sparsity(model):
    zeros = 0
    total = 0

    for layer in model.linears:
        W = layer.weight.data
        zeros += (W == 0).sum().item()
        total += W.numel()

    return zeros / max(total, 1)


def compute_layer_scores_for_seed(
    seed=0,
    taus=(1e-3, 1e-2, 1e-1, 1.0, 2.0),
    depth=8,
    hidden_dim=96,
    epochs=40,
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = make_digits_loaders(batch_size=128)

    model = DeepMLP(
        input_dim=64,
        hidden_dim=hidden_dim,
        depth=depth,
        num_classes=10,
    ).to(device)

    print(f"\n=== Seed {seed} ===")
    train_model(model, train_loader, device, epochs=epochs, lr=1e-3)

    base_acc = evaluate(model, test_loader, device)
    print(f"Base test accuracy: {base_acc * 100:.2f}%")

    calib_batch = next(iter(train_loader))
    layer_groups = get_layer_param_groups(model)
    layer_names, param_counts = layer_param_counts(model)

    q_grad = []
    q_diagF = []
    q_datainf = []
    q_ablation = []

    print("Computing proxy scores...")

    for idx, (name, params) in enumerate(layer_groups.items()):

        _, grad_score, _ = curvature_gain_hvp_cg(
            model=model,
            batch=calib_batch,
            layer_params=params,
            device=device,
            damping=1.0,
            max_cg_iter=20,
            tol=1e-6,
        )

        diagF_score = diagonal_fisher_proxy(
            model=model,
            batch=calib_batch,
            layer_params=params,
            device=device,
            damping=1e-3,
            max_samples=96,
        )

        datainf_score = datainf_style_proxy(
            model=model,
            batch=calib_batch,
            layer_params=params,
            device=device,
            damping=1e-2,
            max_samples=96,
        )

        ablation_score = layer_ablation_proxy(
            model=model,
            batch=calib_batch,
            layer_index=idx,
            device=device,
            epsilon=0.10,
        )

        q_grad.append(grad_score)
        q_diagF.append(diagF_score)
        q_datainf.append(datainf_score)
        q_ablation.append(ablation_score)

    q_grad = normalize_sum_one(q_grad)
    q_diagF = normalize_sum_one(q_diagF)
    q_datainf = normalize_sum_one(q_datainf)
    q_ablation = normalize_sum_one(q_ablation)

    q_curv_by_tau = {}
    cg_info_by_tau = {}

    print("Computing direct curvature scores by damping tau...")

    for tau in taus:
        q_curv = []
        neg_count = 0
        conv_count = 0
        avg_iter = []

        for name, params in layer_groups.items():
            curv_score, _, info = curvature_gain_hvp_cg(
                model=model,
                batch=calib_batch,
                layer_params=params,
                device=device,
                damping=tau,
                max_cg_iter=80,
                tol=1e-6,
            )

            q_curv.append(curv_score)
            neg_count += int(info["negative_curvature"])
            conv_count += int(info["converged"])
            avg_iter.append(info["iterations"])

        q_curv_by_tau[tau] = normalize_sum_one(q_curv)
        cg_info_by_tau[tau] = {
            "negative_curvature_layers": neg_count,
            "converged_layers": conv_count,
            "avg_cg_iter": float(np.mean(avg_iter)),
        }

    return {
        "seed": seed,
        "model": model,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "device": device,
        "base_acc": base_acc,
        "layer_names": layer_names,
        "param_counts": param_counts,
        "q_grad": q_grad,
        "q_diagF": q_diagF,
        "q_datainf": q_datainf,
        "q_ablation": q_ablation,
        "q_curv_by_tau": q_curv_by_tau,
        "cg_info_by_tau": cg_info_by_tau,
    }



def evaluate_score_and_algorithm1_agreement(
    seed_result,
    total_budget=20.0,
    total_experts=20,
    alpha=0.5,
    beta=1.0,
):
    rows = []

    K = len(seed_result["layer_names"])
    costs = np.ones(K, dtype=np.float64)

    proxies = {
        "datainf_style": seed_result["q_datainf"],
        "grad_norm": seed_result["q_grad"],
        "diag_fisher": seed_result["q_diagF"],
        "layer_ablation": seed_result["q_ablation"],
    }

    # Baseline allocations independent of score
    rng = np.random.default_rng(seed_result["seed"])
    alloc_baselines = {
        "uniform": allocation_uniform(K, total_experts),
        "random": allocation_random(K, total_experts, rng),
        "parameter_prop": allocation_parameter_proportional(
            seed_result["param_counts"],
            total_experts,
        ),
    }

    for tau, q_curv in seed_result["q_curv_by_tau"].items():

        # Compare paper Algorithm 1 allocations.
        alloc_rows = compare_algorithm1_allocations(
            q_curv=q_curv,
            proxy_scores=proxies,
            costs=costs,
            total_budget=total_budget,
            total_experts=total_experts,
            alpha=alpha,
            gamma=None,
            beta=beta,
        )

        alloc_by_proxy = {r["proxy"]: r for r in alloc_rows}

        # Curvature allocation for comparing score-free baselines.
        e_curv_cont, _, gamma_used = algorithm1_mdl_allocation_continuous(
            q=q_curv,
            costs=costs,
            B=total_budget,
            alpha=alpha,
            gamma=None,
            beta=beta,
        )
        e_curv_int = round_largest_remainder_equal_cost(e_curv_cont, total_experts)

        for proxy_name, q_proxy in proxies.items():
            ar = alloc_by_proxy[proxy_name]

            rows.append({
                "seed": seed_result["seed"],
                "tau": tau,
                "proxy": proxy_name,
                "spearman": spearman_corr(q_curv, q_proxy),
                "kendall": kendall_tau(q_curv, q_proxy),
                "top3_score": top_m_overlap(q_curv, q_proxy, m=3),
                "alg1_alloc_l1_cont": ar["alloc_l1_cont"],
                "alg1_alloc_l1_int": ar["alloc_l1_int"],
                "alg1_alloc_top3_int": ar["alloc_top3_int"],
                "gamma_used": ar["gamma"],
                **seed_result["cg_info_by_tau"][tau],
            })

        for base_name, e_base in alloc_baselines.items():
            rows.append({
                "seed": seed_result["seed"],
                "tau": tau,
                "proxy": base_name,
                "spearman": np.nan,
                "kendall": np.nan,
                "top3_score": np.nan,
                "alg1_alloc_l1_cont": np.nan,
                "alg1_alloc_l1_int": l1_relative(e_curv_int, e_base),
                "alg1_alloc_top3_int": top_m_overlap(e_curv_int, e_base, m=3),
                "gamma_used": gamma_used,
                **seed_result["cg_info_by_tau"][tau],
            })

    return rows


def evaluate_algorithm1_hyperparameter_sensitivity(
    seed_result,
    tau=1e-2,
    total_budget=20.0,
    total_experts=20,
):
    """
    Sensitivity to Algorithm 1 hyperparameters alpha, gamma scale, beta.
    """

    q = seed_result["q_curv_by_tau"][tau]
    K = len(q)
    costs = np.ones(K, dtype=np.float64)

    # Reference allocation
    e_ref_cont, _, gamma_ref = algorithm1_mdl_allocation_continuous(
        q=q,
        costs=costs,
        B=total_budget,
        alpha=0.5,
        gamma=None,
        beta=1.0,
    )
    e_ref_int = round_largest_remainder_equal_cost(e_ref_cont, total_experts)

    alpha_grid = [0.25, 0.5, 1.0]
    beta_grid = [0.5, 1.0, 2.0]
    gamma_scale_grid = [0.5, 1.0, 2.0]

    rows = []

    for alpha in alpha_grid:
        for beta in beta_grid:
            gamma_base = calibrate_gamma_for_active_budget(
                q=q,
                costs=costs,
                B=total_budget,
                alpha=alpha,
                beta=beta,
            )

            for gamma_scale in gamma_scale_grid:
                gamma = gamma_scale * gamma_base

                e_cont, _, _ = algorithm1_mdl_allocation_continuous(
                    q=q,
                    costs=costs,
                    B=total_budget,
                    alpha=alpha,
                    gamma=gamma,
                    beta=beta,
                )
                e_int = round_largest_remainder_equal_cost(e_cont, total_experts)

                rows.append({
                    "seed": seed_result["seed"],
                    "tau": tau,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma_scale": gamma_scale,
                    "alg1_l1_vs_ref": l1_relative(e_ref_int, e_int),
                    "alg1_top3_vs_ref": top_m_overlap(e_ref_int, e_int, m=3),
                })

    return rows


def evaluate_exact_budget_pruning(
    seed_result,
    tau=1e-2,
    target_sparsities=(0.3, 0.5, 0.7),
):
    model = seed_result["model"]
    test_loader = seed_result["test_loader"]
    device = seed_result["device"]
    param_counts = seed_result["param_counts"]

    scores = {
        "curv": seed_result["q_curv_by_tau"][tau],
        "datainf_style": seed_result["q_datainf"],
        "grad_norm": seed_result["q_grad"],
    }

    rows = []

    for sparsity in target_sparsities:

        # Uniform pruning baseline
        rho_uniform = pruning_uniform(param_counts, target_sparsity=sparsity)
        model_p = copy.deepcopy(model)
        apply_layerwise_magnitude_pruning(model_p, rho_uniform)
        acc = evaluate(model_p, test_loader, device)
        achieved = achieved_weight_sparsity(model_p)

        rows.append({
            "seed": seed_result["seed"],
            "tau": tau,
            "target_sparsity": sparsity,
            "score": "uniform",
            "penalty": "uniform",
            "accuracy": acc,
            "achieved_sparsity": achieved,
        })

        for score_name, q in scores.items():
            for penalty in ["quadratic", "log"]:
                if penalty == "quadratic":
                    rho, _ = exact_budget_pruning_quadratic(
                        q=q,
                        n_params=param_counts,
                        target_sparsity=sparsity,
                        eta=2.0,
                        kappa=1.0,
                        rho_max=0.95,
                    )
                else:
                    rho, _ = exact_budget_pruning_logbarrier(
                        q=q,
                        n_params=param_counts,
                        target_sparsity=sparsity,
                        eta=2.0,
                        kappa=1.0,
                        rho_max=0.95,
                    )

                model_p = copy.deepcopy(model)
                apply_layerwise_magnitude_pruning(model_p, rho)
                acc = evaluate(model_p, test_loader, device)
                achieved = achieved_weight_sparsity(model_p)

                rows.append({
                    "seed": seed_result["seed"],
                    "tau": tau,
                    "target_sparsity": sparsity,
                    "score": score_name,
                    "penalty": penalty,
                    "accuracy": acc,
                    "achieved_sparsity": achieved,
                })

    return rows


def mean_std(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return vals.mean(), vals.std(ddof=0)


def fmt_ms(m, s):
    if np.isnan(m):
        return "      --       "
    return f"{m:6.3f}±{s:5.3f}"


def print_score_and_algorithm1_agreement(rows):
    taus = sorted(set(r["tau"] for r in rows))
    proxies = sorted(set(r["proxy"] for r in rows))

    print("\n=== Score agreement and exact Algorithm 1 allocation agreement ===")

    for tau in taus:
        print(f"\nTau / damping = {tau:g}")
        print("-" * 150)
        print(
            f"{'Proxy/Baseline':18s} | {'Spearman':>15s} | {'Kendall':>15s} | "
            f"{'Top3 score':>15s} | {'Alg1 L1 cont':>15s} | {'Alg1 L1 int':>15s} | "
            f"{'Alg1 Top3 int':>15s} | {'NegCurv':>10s}"
        )
        print("-" * 150)

        for proxy in proxies:
            sub = [r for r in rows if r["tau"] == tau and r["proxy"] == proxy]

            def metric(key):
                vals = [r[key] for r in sub if not np.isnan(r[key])]
                if len(vals) == 0:
                    return np.nan, np.nan
                return mean_std(vals)

            sp_m, sp_s = metric("spearman")
            kt_m, kt_s = metric("kendall")
            t3_m, t3_s = metric("top3_score")
            l1c_m, l1c_s = metric("alg1_alloc_l1_cont")
            l1i_m, l1i_s = metric("alg1_alloc_l1_int")
            at3_m, at3_s = metric("alg1_alloc_top3_int")
            neg_m, neg_s = mean_std([r["negative_curvature_layers"] for r in sub])

            print(
                f"{proxy:18s} | "
                f"{fmt_ms(sp_m, sp_s):>15s} | "
                f"{fmt_ms(kt_m, kt_s):>15s} | "
                f"{fmt_ms(t3_m, t3_s):>15s} | "
                f"{fmt_ms(l1c_m, l1c_s):>15s} | "
                f"{fmt_ms(l1i_m, l1i_s):>15s} | "
                f"{fmt_ms(at3_m, at3_s):>15s} | "
                f"{neg_m:6.2f}±{neg_s:4.2f}"
            )

        print("-" * 150)


def print_algorithm1_sensitivity(rows):
    print("\n=== Exact Algorithm 1 hyperparameter sensitivity ===")
    print("-" * 100)
    print(
        f"{'alpha':>8s} | {'beta':>8s} | {'gamma scale':>12s} | "
        f"{'Alg1 L1 vs ref':>18s} | {'Alg1 Top3 vs ref':>18s}"
    )
    print("-" * 100)

    keys = sorted(set((r["alpha"], r["beta"], r["gamma_scale"]) for r in rows))

    for alpha, beta, gamma_scale in keys:
        sub = [
            r for r in rows
            if r["alpha"] == alpha and r["beta"] == beta and r["gamma_scale"] == gamma_scale
        ]

        l1_m, l1_s = mean_std([r["alg1_l1_vs_ref"] for r in sub])
        t3_m, t3_s = mean_std([r["alg1_top3_vs_ref"] for r in sub])

        print(
            f"{alpha:8.3g} | {beta:8.3g} | {gamma_scale:12.3g} | "
            f"{l1_m:8.3f}±{l1_s:5.3f} | "
            f"{t3_m:8.3f}±{t3_s:5.3f}"
        )

    print("-" * 100)


def print_pruning_results(rows):
    print("\n=== Exact-budget pruning results ===")
    print("-" * 120)
    print(
        f"{'Target':>8s} | {'Score':>14s} | {'Penalty':>12s} | "
        f"{'Accuracy':>18s} | {'Achieved sparsity':>20s}"
    )
    print("-" * 120)

    keys = sorted(set((r["target_sparsity"], r["score"], r["penalty"]) for r in rows))

    for target, score, penalty in keys:
        sub = [
            r for r in rows
            if r["target_sparsity"] == target
            and r["score"] == score
            and r["penalty"] == penalty
        ]

        acc_m, acc_s = mean_std([r["accuracy"] for r in sub])
        sp_m, sp_s = mean_std([r["achieved_sparsity"] for r in sub])

        print(
            f"{target:8.2f} | {score:14s} | {penalty:12s} | "
            f"{acc_m * 100:8.2f}±{acc_s * 100:5.2f} | "
            f"{sp_m:8.3f}±{sp_s:5.3f}"
        )

    print("-" * 120)


def run_full_diagnostic(
    num_seeds=3,
    taus=(1e-3, 1e-2, 1e-1, 1.0, 2.0),
    depth=8,
    hidden_dim=96,
    epochs=40,
    total_budget=20.0,
    total_experts=20,
):
    all_agreement_rows = []
    all_sensitivity_rows = []
    all_pruning_rows = []

    for seed in range(num_seeds):
        result = compute_layer_scores_for_seed(
            seed=seed,
            taus=taus,
            depth=depth,
            hidden_dim=hidden_dim,
            epochs=epochs,
        )

        agreement_rows = evaluate_score_and_algorithm1_agreement(
            seed_result=result,
            total_budget=total_budget,
            total_experts=total_experts,
            alpha=0.5,
            beta=1.0,
        )

        tau_for_sensitivity = 1e-2 if 1e-2 in result["q_curv_by_tau"] else taus[0]

        sensitivity_rows = evaluate_algorithm1_hyperparameter_sensitivity(
            seed_result=result,
            tau=tau_for_sensitivity,
            total_budget=total_budget,
            total_experts=total_experts,
        )

        pruning_rows = evaluate_exact_budget_pruning(
            seed_result=result,
            tau=tau_for_sensitivity,
            target_sparsities=(0.3, 0.5, 0.7),
        )

        all_agreement_rows.extend(agreement_rows)
        all_sensitivity_rows.extend(sensitivity_rows)
        all_pruning_rows.extend(pruning_rows)

    print_score_and_algorithm1_agreement(all_agreement_rows)
    print_algorithm1_sensitivity(all_sensitivity_rows)
    print_pruning_results(all_pruning_rows)

    return {
        "agreement": all_agreement_rows,
        "sensitivity": all_sensitivity_rows,
        "pruning": all_pruning_rows,
    }


if __name__ == "__main__":
    results = run_full_diagnostic(
        num_seeds=3,
        taus=(1e-3, 1e-2, 1e-1, 1.0, 2.0),
        depth=8,
        hidden_dim=96,
        epochs=40,
        total_budget=20.0,
        total_experts=20,
    )