"""Check dx/dL_c[k] = -V A^{-1} w,  w_r = lambda_c (delta_{k,l0} - delta_{k,lr}),
against autograd, for m = 2, 3, 4 tied labels on a random simplex."""
import torch
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
K = 6
for m in (2, 3, 4):
    worst = 0.0
    for trial in range(200):
        v = torch.randn(m, 3)
        S = torch.randperm(K)[:m]
        # make a tie inside the simplex: start from random L and shift tied labels
        L = torch.randn(m, K)
        lam_true = torch.distributions.Dirichlet(torch.ones(m)).sample()
        for r in range(1, m):
            # adjust L[:, S[r]] so that sum lam_a (L_a[l0]-L_a[lr]) = 0 at lam_true
            g = (lam_true * (L[:, S[0]] - L[:, S[r]])).sum()
            L[:, S[r]] += g
        L.requires_grad_(True)
        d = torch.stack([L[:, S[0]] - L[:, S[r]] for r in range(1, m)])  # (m-1, m)
        A = (d[:, 1:] - d[:, :1])
        b = -d[:, 0]
        lp = torch.linalg.solve(A, b)
        lam = torch.cat([1 - lp.sum(0, keepdim=True), lp])
        x = (lam[:, None] * v).sum(0)
        J_auto = torch.stack([torch.autograd.grad(x[i], L, retain_graph=True)[0]
                              for i in range(3)])  # (3, m, K)
        with torch.no_grad():
            V = (v[1:] - v[0]).T  # (3, m-1)
            Ainv = torch.linalg.inv(A)
            J = torch.zeros(3, m, K)
            for c in range(m):
                for k in range(K):
                    w = torch.zeros(m - 1)
                    for r in range(1, m):
                        w[r - 1] = lam[c] * ((k == S[0]).double() - (k == S[r]).double())
                    J[:, c, k] = -V @ (Ainv @ w)
        err = (J - J_auto).abs().max() / J_auto.abs().max().clamp_min(1e-300)
        worst = max(worst, float(err))
    print(f"m={m}: worst relative difference to autograd {worst:.2e}")
