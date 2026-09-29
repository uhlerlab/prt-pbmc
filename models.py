import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class MultiFeatClassifier(nn.Module):
    def __init__(self, classes=5, input_dim=512):
        super(MultiFeatClassifier, self).__init__()
        self.M = input_dim
        self.L = 128
        self.classes = classes

        self.classifier = nn.Sequential(
            nn.Linear(self.M, self.M),
            nn.ReLU(),
            nn.Linear(self.M, self.L),
            nn.ReLU(),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Linear(self.L, self.classes)
        )

    def forward(self, x, softmax=True):
        Z = H = x.squeeze(0)

        Y_prob = self.classifier(Z).squeeze()
        if softmax:
          Y_prob = torch.nn.functional.softmax(Y_prob, dim=-1)
        Y_hat = torch.argmax(Y_prob, dim=-1)

        return Y_prob, Y_hat, H

    def calculate_objective(self, X, Y):
        Y_logit, Y_hat, _ = self.forward(X, softmax=False)
        loss = torch.nn.functional.cross_entropy(Y_logit, Y)

        return loss, Y_hat


class FeatClassifier(nn.Module):
    def __init__(self, input_dim=512):
        super(FeatClassifier, self).__init__()
        self.M = input_dim
        self.L = 128

        self.classifier = nn.Sequential(
            nn.Linear(self.M, self.M),
            nn.ReLU(),
            nn.Linear(self.M, self.L),
            nn.ReLU(),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Linear(self.L, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        Z = H = x.squeeze(0)

        Y_prob = self.classifier(Z).squeeze()
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, H

    def calculate_objective(self, X, Y):
        Y_prob, Y_hat, _ = self.forward(X)
        loss = torch.nn.functional.binary_cross_entropy(Y_prob, Y)

        return loss, Y_hat
    

class MultiClassifier(nn.Module):
    def __init__(self, feature_norm=None, classes=5):
        super(MultiClassifier, self).__init__()
        self.M = 500
        self.L = 128
        self.classes = classes

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        if feature_norm is None:
          actv_norm = (nn.ReLU(),)
        elif feature_norm == 'softmax':
          actv_norm = (nn.Softmax(dim=1),)
        elif feature_norm == 'layernorm':
          actv_norm = (nn.ReLU(), nn.LayerNorm(500))
        elif feature_norm == 'l1':
          actv_norm = (nn.ReLU(),)
        else:
          assert False

        self.feat_l1_norm = feature_norm == 'l1'

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            *actv_norm
        )
        self.classifier = nn.Linear(self.M, self.classes)

    def forward(self, x, softmax=True):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)
        if self.feat_l1_norm:
          H = F.normalize(H, dim=1)

        Z = H

        Y_prob = self.classifier(Z).squeeze()
        if softmax:
          Y_prob = torch.nn.functional.softmax(Y_prob, dim=-1)
        Y_hat = torch.argmax(Y_prob, dim=-1)

        return Y_prob, Y_hat, H

    def calculate_objective(self, X, Y):
        Y_logit, Y_hat, _ = self.forward(X, softmax=False)
        loss = torch.nn.functional.cross_entropy(Y_logit, Y)

        return loss, Y_hat
    

class Classifier(nn.Module):
    def __init__(self, feature_norm=None):
        super(Classifier, self).__init__()
        self.M = 500
        self.L = 128

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        if feature_norm is None:
          actv_norm = (nn.ReLU(),)
        elif feature_norm == 'softmax':
          actv_norm = (nn.Softmax(dim=1),)
        elif feature_norm == 'layernorm':
          actv_norm = (nn.ReLU(), nn.LayerNorm(500))
        elif feature_norm == 'l1':
          actv_norm = (nn.ReLU(),)
        else:
          assert False

        self.feat_l1_norm = feature_norm == 'l1'

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            *actv_norm
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.M, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)
        if self.feat_l1_norm:
          H = F.normalize(H, dim=1)

        Z = H

        Y_prob = self.classifier(Z).squeeze()
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, H

    def calculate_objective(self, X, Y):
        Y_prob, Y_hat, _ = self.forward(X)
        loss = torch.nn.functional.binary_cross_entropy(Y_prob, Y)

        return loss, Y_hat
    



# adapted from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
class GatedAttention(nn.Module):
    def __init__(self, branches=1, feature_norm=None):
        super(GatedAttention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = branches

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        if feature_norm is None:
          actv_norm = (nn.ReLU(),)
        elif feature_norm == 'softmax':
          actv_norm = (nn.Softmax(dim=1),)
        elif feature_norm == 'layernorm':
          actv_norm = (nn.ReLU(), nn.LayerNorm(500))
        elif feature_norm == 'l1':
          actv_norm = (nn.ReLU(),)
        else:
          assert False

        self.feat_l1_norm = feature_norm == 'l1'

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            *actv_norm
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)
        if self.feat_l1_norm:
          H = F.normalize(H, dim=1)

        A_V = self.attention_V(H)  # KxL
        # print('A_V', A_V.shape)
        A_U = self.attention_U(H)  # KxL
        # print('A_U', A_U.shape)
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        # print('A', A.shape)
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK

        raw_A = A
        A = F.softmax(A, dim=1)  # softmax over K
        # print('A', A.shape)

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM
        # print('Z', Z.shape)
        if self.ATTENTION_BRANCHES > 1:
          Z = Z.T.flatten().unsqueeze(0)

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A, raw_A, H

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat, _, _, _ = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, Y_hat
    

class GatedAttentionMulti(nn.Module):
    def __init__(self, branches=1, feature_norm=None, classes=5):
        super(GatedAttentionMulti, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = branches
        self.classes = classes

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        if feature_norm is None:
          actv_norm = (nn.ReLU(),)
        elif feature_norm == 'softmax':
          actv_norm = (nn.Softmax(dim=1),)
        elif feature_norm == 'layernorm':
          actv_norm = (nn.ReLU(), nn.LayerNorm(500))
        elif feature_norm == 'l1':
          actv_norm = (nn.ReLU(),)
        else:
          assert False

        self.feat_l1_norm = feature_norm == 'l1'

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            *actv_norm
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Linear(self.M*self.ATTENTION_BRANCHES, self.classes)

    def forward(self, x, softmax=True):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)
        if self.feat_l1_norm:
          H = F.normalize(H, dim=1)

        A_V = self.attention_V(H)  # KxL
        # print('A_V', A_V.shape)
        A_U = self.attention_U(H)  # KxL
        # print('A_U', A_U.shape)
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        # print('A', A.shape)
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK

        raw_A = A
        A = F.softmax(A, dim=1)  # softmax over K
        # print('A', A.shape)

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM
        # print('Z', Z.shape)
        if self.ATTENTION_BRANCHES > 1:
          Z = Z.T.flatten().unsqueeze(0)

        Y_prob = self.classifier(Z).squeeze()
        if softmax:
          Y_prob = torch.nn.functional.softmax(Y_prob, dim=-1)
        Y_hat = torch.argmax(Y_prob)

        return Y_prob, Y_hat, A, raw_A, H

    def calculate_objective(self, X, Y):
        Y_logit, Y_hat, A, _, _ = self.forward(X, softmax=False)
        loss = torch.nn.functional.cross_entropy(Y_logit, Y)

        return loss, Y_hat

# ---------------------------------------------------------------------------
# Transformer-based (TransMIL-style; Shao et al., 2021) and Mixture-of-Aggregators
# (MoA; Ozlugedik et al., 2025) MIL aggregators, adapted to our 50-cell bags and
# M=500 cell embeddings. MoA routes among 4 same-architecture experts with a linear
# router (top-2 gating, Gumbel noise, load-balancing loss); each expert returns a
# bag-level representation, the top-2 are weighted and summed, then a shared
# classifier is applied. Training loops are in training.py (train_batched / train_moa).
# ---------------------------------------------------------------------------


def _feature_extractor():
    part1 = nn.Sequential(
        nn.Conv2d(1, 20, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(20, 50, 5), nn.ReLU(), nn.MaxPool2d(2, 2))
    part2 = nn.Sequential(nn.Linear(50 * 4 * 4, 500), nn.ReLU())
    return part1, part2


class _ABMILExpert(nn.Module):
    """Gated-attention pooling with `branches` heads -> (M*branches) representation
    (matches GatedAttentionMulti's aggregation; no classifier)."""
    def __init__(self, M=500, L=128, branches=5):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(M, L), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(M, L), nn.Sigmoid())
        self.w = nn.Linear(L, branches)
        self.branches = branches
        self.out_dim = M * branches

    def forward(self, H):                                   # H: [N, M]
        A = self.w(self.V(H) * self.U(H))                   # [N, branches]
        A = F.softmax(A.transpose(0, 1), dim=1)             # [branches, N]
        Z = torch.mm(A, H)                                  # [branches, M]
        return Z.reshape(1, -1)                             # [1, M*branches]


class _TransMILExpert(nn.Module):
    """Permutation-invariant transformer (CLS token) -> dim-d representation."""
    def __init__(self, M=500, dim=500, depth=2, heads=10, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(M, dim)
        self.cls = nn.Parameter(torch.zeros(1, dim)); nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, H):                                   # H: [N, M]
        seq = torch.cat([self.cls, self.proj(H)], 0).unsqueeze(0)   # [1, N+1, dim]
        return self.norm(self.encoder(seq)[:, 0])           # [1, dim]


class TransMIL(nn.Module):
    """Standalone TransMIL aggregator (single expert + classifier), at our M=500
    embedding dim (10 heads, since 8 does not divide 500)."""
    def __init__(self, classes=4, dim=500, depth=2, heads=10, mlp_dim=1024):
        super().__init__()
        self.feature_extractor_part1, self.feature_extractor_part2 = _feature_extractor()
        self.expert = _TransMILExpert(500, dim, depth, heads, mlp_dim)
        self.classifier = nn.Linear(dim, classes)

    def _embed(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x).view(-1, 50 * 4 * 4)
        return self.feature_extractor_part2(H)

    def forward(self, x, softmax=True):
        logits = self.classifier(self.expert(self._embed(x))).squeeze()
        Y_prob = F.softmax(logits, dim=-1) if softmax else logits
        return Y_prob, torch.argmax(logits, dim=-1), None

    def calculate_objective(self, X, Y):
        logit, Y_hat, _ = self.forward(X, softmax=False)
        loss = F.cross_entropy(logit.unsqueeze(0), Y.unsqueeze(0) if Y.dim() == 0 else Y)
        return loss, Y_hat


class MoA(nn.Module):
    """Mixture of Aggregators: N same-arch experts + linear router, top-k gating."""
    def __init__(self, classes=4, base='abmil', n_experts=4, top_k=2, router_dim=128):
        super().__init__()
        self.feature_extractor_part1, self.feature_extractor_part2 = _feature_extractor()
        self.M = 500
        self.base, self.n_experts, self.top_k = base, n_experts, top_k
        if base == 'abmil':
            self.experts = nn.ModuleList([_ABMILExpert(self.M) for _ in range(n_experts)])
        elif base == 'transmil':
            self.experts = nn.ModuleList([_TransMILExpert(self.M) for _ in range(n_experts)])
        else:
            raise ValueError(base)
        rep_dim = self.experts[0].out_dim
        self.router_proj = nn.Linear(self.M, router_dim)
        self.router_lin = nn.Linear(router_dim, n_experts)
        self.classifier = nn.Linear(rep_dim, classes)

    def _embed(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x).view(-1, 50 * 4 * 4)
        return self.feature_extractor_part2(H)

    def _route(self, H, temperature, gumbel):
        logits = self.router_lin(self.router_proj(H).mean(dim=0))       # [n_experts]
        if gumbel and self.training:
            u = torch.rand_like(logits).clamp_(1e-9, 1 - 1e-9)
            logits = logits + (-torch.log(-torch.log(u)))
        return F.softmax(logits / temperature, dim=-1)                  # [n_experts]

    def forward(self, x, softmax=True, temperature=1.0, gumbel=False, dense=False):
        H = self._embed(x)
        probs = self._route(H, temperature, gumbel)                     # [n_experts]
        reps = torch.cat([e(H) for e in self.experts], dim=0)           # [n_experts, rep_dim]
        if dense:
            w = probs
        else:
            topv, topi = torch.topk(probs, self.top_k)
            w = torch.zeros_like(probs); w = w.scatter(0, topi, topv / topv.sum())
        z = (w.unsqueeze(-1) * reps).sum(dim=0, keepdim=True)           # [1, rep_dim]
        logits = self.classifier(z).squeeze()
        Y_prob = F.softmax(logits, dim=-1) if softmax else logits
        return Y_prob, torch.argmax(logits, dim=-1), probs

    def calculate_objective(self, X, Y):                               # CE only (LB added in train_moa)
        logit, Y_hat, _ = self.forward(X, softmax=False)
        loss = F.cross_entropy(logit.unsqueeze(0), Y.unsqueeze(0) if Y.dim() == 0 else Y)
        return loss, Y_hat
