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
