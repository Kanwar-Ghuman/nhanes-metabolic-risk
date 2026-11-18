"""
Feed-forward neural network in PyTorch with a scikit-learn interface, so it can
sit inside the same Pipeline / GridSearchCV / permutation_importance code as
the other models.

Early stopping uses a validation split carved out of whatever data fit()
receives, so inside cross-validation it never sees the held-out fold.
"""
import copy

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn

# Tiny network: a single CPU thread is faster than GPU/MPS transfer overhead,
# avoids oversubscription when scikit-learn runs folds in parallel, and keeps runs reproducible
torch.set_num_threads(1)


class MLP(nn.Module):
    def __init__(self, n_features, hidden, dropout):
        super().__init__()
        layers, width = [], n_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden=(64, 32), dropout=0.2, lr=1e-3, weight_decay=1e-4, batch_size=128,
                 max_epochs=300, patience=20, val_fraction=0.15, random_state=0):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.val_fraction, stratify=y, random_state=self.random_state)
        X_tr, y_tr = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
        X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)

        self.model_ = MLP(X.shape[1], self.hidden, self.dropout)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(self.random_state)

        best_loss, best_state, wait = np.inf, None, 0
        self.val_loss_ = []
        for _ in range(self.max_epochs):
            self.model_.train()
            for idx in torch.randperm(len(X_tr), generator=gen).split(self.batch_size):
                opt.zero_grad()
                loss_fn(self.model_(X_tr[idx]), y_tr[idx]).backward()
                opt.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(X_val), y_val).item()
            self.val_loss_.append(val_loss)

            if val_loss < best_loss - 1e-4:
                best_loss, best_state, wait = val_loss, copy.deepcopy(self.model_.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        self.model_.load_state_dict(best_state)
        self.best_epoch_ = int(np.argmin(self.val_loss_)) + 1
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.as_tensor(np.asarray(X, dtype=np.float32)))
        p = torch.sigmoid(logits).numpy().astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
