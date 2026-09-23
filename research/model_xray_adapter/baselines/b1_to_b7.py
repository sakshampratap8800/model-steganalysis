"""
ModelXRay Baselines B1 - B7
Attribution: Gilkarov & Dubin (ModelXRay, arXiv:2409.19310)
Independent reimplementation.
"""
import numpy as np
import xgboost as xgb
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn

# B1: Flat weights + XGBoost
class BaselineB1:
    def __init__(self):
        self.model = xgb.XGBClassifier()
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)

# B2: NIST Statistical Features (Yin et al.)
def extract_b2_features(weights: np.ndarray) -> np.ndarray:
    """
    Extracts 92-dimensional NIST statistical features from IEEE-754 mantissa bits.
    4 statistical functions (mean, std, var, median) applied to 23 mantissa bit arrays.
    """
    w_float = weights.astype(np.float32).flatten()
    w_int = w_float.view(np.uint32)
    
    # Extract all 23 mantissa bit planes
    bit_planes = []
    for i in range(23):
        bit_plane = (w_int >> i) & 1
        bit_planes.append(bit_plane)
        
    features = []
    for bit_plane in bit_planes:
        features.extend([
            np.mean(bit_plane),
            np.std(bit_plane),
            np.var(bit_plane),
            np.median(bit_plane)
        ])
    return np.array(features)

class BaselineB2:
    def __init__(self):
        self.model = xgb.XGBClassifier()
    def fit(self, X_weights, y):
        features = np.array([extract_b2_features(w) for w in X_weights])
        self.model.fit(features, y)
    def predict(self, X_weights):
        features = np.array([extract_b2_features(w) for w in X_weights])
        return self.model.predict(features)
    def predict_proba(self, X_weights):
        features = np.array([extract_b2_features(w) for w in X_weights])
        return self.model.predict_proba(features)

# B3: MalConv-lite (1D CNN on raw bytes)
class MalConvLite(nn.Module):
    def __init__(self, vocab_size=256, embedding_dim=8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, 16, kernel_size=8, stride=4)
        self.fc = nn.Linear(16, 1)
    
    def forward(self, x):
        x = self.emb(x).transpose(1, 2)
        x = torch.relu(self.conv(x))
        x = torch.max(x, dim=-1)[0]
        return torch.sigmoid(self.fc(x))

# B4: Byte Autocorrelation
class BaselineB4:
    def __init__(self):
        self.clf = LogisticRegression()
    def _extract(self, bytes_array):
        if len(bytes_array) < 2: return 0.0
        return np.corrcoef(bytes_array[:-1], bytes_array[1:])[0, 1]
    def fit(self, X_bytes, y):
        feats = np.array([self._extract(b) for b in X_bytes]).reshape(-1, 1)
        self.clf.fit(feats, y)
    def predict_proba(self, X_bytes):
        feats = np.array([self._extract(b) for b in X_bytes]).reshape(-1, 1)
        return self.clf.predict_proba(feats)

# B5: Byte Entropy
class BaselineB5:
    def __init__(self):
        self.clf = LogisticRegression()
    def _extract(self, byte_array):
        counts = np.bincount(byte_array, minlength=256)
        probs = counts[counts > 0] / len(byte_array)
        return entropy(probs, base=2)
    def fit(self, X_bytes, y):
        feats = np.array([self._extract(b) for b in X_bytes]).reshape(-1, 1)
        self.clf.fit(feats, y)
    def predict_proba(self, X_bytes):
        feats = np.array([self._extract(b) for b in X_bytes]).reshape(-1, 1)
        return self.clf.predict_proba(feats)

# B6: Histogram KL Divergence
class BaselineB6:
    def __init__(self):
        self.clf = LogisticRegression()
        self.clean_ref = None
    def fit(self, X_hists, y):
        # Assumes y=0 is clean. Average clean hists for reference.
        clean_hists = [x for x, label in zip(X_hists, y) if label == 0]
        if not clean_hists:
            self.clean_ref = np.ones(256) / 256.0
        else:
            self.clean_ref = np.mean(clean_hists, axis=0) + 1e-9 # avoid div by zero
        
        feats = np.array([entropy(h + 1e-9, self.clean_ref) for h in X_hists]).reshape(-1, 1)
        self.clf.fit(feats, y)
    def predict_proba(self, X_hists):
        feats = np.array([entropy(h + 1e-9, self.clean_ref) for h in X_hists]).reshape(-1, 1)
        return self.clf.predict_proba(feats)

# B7: Weight Value Distribution (Kolmogorov-Smirnov distance)
class BaselineB7:
    def __init__(self):
        self.clf = LogisticRegression()
        self.clean_ref = None
    def fit(self, X_weights, y):
        from scipy.stats import ks_2samp
        # Gather a sample of clean weights for reference
        clean_weights = np.concatenate([x.flatten() for x, label in zip(X_weights, y) if label == 0])
        # Subsample if too large
        if len(clean_weights) > 10000:
            clean_weights = np.random.choice(clean_weights, 10000, replace=False)
        self.clean_ref = clean_weights
        
        feats = np.array([ks_2samp(w.flatten(), self.clean_ref).statistic for w in X_weights]).reshape(-1, 1)
        self.clf.fit(feats, y)
    def predict_proba(self, X_weights):
        from scipy.stats import ks_2samp
        feats = np.array([ks_2samp(w.flatten(), self.clean_ref).statistic for w in X_weights]).reshape(-1, 1)
        return self.clf.predict_proba(feats)
