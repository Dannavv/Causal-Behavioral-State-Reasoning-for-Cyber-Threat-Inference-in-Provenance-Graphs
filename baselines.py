import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest


class PyTorchAutoencoder(nn.Module if _HAS_TORCH else object):
    def __init__(self, input_dim, hidden_dim=8):
        if not _HAS_TORCH:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")
        super(PyTorchAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_and_evaluate_ae(train_data, test_data, epochs=50, batch_size=32, lr=0.001):
    if not _HAS_TORCH:
        raise ImportError("PyTorch is not installed. Install with: pip install torch")
        
    input_dim = train_data.shape[1]
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PyTorchAutoencoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Standardize data
    train_mean = np.mean(train_data, axis=0)
    train_std = np.std(train_data, axis=0) + 1e-9
    train_norm = (train_data - train_mean) / train_std
    test_norm = (test_data - train_mean) / train_std
    
    train_tensor = torch.FloatTensor(train_norm).to(device)
    test_tensor = torch.FloatTensor(test_norm).to(device)
    
    dataset = TensorDataset(train_tensor, train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        reconstructed = model(test_tensor)
        # MSE per sample
        mse = torch.mean((test_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
    
    return mse # Higher MSE = more anomalous

def train_ae_and_get_scores(train_data, eval_data, epochs=50, batch_size=32, lr=0.001, seed=42):
    """
    Train AE on train_data and return (train_scores, eval_scores) from the same model.
    Provides a consistent threshold: set threshold = percentile(train_scores, 95) then
    flag eval_data windows where eval_scores > threshold as anomalous.
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch is not installed.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = train_data.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyTorchAutoencoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_mean = np.mean(train_data, axis=0)
    train_std = np.std(train_data, axis=0) + 1e-9
    train_norm = (train_data - train_mean) / train_std
    eval_norm = (eval_data - train_mean) / train_std

    train_tensor = torch.FloatTensor(train_norm).to(device)
    eval_tensor = torch.FloatTensor(eval_norm).to(device)

    dataset = TensorDataset(train_tensor, train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_scores = torch.mean((train_tensor - model(train_tensor)) ** 2, dim=1).cpu().numpy()
        eval_scores = torch.mean((eval_tensor - model(eval_tensor)) ** 2, dim=1).cpu().numpy()

    return train_scores, eval_scores


def run_classical_baselines(train_data, test_data, contamination=0.05):
    # Isolation Forest
    iforest = IsolationForest(contamination=contamination, random_state=42)
    iforest.fit(train_data)
    iforest_scores = -iforest.score_samples(test_data)
    
    # LOF
    lof = LocalOutlierFactor(contamination=contamination, novelty=True)
    lof.fit(train_data)
    lof_scores = -lof.score_samples(test_data)
    
    # One-Class SVM
    ocsvm = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale')
    ocsvm.fit(train_data)
    ocsvm_scores = -ocsvm.score_samples(test_data)
    
    return iforest_scores, lof_scores, ocsvm_scores
