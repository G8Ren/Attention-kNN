import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import TensorDataset, DataLoader

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess the dataset
df = pd.read_csv('WineQT.csv')
X = df.drop('quality', axis=1)
y = df['quality']

scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors and move to GPU
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device) - y_train.min()
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device) - y_test.min()

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, projected_dim, nhead, num_layers, output_dim, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, projected_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, projected_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=projected_dim, nhead=nhead, 
                                                   dim_feedforward=projected_dim*4, 
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(projected_dim, projected_dim)
        self.fc2 = nn.Linear(projected_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        src = self.input_projection(src)
        src = src + self.positional_encoding
        src = self.transformer_encoder(src)
        output = self.fc1(src[:, -1, :])
        output = self.activation(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output

# Model parameters
input_dim = X_train_tensor.shape[2]
projected_dim = 128
nhead = 8
num_layers = 4
output_dim = len(np.unique(y_train))
dropout = 0.3

# Create the model and move it to GPU
model = TransformerEncoderModel(input_dim, projected_dim, nhead, num_layers, output_dim, dropout).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training with K-fold cross-validation
num_epochs = 200
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold + 1}")
    
    model = TransformerEncoderModel(input_dim, projected_dim, nhead, num_layers, output_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    X_train_fold = X_train_tensor[train_idx]
    y_train_fold = y_train_tensor[train_idx]
    X_val_fold = X_train_tensor[val_idx]
    y_val_fold = y_train_tensor[val_idx]
    
    train_dataset = TensorDataset(X_train_fold, y_train_fold)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_fold)
            val_loss = criterion(val_outputs, y_val_fold)
            _, predicted = torch.max(val_outputs.data, 1)
            accuracy = accuracy_score(y_val_fold.cpu(), predicted.cpu())
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss.item():.4f}, Val Accuracy: {accuracy:.4f}')

# Final evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = accuracy_score(y_test_tensor.cpu(), predicted.cpu())
print(f'Final Test Accuracy: {accuracy:.4f}')

# KNN with Transformer features
with torch.no_grad():
    transformer_encoder_output = model.transformer_encoder(
        model.input_projection(X_train_tensor) + model.positional_encoding
    ).squeeze(1).cpu().numpy()

knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', weights='distance')
knn.fit(transformer_encoder_output, y_train_tensor.cpu().numpy())

# Evaluate the KNN model
with torch.no_grad():
    test_transformer_encoder_output = model.transformer_encoder(
        model.input_projection(X_test_tensor) + model.positional_encoding
    ).squeeze(1).cpu().numpy()

predicted_knn = knn.predict(test_transformer_encoder_output)
accuracy_knn = accuracy_score(y_test_tensor.cpu().numpy(), predicted_knn)
print(f'Test Accuracy with KNN (BallTree): {accuracy_knn:.4f}')

# Ensemble prediction (Average of Transformer and KNN)
transformer_proba = nn.functional.softmax(outputs, dim=1).cpu().numpy()
knn_proba = knn.predict_proba(test_transformer_encoder_output)
ensemble_proba = (transformer_proba + knn_proba) / 2
ensemble_predicted = np.argmax(ensemble_proba, axis=1)
accuracy_ensemble = accuracy_score(y_test_tensor.cpu().numpy(), ensemble_predicted)
print(f'Test Accuracy with Ensemble: {accuracy_ensemble:.4f}')