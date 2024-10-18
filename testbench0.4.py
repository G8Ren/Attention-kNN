import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import networkx as nx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess the dataset
df = pd.read_csv('WineQT.csv')
X = df.drop('quality', axis=1)
y = df['quality']

scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

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
        self.fc1 = nn.Linear(projected_dim, projected_dim*2)
        self.fc2 = nn.Linear(projected_dim*2, projected_dim)
        self.fc3 = nn.Linear(projected_dim, output_dim)
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
        output = self.activation(output)
        output = self.dropout(output)
        output = self.fc3(output)
        return output

# Model parameters
input_dim = X_train_tensor.shape[2]
projected_dim = 256
nhead = 8
num_layers = 6
output_dim = len(np.unique(y_train))
dropout = 0.2

def train_model(model, train_loader, val_data, num_epochs, patience=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_accuracy = 0
    epochs_without_improvement = 0

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
            val_outputs = model(val_data[0])
            val_loss = criterion(val_outputs, val_data[1])
            _, predicted = torch.max(val_outputs.data, 1)
            accuracy = accuracy_score(val_data[1].cpu(), predicted.cpu())
        
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss.item():.4f}, Val Accuracy: {accuracy:.4f}')

    return model

# K-fold cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold + 1}")
    
    model = TransformerEncoderModel(input_dim, projected_dim, nhead, num_layers, output_dim, dropout).to(device)
    
    X_train_fold = X_train_tensor[train_idx]
    y_train_fold = y_train_tensor[train_idx]
    X_val_fold = X_train_tensor[val_idx]
    y_val_fold = y_train_tensor[val_idx]
    
    train_dataset = TensorDataset(X_train_fold, y_train_fold)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = train_model(model, train_loader, (X_val_fold, y_val_fold), num_epochs=300)

# Feature Engineering
def extract_features(model, X):
    with torch.no_grad():
        transformer_output = model.transformer_encoder(
            model.input_projection(X) + model.positional_encoding
        )
        features = torch.cat([
            transformer_output[:, -1, :],
            torch.mean(transformer_output, dim=1),
            torch.max(transformer_output, dim=1)[0]
        ], dim=1)
    return features.cpu().numpy()

train_features = extract_features(model, X_train_tensor)
test_features = extract_features(model, X_test_tensor)

pca = PCA(n_components=0.95)
train_features_pca = pca.fit_transform(train_features)
test_features_pca = pca.transform(test_features)

knn_variants = {
    'uniform': KNeighborsClassifier(weights='uniform'),
    'distance': KNeighborsClassifier(weights='distance'),
    'ball_tree': KNeighborsClassifier(algorithm='ball_tree'),
    'kd_tree': KNeighborsClassifier(algorithm='kd_tree')
}

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

best_accuracy = 0
best_knn = None

for name, knn in knn_variants.items():
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(train_features_pca, y_train_tensor.cpu().numpy())
    
    accuracy = grid_search.score(test_features_pca, y_test_tensor.cpu().numpy())
    print(f"{name} KNN Accuracy: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_knn = grid_search.best_estimator_

print(f"Best KNN Accuracy: {best_accuracy:.4f}")
print(f"Best KNN Parameters: {best_knn.get_params()}")

transformer_proba = nn.functional.softmax(model(X_test_tensor), dim=1).detach().cpu().numpy()
knn_proba = best_knn.predict_proba(test_features_pca)
ensemble_proba = (transformer_proba + knn_proba) / 2
ensemble_predicted = np.argmax(ensemble_proba, axis=1)
accuracy_ensemble = accuracy_score(y_test_tensor.cpu().numpy(), ensemble_predicted)
print(f'Test Accuracy with Ensemble: {accuracy_ensemble:.4f}')

# Diagram generation functions (same as before)
# ...
def create_model_architecture_diagram():
    G = nx.DiGraph()
    nodes = [
        "Input Data", "Transformer Encoder", "Feature Extraction", "Concatenate Features",
        "PCA", "KNN Variants", "Grid Search", "Best KNN Model", "Transformer Prediction",
        "KNN Prediction", "Ensemble Prediction"
    ]
    G.add_nodes_from(nodes)
    edges = [
        ("Input Data", "Transformer Encoder"),
        ("Transformer Encoder", "Feature Extraction"),
        ("Feature Extraction", "Concatenate Features"),
        ("Concatenate Features", "PCA"),
        ("PCA", "KNN Variants"),
        ("KNN Variants", "Grid Search"),
        ("Grid Search", "Best KNN Model"),
        ("Transformer Encoder", "Transformer Prediction"),
        ("Best KNN Model", "KNN Prediction"),
        ("Transformer Prediction", "Ensemble Prediction"),
        ("KNN Prediction", "Ensemble Prediction")
    ]
    G.add_edges_from(edges)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=8, font_weight='bold', arrows=True)
    nx.draw_networkx_labels(G, pos)
    plt.title("Improved Transformer+KNN Model Architecture")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('model_architecture.png')
    plt.close()

def create_feature_engineering_diagram():
    G = nx.DiGraph()
    nodes = [
        "Raw Features", "Transformer Encoder", "Feature Extraction", "Last Token Features",
        "Mean Pooling Features", "Max Pooling Features", "Concatenated Features",
        "PCA", "Reduced Feature Set", "KNN Input"
    ]
    G.add_nodes_from(nodes)
    edges = [
        ("Raw Features", "Transformer Encoder"),
        ("Transformer Encoder", "Feature Extraction"),
        ("Feature Extraction", "Last Token Features"),
        ("Feature Extraction", "Mean Pooling Features"),
        ("Feature Extraction", "Max Pooling Features"),
        ("Last Token Features", "Concatenated Features"),
        ("Mean Pooling Features", "Concatenated Features"),
        ("Max Pooling Features", "Concatenated Features"),
        ("Concatenated Features", "PCA"),
        ("PCA", "Reduced Feature Set"),
        ("Reduced Feature Set", "KNN Input")
    ]
    G.add_edges_from(edges)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=3000, font_size=8, font_weight='bold', arrows=True)
    nx.draw_networkx_labels(G, pos)
    plt.title("Feature Engineering Process")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('feature_engineering.png')
    plt.close()
# Generate the diagrams
create_model_architecture_diagram()
create_feature_engineering_diagram()

print("Diagrams have been generated and saved as 'model_architecture.png' and 'feature_engineering.png'.")