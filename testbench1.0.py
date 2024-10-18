import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess the dataset
df = pd.read_csv('WineQT.csv')
X = df.drop('quality', axis=1)
y = df['quality']
y = (y >= 6).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_train_tensor = torch.LongTensor(y_train.values).to(device)
y_test_tensor = torch.LongTensor(y_test.values).to(device)

class AttentionFeatureExtractor(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionFeatureExtractor, self).__init__()
        self.attention = nn.Linear(input_dim, attention_dim)
        self.context = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(self, x):
        attention_weights = F.relu(self.attention(x))  # Using ReLU instead of tanh
        attention_weights = self.context(attention_weights)
        attention_weights = F.softmax(attention_weights, dim=1)
        weighted_features = x * attention_weights
        return weighted_features, attention_weights

class ImprovedAttentionModel(nn.Module):
    def __init__(self, input_dim, attention_dim, hidden_dim, num_classes, dropout_rate=0.5):
        super(ImprovedAttentionModel, self).__init__()
        self.feature_extractor = AttentionFeatureExtractor(input_dim, attention_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        weighted_features, attention_weights = self.feature_extractor(x)
        x = self.dropout(F.relu(self.bn1(self.fc1(weighted_features))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        output = self.fc3(x)
        return output, weighted_features, attention_weights

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs, _, _ = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        scheduler.step(loss)
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs, _, _ = model(X_test_tensor)
                _, predicted = torch.max(test_outputs.data, 1)
                accuracy = accuracy_score(y_test_tensor.cpu(), predicted.cpu())
                print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.4f}')

    return model

input_dim = X_train_scaled.shape[1]
attention_dim = 64
hidden_dim = 128
num_classes = 2
dropout_rate = 0.5
learning_rate = 0.001
batch_size = 64
num_epochs = 200

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    print(f"Fold {fold + 1}")
    
    X_train_fold = X_train_tensor[train_idx]
    y_train_fold = y_train_tensor[train_idx]
    X_val_fold = X_train_tensor[val_idx]
    y_val_fold = y_train_tensor[val_idx]
    
    model = ImprovedAttentionModel(input_dim, attention_dim, hidden_dim, num_classes, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
    
    train_dataset = TensorDataset(X_train_fold, y_train_fold)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs)
    
    model.eval()
    with torch.no_grad():
        val_outputs, _, _ = model(X_val_fold)
        _, predicted = torch.max(val_outputs.data, 1)
        accuracy = accuracy_score(y_val_fold.cpu(), predicted.cpu())
        cv_scores.append(accuracy)
        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Train the final model on the entire training set
final_model = ImprovedAttentionModel(input_dim, attention_dim, hidden_dim, num_classes, dropout_rate).to(device)
final_criterion = nn.CrossEntropyLoss()
final_optimizer = optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=1e-5)
final_scheduler = ReduceLROnPlateau(final_optimizer, 'min', patience=10, factor=0.5, verbose=True)

final_train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
final_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True)

final_model = train_model(final_model, final_train_loader, final_criterion, final_optimizer, final_scheduler, num_epochs)

# Extract attention-weighted features
final_model.eval()
with torch.no_grad():
    _, train_features, _ = final_model(X_train_tensor)
    _, test_features, _ = final_model(X_test_tensor)

train_features = train_features.cpu().numpy()
test_features = test_features.cpu().numpy()

# Apply KNN on attention-weighted features
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_features, y_train)
knn_predictions = knn.predict(test_features)
knn_accuracy = accuracy_score(y_test, knn_predictions)

print(f"KNN Accuracy on attention-weighted features: {knn_accuracy:.4f}")

# Create an ensemble of the neural network and KNN
nn_predictions = final_model(X_test_tensor)[0].argmax(dim=1).cpu().numpy()
ensemble_predictions = np.round((nn_predictions + knn_predictions) / 2).astype(int)
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)

print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")

# After training, calculate feature importance
def calculate_feature_importance(model, X):
    model.eval()
    with torch.no_grad():
        _, _, attention_weights = model(X)
    
    # Print shape and type of attention_weights for debugging
    print(f"Shape of attention_weights: {attention_weights.shape}")
    print(f"Type of attention_weights: {type(attention_weights)}")
    
    # Ensure attention_weights is 2D
    if attention_weights.dim() == 3:
        attention_weights = attention_weights.squeeze(2)
    elif attention_weights.dim() == 1:
        attention_weights = attention_weights.unsqueeze(1)
    
    # Average attention weights across all samples
    feature_importance = attention_weights.mean(dim=0).cpu().numpy()
    
    # Print shape and type of feature_importance for debugging
    print(f"Shape of feature_importance: {feature_importance.shape}")
    print(f"Type of feature_importance: {type(feature_importance)}")
    
    # Ensure feature_importance is 1D
    if feature_importance.ndim == 0:
        feature_importance = np.array([feature_importance])
    
    # Normalize importance scores
    feature_importance = feature_importance / feature_importance.sum()
    
    return feature_importance

# Calculate feature importance
feature_importance = calculate_feature_importance(final_model, X_train_tensor)

# Print feature importance
feature_names = X.columns
for i, feature in enumerate(feature_names):
    if i < len(feature_importance):
        print(f"{feature}: {feature_importance[i]:.4f}")
    else:
        print(f"Warning: No importance score for {feature}")

# Visualize feature importance
plt.figure(figsize=(12, 6))
bars = plt.bar(feature_names, feature_importance)
plt.title("Feature Importance", fontsize=16)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Importance", fontsize=12)
plt.xticks(rotation=45, ha='right')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

plt.tight_layout()

# Show the plot in a pop-up window
plt.show()

print("Feature importance plot displayed in a pop-up window.")