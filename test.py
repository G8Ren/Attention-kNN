import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess data
df = pd.read_csv('WineQT.csv')
X = df.drop('quality', axis=1)
y = df['quality']
y = (y >= 6).astype(int)

# Feature engineering
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_train_tensor = torch.LongTensor(y_train.values).to(device)
y_test_tensor = torch.LongTensor(y_test.values).to(device)

class ImprovedAttentionModel(nn.Module):
    def __init__(self, input_dim, attention_dim, hidden_dim, num_classes, dropout_rate=0.5):
        super(ImprovedAttentionModel, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1)
        )
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        weighted_features = x * attention_weights
        x = self.dropout(F.leaky_relu(self.bn1(self.fc1(weighted_features))))
        x = self.dropout(F.leaky_relu(self.bn2(self.fc2(x))))
        output = self.fc3(x)
        return output, weighted_features, attention_weights

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=200):
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
attention_dim = 128
hidden_dim = 256
num_classes = 2
dropout_rate = 0.5
learning_rate = 0.001
batch_size = 64
num_epochs = 300

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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
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
final_optimizer = optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=1e-4)
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

# Create an ensemble of multiple models
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

knn.fit(train_features, y_train)
rf.fit(train_features, y_train)
gb.fit(train_features, y_train)

knn_predictions = knn.predict(test_features)
rf_predictions = rf.predict(test_features)
gb_predictions = gb.predict(test_features)
nn_predictions = final_model(X_test_tensor)[0].argmax(dim=1).cpu().numpy()

ensemble_predictions = np.round((knn_predictions + rf_predictions + gb_predictions + nn_predictions) / 4).astype(int)
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)

print(f"KNN Accuracy: {accuracy_score(y_test, knn_predictions):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_predictions):.4f}")
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_predictions):.4f}")
print(f"Neural Network Accuracy: {accuracy_score(y_test, nn_predictions):.4f}")
print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")