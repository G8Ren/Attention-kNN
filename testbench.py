import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load and preprocess the dataset
df = pd.read_csv('WineQT.csv')
X = df.drop('quality', axis=1)
y = df['quality']

# Apply robust scaling to features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train).float().unsqueeze(1)
X_test_tensor = torch.tensor(X_test).float().unsqueeze(1)
y_train_tensor = torch.tensor(y_train.values).long() - y_train.min()
y_test_tensor = torch.tensor(y_test.values).long() - y_test.min()

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, projected_dim, nhead, num_layers, output_dim, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, projected_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=projected_dim, nhead=nhead, 
                                                   dim_feedforward=projected_dim*4, 
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(projected_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.input_projection(src)
        src = self.transformer_encoder(src)
        output = self.fc(self.dropout(src[:, -1, :]))
        return output

# Model parameters
input_dim = X_train_tensor.shape[2]
projected_dim = 64  # Increased from 16
nhead = 8  # Increased from 4
num_layers = 3  # Increased from 1
output_dim = len(np.unique(y_train))
dropout = 0.2

# Create and train the model
model = TransformerEncoderModel(input_dim, projected_dim, nhead, num_layers, output_dim, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

num_epochs = 100  # Increased from 10
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
    
    scheduler.step(val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = accuracy_score(y_test_tensor, predicted)
print(f'Test Accuracy: {accuracy:.4f}')

# Use the trained Transformer encoder with k-NN
transformer_encoder_output = model.transformer_encoder(model.input_projection(X_train_tensor)).squeeze(1)
knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
knn.fit(transformer_encoder_output.detach().numpy(), y_train_tensor.numpy())

# Evaluate the k-NN model
test_transformer_encoder_output = model.transformer_encoder(model.input_projection(X_test_tensor)).squeeze(1)
predicted_knn = knn.predict(test_transformer_encoder_output.detach().numpy())
accuracy_knn = accuracy_score(y_test_tensor.numpy(), predicted_knn)
print(f'Test Accuracy with k-NN (BallTree): {accuracy_knn:.4f}')