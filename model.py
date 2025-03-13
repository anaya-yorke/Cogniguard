import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    def __init__(self, input_channels=8, sequence_length=250):
        super(CNNLSTM, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        
        self.fc1 = nn.Linear(64 * 2, 32)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = self.dropout(x)
        
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        
        x = F.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x

def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, learning_rate=0.001):
    model = CNNLSTM(input_channels=X_train.shape[2], sequence_length=X_train.shape[1])
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
    for epoch in range(epochs):
        model.train()
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_preds = (val_outputs > 0.5).float()
            accuracy = (val_preds == y_val_tensor).float().mean()
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss.item():.4f}, Accuracy: {accuracy.item():.4f}')
    
    return model

def transfer_learning(pretrained_model, X_fine_tune, y_fine_tune, epochs=20, batch_size=32, learning_rate=0.0001):
    model = pretrained_model
    
    for param in model.conv1.parameters():
        param.requires_grad = False
    
    for param in model.conv2.parameters():
        param.requires_grad = False
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    X_fine_tune_tensor = torch.FloatTensor(X_fine_tune)
    y_fine_tune_tensor = torch.FloatTensor(y_fine_tune).unsqueeze(1)
    
    for epoch in range(epochs):
        model.train()
        
        for i in range(0, len(X_fine_tune), batch_size):
            batch_X = X_fine_tune_tensor[i:i+batch_size]
            batch_y = y_fine_tune_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    return model 