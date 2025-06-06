import torch.nn as nn
class CNNLSTM_ImageRadFusion(nn.Module):
    def __init__(self, input_dim=513, hidden_dim=128):
        super(CNNLSTM_ImageRadFusion, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: (B, T, 513)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # lấy đầu ra thời điểm cuối
        out = self.fc(out)
        return out.squeeze()