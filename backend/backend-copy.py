import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import anvil.server

# CNN MODEL DEFINITION
class ConnectFourWiderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 7 * 8, 1024)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 7)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# TRANSFORMER MODEL DEFINITION
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=42):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x * math.sqrt(x.size(-1)) + self.pe[:, :x.size(1)]


class ConnectFourTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=3, dim_ff=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers
        )

        self.fc_out = nn.Linear(d_model * 42, 7)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)      # (N, 6, 7, 2)
        x = x.reshape(x.size(0), 42, 2)

        x = self.embedding(x)
        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)

        x = x.reshape(x.size(0), -1)
        return self.fc_out(x)


# LOAD MODELS
print("Loading models...")
device = torch.device("cpu")

# ---- CNN (raw state_dict) ----
cnn_model = ConnectFourWiderCNN().to(device)
cnn_model.load_state_dict(
    torch.load("/connect4/connect4_cnn.pth", map_location=device)
)
cnn_model.eval()
print("✓ CNN loaded")

# ---- TRANSFORMER (checkpoint) ----
transformer_model = ConnectFourTransformer().to(device)

transformer_checkpoint = torch.load(
    "/connect4/transformer_model.pth",
    map_location=device
)

transformer_model.load_state_dict(
    transformer_checkpoint["model_state_dict"]
)
transformer_model.eval()
print("✓ Transformer loaded")

print("✓ Models loaded successfully")


# SHARED PREPROCESSING
def board_to_tensor(board_2d):
    tensor = np.zeros((6, 7, 2), dtype=np.float32)
    for r in range(6):
        for c in range(7):
            if board_2d[r][c] == 1:
                tensor[r, c, 0] = 1
            elif board_2d[r][c] == 2:
                tensor[r, c, 1] = 1
    return torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0)


# PREDICTION API
@anvil.server.callable
def predict(board_2d, model_type="cnn"):
    print(f"Prediction request using model: {model_type}")

    x = board_to_tensor(board_2d).float().to(device)

    model = cnn_model if model_type == "cnn" else transformer_model

    with torch.no_grad():
        logits = model(x)
        move = int(torch.argmax(logits, dim=1).item())

    updated_board = [row[:] for row in board_2d]
    placed = False

    for row in reversed(range(6)):
        if updated_board[row][move] == 0:
            updated_board[row][move] = 2
            placed = True
            break

    if not placed:
        for c in range(7):
            for row in reversed(range(6)):
                if updated_board[row][c] == 0:
                    updated_board[row][c] = 2
                    move = c
                    placed = True
                    break
            if placed:
                break

    return move, updated_board


# ANVIL UPLINK
print("Connecting to Anvil uplink...")
try:
    anvil.server.connect("server_PD6OWDY4WQBZ74GBG3X6K25W-HD2M4NJEKKLE5FUS")
    print("✓ Backend connected successfully!")
    print("Waiting for calls...")
    anvil.server.wait_forever()
except Exception as e:
    print(f"✗ Connection failed: {e}")
    import traceback
    traceback.print_exc()
