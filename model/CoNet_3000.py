import torch
import torch.nn as nn
import sys
sys.path.append('/home/lvfengmao/wanggang/GeneInference/CoNet/model')

class CoNet_3000(nn.Module):
    def __init__(self, in_size, out_size, d_size, hidden_d1_size, hidden_d2_size, hidden_d3_size, dropout_rate_AE):
        super(CoNet_3000, self).__init__()
        self.fcnet = torch.nn.Sequential(
            torch.nn.Linear(in_size, d_size),
        )
        self.encoder = nn.Sequential(
            nn.Linear(out_size, hidden_d1_size),
            nn.Dropout(dropout_rate_AE),
            nn.ReLU(),
            nn.Linear(hidden_d1_size, d_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_size, hidden_d1_size),
            nn.Dropout(dropout_rate_AE),
            nn.ReLU(),
            nn.Linear(hidden_d1_size, out_size)
        )
    def forward(self, x, y):
        d = self.fcnet(x)
        y_fc = self.decoder(d)
        y = self.encoder(y)
        y_ae = self.decoder(y)
        return y_fc, y_ae
