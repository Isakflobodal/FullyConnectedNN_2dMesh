
import torch.nn as nn
import torch
import torch_geometric.nn as pyg_nn
from torch.nn.functional import conv2d
from createdata import dim, step


class NeuralNet(nn.Module):
    def __init__(self, layer_sizes, dropout, device="cuda"):
        super(NeuralNet, self).__init__()
        self.device = device

        layers = []
        input_features = (dim*dim*2 + 8)
        for i, output_features in enumerate(layer_sizes):
            layers.append(nn.Linear(input_features, output_features))
            if i != len(layer_sizes) - 1:
                layers.append(nn.ReLU())
                #layers.append(nn.BatchNorm1d(output_features))
                layers.append(nn.Dropout(dropout))
            input_features = output_features

        self.layers = nn.Sequential(*layers)    

    # forward function: 
    def forward(self, Pc, pts):             # [B,P=4,2] [B,dim*dim,2]
        B = Pc.shape[0]
        Pc = Pc.view(B,-1)                  # [B,8]
        pts = pts.view(B,-1)                # [B,dim*dim*2]
        x = torch.cat((Pc, pts), dim=1)     # [B,dim*dim*2+8]
        pred = self.layers(x)               # [B,dim*dim]
        pred = pred.view(B,1,dim,dim)       # [B,1,dim,dim]

        # scond order schemes
        dx_kernel = torch.zeros(5,5,dtype=pred.dtype, device= self.device)
        dx_kernel[2] = torch.tensor([1/12, -2/3, 0, 2/3, -1/12], dtype=pred.dtype, device= self.device)/step
        dx_kernel = dx_kernel.view(1, 1, 5, 5)

        dy_kernel = dx_kernel.transpose(2,3)
        f_x = conv2d(pred, dx_kernel)
        f_y = conv2d(pred, dy_kernel)
        comb = torch.cat([f_y, f_x], 1) # [B,2, H, W]
        f_norm = torch.norm(comb, dim=1)

        pred = pred.squeeze()

        return pred, f_norm, f_x, f_y