
import math
import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F
import random
from PIL import Image
import skfmm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from createdata import dim, max_xy, step, X, Y, format_ax, PlotDistanceField


def GetDF(pts):
    distances = torch.sqrt((pts[:,0][:,None] - X.ravel())**2 + (pts[:,1][:,None] - Y.ravel())**2)
    min_dist, _ = distances.min(axis=0)
    return min_dist.reshape([dim,dim])

class ContourDataset(Dataset):
    def __init__(self, root_dir="data", split="train", contour_scaler=None, grid_scaler=None):
        self.data_path = f"{root_dir}/{split}" 
        self.data = [folder for folder in os.listdir(self.data_path)]
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        folder_name = self.data[idx]
        full_path = f"{self.data_path}/{folder_name}"
        data = torch.load(f'{full_path}/data.pth')

        Pc = torch.from_numpy(data['Pc']).float()[:,:-1]
        Pi = torch.from_numpy(data['mesh_pts']).float()[:,:-1]
        pts = data['pts'][:,:-1]
        pts = torch.from_numpy(pts).float()
        df = data['df']                 
        df = df.view([dim,dim]).float()         # [256,256]

        # # rotation
        # if self.split == "train":
        #     phi = random.uniform(0,2*math.pi)
        #     rot_mat = torch.tensor([[math.cos(phi), -math.sin(phi)],[math.sin(phi), math.cos(phi)]])

        #     #get distance field
        #     Pc = Pc @ rot_mat.T
        #     Pi = Pi @ rot_mat.T
        #     df_rot = GetDF(Pi)
        #     df = df_rot.view([dim,dim]).float()         # [256,256]        

        return Pc, df, pts


def main():
    dataset = ContourDataset(split="train")
    Pc, df, _ = dataset[0]

    P_list = Pc.tolist()
    P_list.append(P_list[0])
    xs,ys = zip(*P_list)

    fig, axs = plt.subplots(1,2, figsize=(12,8))
    
    for pt in Pc:
        axs[0].plot(pt[0],pt[1], "r.")

    axs[0].set_title('Mesh vertices and contour')
    axs[0].axis('off')
    axs[0].plot(xs,ys)

    pc = axs[1].pcolormesh(X,Y, df.squeeze().detach().numpy(), cmap='terrain')
    format_ax(axs[1], pc, fig)
    axs[1].set_title('Distance field')

    plt.show()



if __name__ == "__main__":
    main()