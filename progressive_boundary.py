# progressive_sharpness_v2.py
import argparse, torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import transforms, datasets
import timm, kornia, os, warnings
import torch.nn.functional as F
from tqdm import tqdm, trange
warnings.filterwarnings("ignore")

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', default='resnet18')
    p.add_argument('--scoring',  choices=['energy','msp','maha','maxlogit'], default='energy')
    p.add_argument('--epochs',   type=int, default=100)
    p.add_argument('--batch',    type=int, default=256)
    p.add_argument('--lr',       type=float, default=3e-4)
    p.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

# 1. Universal backbone + scoring wrapper
class UniversalScorer(nn.Module):
    def __init__(self, backbone_name, scoring='energy'):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)  # feature extractor
        self.in_features = self.backbone.num_features
        self.scoring = scoring
        if scoring in ['energy','maxlogit']:
            self.head = nn.Linear(self.in_features, 10)  # 10 CIFAR classes
        elif scoring == 'msp':
            self.head = nn.Linear(self.in_features, 10)
        elif scoring == 'maha':
            self.register_buffer('mean', torch.zeros(10, self.in_features))
            self.register_buffer('inv_cov', torch.eye(self.in_features))

    def forward(self, x, return_feat=False):
        z = self.backbone(x)
        if return_feat:
            return z
        if self.scoring == 'energy':
            logits = self.head(z)
            return -torch.logsumexp(logits, dim=1)
        elif self.scoring == 'msp':
            probs = self.head(z).softmax(1)
            return probs.max(1)[0]          # higher = more ID
        elif self.scoring == 'maxlogit':
            return self.head(z).max(1)[0]
        elif self.scoring == 'maha':
            # Mahalanobis distance to closest class centroid
            dists = torch.cdist(z.unsqueeze(0), self.mean.unsqueeze(0))[0]
            return -dists.min(1)[0]         # higher = more ID

# 2. Progressive sharpness on two CIFAR classes
def main():
    args = parse()
    device = args.device
    tf = transforms.Compose([
        transforms.Resize(224) if 'vit' in args.backbone else transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3,(0.5,)*3)
    ])
    train = datasets.CIFAR10(root='/home/tanmoy/research/data', train=True, download=True, transform=tf)
    idx = (np.array(train.targets)==3)|(np.array(train.targets)==5)
    train.data, train.targets = train.data[idx], np.array(train.targets)[idx]
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True)

    model = UniversalScorer(args.backbone, args.scoring).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # PCA grid for viz
    X_raw = torch.cat([x for x,_ in loader])  # Shape: (N, C, H, W)
    X_raw = X_raw.permute(0, 2, 3, 1).contiguous()  # Shape: (N, H, W, C)
    X_raw = X_raw.view(-1, 3).numpy()  # Shape: (N*H*W, C)
    X2 = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X_raw))
    x_min,x_max = X2[:,0].min()-1,X2[:,0].max()+1
    y_min,y_max = X2[:,1].min()-1,X2[:,1].max()+1
    grid_size = 75  # Reduced from 150 to lower memory usage
    xx,yy = np.meshgrid(np.linspace(x_min,x_max,grid_size), np.linspace(y_min,y_max,grid_size))
    grid = PCA(2).fit(X_raw).inverse_transform(np.c_[xx.ravel(),yy.ravel()])
    img_size = 224 if 'vit' in args.backbone else 32
    n_points = xx.shape[0] * xx.shape[1]  # grid_size * grid_size points
    grid = grid.reshape(n_points, 3)  # First reshape to (N, C)
    grid = np.tile(grid.reshape(1, n_points, 3), (img_size * img_size, 1, 1))  # Repeat for each pixel
    grid = grid.reshape(-1, img_size, img_size, 3)  # Reshape to (N, H, W, C)
    grid = np.transpose(grid, (0, 3, 1, 2))  # Convert to (N, C, H, W)
    grid_t = torch.tensor(grid, dtype=torch.float32, device=device)

    def sharpness(z):
        z.requires_grad_(True)
        batch_size = 32  # Process in smaller batches
        results = []
        for i in range(0, len(z), batch_size):
            batch = z[i:i+batch_size]
            s = model(batch)
            g = torch.autograd.grad(s.sum(), batch, create_graph=False)[0]
            results.append(g.contiguous().view(g.size(0),-1).norm(2,dim=1).cpu().numpy())
        return np.concatenate(results)

    frames=[]
    for epoch in trange(args.epochs, desc='Training epochs'):
        model.train()
        for x,_ in tqdm(loader, desc=f'Epoch {epoch+1}', leave=False):
            x = x.to(device)
            x_aug = kornia.geometry.rotate(x, (torch.rand(x.size(0), device=device)-.5)*30)
            loss = F.relu(1 + model(x_aug).mean() - model(x).mean())
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            # Process grid in batches for scoring
            batch_size = 32
            score_grid = []
            for i in range(0, len(grid_t), batch_size):
                batch = grid_t[i:i+batch_size]
                score_grid.append(model(batch).cpu().numpy())
            score_grid = np.concatenate(score_grid).reshape(grid_size,grid_size)
            
        sharp_grid = sharpness(grid_t).reshape(grid_size,grid_size)
        frames.append((score_grid, sharp_grid, np.median(score_grid)))

    # GIF
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    def animate(i):
        for a in ax: a.clear()
        s, sh, med = frames[i]
        ax[0].contourf(xx, yy, s, levels=50, cmap='coolwarm')
        ax[0].contour(xx, yy, s, levels=[med], colors='k')
        ax[0].scatter(X2[:,0], X2[:,1], c='k', s=8, alpha=.4)
        ax[0].set_title(f"{args.scoring} surface – epoch {i+1}")
        ax[1].imshow(sh, extent=[x_min,x_max,y_min,y_max], origin='lower', cmap='viridis')
        ax[1].set_title("Sharpness ‖∇S‖₂")
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=800)
    os.makedirs('figs', exist_ok=True)
    anim.save(f"figs/progressive_{args.backbone}_{args.scoring}.gif", writer=PillowWriter(fps=2))
    print(f"Saved figs/progressive_{args.backbone}_{args.scoring}.gif")

if __name__ == "__main__":
    main()