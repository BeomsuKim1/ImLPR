import torch
REPO_DIR = "models/dinov3"
IMAGE_DIR = "data/image.jpg"


def main():
    dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights="models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights="models/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")

if __name__ == "__main__":
    main()