# CIFAR-10 Segmentation Dataset
class CIFAR10Segmentation(torchvision.datasets.CIFAR10):
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        mask = torch.full((img.size(1), img.size(2)), label, dtype=torch.long)
        return img, mask

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
dataset = CIFAR10Segmentation(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Model DeepLabV3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = xm.xla_device()
modelA = deeplabv3_resnet50(weights="DEFAULT").to(device).eval()
torch.save(modelA.state_dict(), "new_model_on_perturbed.pth")

class SegmentationModel:
    def __init__(self, modelA, device):
        self.model = modelA
        self.device = device

    def predict(self, img_np):
        img_tensor = torch.from_numpy(img_np.transpose(2,0,1)).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            pred = self.model(img_tensor)['out']
            pred_mask = pred.argmax(1).squeeze().cpu().numpy()
        return pred_mask

seg_model = SegmentationModel(modelA, device)