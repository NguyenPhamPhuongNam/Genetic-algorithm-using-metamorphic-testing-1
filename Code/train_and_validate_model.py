# 1) Chuẩn bị CIFAR10Segmentation và Normalize
base_dataset = CIFAR10Segmentation(root='./data', train=True, download=True, transform=transform)
val_dataset  = CIFAR10Segmentation(root='./data', train=False, download=False, transform=transform)
normalize_only = T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
best_chromo = best_chromosome  

# 2) Dataset perturbed cho cả train và val
class PerturbedCIFAR10(Dataset):
    def __init__(self, base_ds, chromo):
        self.base   = base_ds
        self.chromo = chromo

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]
        img_np = img.cpu().numpy().transpose(1,2,0)*0.5 + 0.5
        per_np = apply_chromosome(img_np, best_chromo)
        per_t  = torch.from_numpy(per_np.transpose(2,0,1)).float()
        per_t  = normalize_only(per_t)
        return per_t, mask

# 3) Khởi tạo loaders
train_dataset      = PerturbedCIFAR10(base_dataset, best_chromo)
train_loader       = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_clean_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False)
val_pert_loader    = DataLoader(PerturbedCIFAR10(val_dataset, best_chromo),
                                batch_size=8, shuffle=False)

# 4) Model, optimizer
device    = 'cuda' if torch.cuda.is_available() else 'cpu'
model     = fcn_resnet50(pretrained=False, num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)



# Composite loss (như bạn đã định nghĩa)
def composite_loss(model, x_clean, y_clean, x_adv, y_adv, alpha_1 =1.0, beta_1 =0.5):
    lc = F.cross_entropy(model(x_clean)['out'], y_clean)
    la = F.cross_entropy(model(x_adv)['out'],   y_adv)
    lr = F.mse_loss(model(x_clean)['out'], model(x_adv)['out'])
    return lc + alpha_1*la + beta_1*lr

# 5) Training + 2‑way Validation
epochs = 30
train_losses = []
val_losses_clean, val_losses_pert = [], []
val_accs_clean, val_accs_pert     = [], []

for ep in range(1, epochs+1):
    # --- train ---
    model.train()
    running = 0.0
    for x_adv, masks in train_loader:
        x_adv, masks = x_adv.to(device), masks.to(device)
        # here x_clean == x_adv, vì chúng ta chỉ train trên perturbed
        optimizer.zero_grad()
        loss = composite_loss(model, x_adv, masks, x_adv, masks, alpha_1=1.0, beta_1=0.5)
        loss.backward()
        optimizer.step()
        running += loss.item()
        
    train_loss = running / len(train_loader)
    train_losses.append(train_loss)

    # --- val clean & perturbed ---
    model.eval()
    rc, rp = 0.0, 0.0      # losses
    cc, cp = 0, 0          # correct
    tc, tp = 0, 0          # total

    with torch.no_grad():
        for (imgs_c, masks_c), (imgs_p, masks_p) in zip(val_clean_loader, val_pert_loader):
            # clean
            imgs_c, masks_c = imgs_c.to(device), masks_c.to(device)
            out_c = model(imgs_c)['out']
            rc   += F.cross_entropy(out_c, masks_c).item()
            preds_c = out_c.argmax(1)
            cc   += (preds_c==masks_c).sum().item()
            tc   += masks_c.numel()

            # perturbed
            imgs_p, masks_p = imgs_p.to(device), masks_p.to(device)
            out_p = model(imgs_p)['out']
            rp   += F.cross_entropy(out_p, masks_p).item()
            preds_p = out_p.argmax(1)
            cp    += (preds_p==masks_p).sum().item()
            tp    += masks_p.numel()

    val_clean_loss = rc / len(val_clean_loader)
    val_pert_loss  = rp / len(val_pert_loader)
    val_clean_acc  = cc / tc
    val_pert_acc   = cp / tp


    val_losses_clean.append(val_clean_loss)
    val_losses_pert.append(val_pert_loss)
    val_accs_clean.append(val_clean_acc)
    val_accs_pert.append(val_pert_acc)

    print(f"Epoch {ep}/{epochs}"
          f" | Train: {train_loss:.4f}"
          f" | Val Clean: {val_clean_loss:.4f}, Acc: {val_clean_acc:.4f}"
          f" | Val Pert: {val_pert_loss:.4f}, Acc: {val_pert_acc:.4f}")


  # --- Scheduler step based on perturbed val loss ---
    scheduler.step(val_pert_loss)


# 6) Vẽ đồ thị
plt.figure()
plt.plot(train_losses,   label="Train Loss")
plt.plot(val_losses_clean, label="Val Clean Loss")
plt.plot(val_losses_pert,  label="Val Perturbed Loss")
plt.legend(); plt.title("Loss Curves"); plt.show()

plt.figure()
plt.plot(val_accs_clean, label="Val Clean Acc")
plt.plot(val_accs_pert,  label="Val Perturbed Acc")
plt.legend(); plt.title("Accuracy"); plt.show() 

# 7) Lưu model
torch.save(model.state_dict(), "new_model_on_perturbed.pth")
torch.save(modelA.state_dict(), "deeplabv3_cifar10.pth")