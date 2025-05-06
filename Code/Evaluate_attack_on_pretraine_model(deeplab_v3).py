
# modelA (DeepLabV3) đã load và modelA.eval()
# loader_test: DataLoader cho val_dataset (clean)
# apply_chromosome(img_np, best_chromo)
# compute_iou(pred_mask, true_mask)
modelA.load_state_dict(torch.load("deeplabv3_cifar10.pth", map_location=device, weights_only=True))
modelA.eval()
loader_test = DataLoader(val_dataset, batch_size=8, shuffle=False)


# helper: tính PSNR giữa two tensors [C,H,W] trong [0,1]
fgsm = torchattacks.FGSM(modelA, eps=0.07)
pgd = torchattacks.PGD(modelA, eps=0.07, alpha=0.01, steps=40)
deepfool = torchattacks.DeepFool(modelA)
cw = torchattacks.CW(modelA, c=1e-4, kappa=0, steps=1000, lr=1e-5)

def psnr_tensor(x, x_adv):
    x_np    = x.cpu().numpy().transpose(1,2,0)
    xadv_np = x_adv.cpu().numpy().transpose(1,2,0)
    return compute_psnr(x_np, xadv_np, data_range=1.0)


def segrmt_attack(x, best_chromo):
    x_np = x.cpu().numpy().transpose(0,2,3,1)*0.5 + 0.5
    out=[]
    for img in x_np:
        p = apply_chromosome(img, best_chromo)  # HxWxC
        out.append(torch.from_numpy(p.transpose(2,0,1)).float())
    return torch.stack(out).to(x.device)

def make_adv(atk, x, y, best_chromo=None):
    if atk == 'FGSM': return fgsm(x, y)
    if atk == 'PGD-10': return pgd(x, y)
    if atk == 'PGD-100': return pgd(x, y, steps=100)
    if atk == 'DeepFool': return deepfool(x, y)
    if atk == 'C&W': return cw(x, y)
    if atk == 'SegRMT': return segrmt_attack(x, best_chromo)
    raise ValueError(atk)

# --- 1) Tính IoU gốc (clean IoU) ---
clean_iou_list = []
for x, y in loader_test:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        out = modelA(x)['out']
        preds = out.argmax(1).cpu().numpy()
    for p, t in zip(preds, y.cpu().numpy()):
        clean_iou_list.append(compute_iou(p, t))
clean_iou = np.mean(clean_iou_list)
print(f"Clean IoU = {clean_iou:.4f}")

# --- 2) Đánh giá mỗi tấn công ---
attacks = ['FGSM', 'PGD-10', 'PGD-100', 'DeepFool', 'C&W', 'SegRMT']
res = {}
for atk in attacks:
    psnrs, adv_ious = [], []
    for x, y in loader_test:
        x, y = x.to(device), y.to(device)
        x_adv = make_adv(atk, x, y, best_chromo)
        # PSNR per-sample
        for xi, xai in zip(x, x_adv):
            psnrs.append(psnr_tensor(xi, xai))
        # IoU on adv batch
        with torch.no_grad():
            out = modelA(x_adv)['out']
            preds = out.argmax(1).cpu().numpy()
        for p, t in zip(preds, y.cpu().numpy()):
            adv_ious.append(compute_iou(p, t))
    res[atk] = {
        'PSNR': np.mean(psnrs),
        'AdvIoU': np.mean(adv_ious),
        'ΔIoU': clean_iou - np.mean(adv_ious)
    }

# --- 3) In kết quả và vẽ bar-chart ---
print(f"{'Attack':<10} {'PSNR':>6}  {'AdvIoU':>7}  {'ΔIoU':>6}")
for atk, v in res.items():
    print(f"{atk:<10} {v['PSNR']:6.2f}   {v['AdvIoU']:6.3f}   {v['ΔIoU']:6.3f}")

# Bar chart ΔIoU
plt.figure(figsize=(6, 4))
plt.bar(res.keys(), [res[a]['ΔIoU'] for a in attacks])
plt.ylabel('ΔIoU'); plt.title('Robustness (ΔIoU) Comparison'); plt.xticks(rotation=45)
plt.tight_layout(); plt.show()