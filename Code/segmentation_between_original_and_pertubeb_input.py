# --- 1) Setup thiết bị và model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 10
model.load_state_dict(torch.load("new_model_on_perturbed.pth", map_location=device))
model.eval()


# --- 2) Transform giống khi train (chỉ normalize, ToTensor sẽ tự convert numpy→tensor) ---
normalize = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# --- 3) Lấy sample và tạo perturbed ngay trong RAM ---
sample_img, _ = next(iter(loader))            # loader là DataLoader CIFAR-10
orig_np       = sample_img[0].cpu().numpy().transpose(1,2,0)*0.5 + 0.5
perturbed_np  = apply_chromosome(orig_np, best_chromo)

# --- 4) Tiền xử lý để inference ---
# chuyển numpy [H,W,3] về tensor [1,3,H,W]
original_tensor = normalize(orig_np).unsqueeze(0).to(device)
perturbed_tensor = normalize(perturbed_np).unsqueeze(0).to(device)

# --- 5) Chạy inference ---
with torch.no_grad():
    # Inference cho ảnh gốc
    out_orig = model(original_tensor)['out']  # [1,10,H,W]
    pred_mask_orig = out_orig.argmax(1).squeeze(0).cpu().numpy()

    # Inference cho ảnh perturbed
    out_pert = model(perturbed_tensor)['out']  # [1,10,H,W]
    pred_mask_pert = out_pert.argmax(1).squeeze(0).cpu().numpy()

# --- 6) Hiển thị ---
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 5))

# Ảnh gốc
ax1.imshow(orig_np)
ax1.set_title("Original Input")
ax1.axis('off')

# Segmentation của ảnh gốc
ax2.imshow(pred_mask_orig, cmap='tab10')
ax2.set_title("Segmentation of Original Input")
ax2.axis('off')

# Ảnh perturbed
ax3.imshow(perturbed_np)
ax3.set_title("Perturbed Input")
ax3.axis('off')

# Segmentation của ảnh perturbed
ax4.imshow(pred_mask_pert, cmap='tab10')
ax4.set_title("Segmentation of Perturbed Input")
ax4.axis('off')

plt.tight_layout()
plt.show()