# 1) Lấy nhiều samples từ loader
def get_samples_from_loader(loader, n_samples):
    samples = []
    for _ in range(n_samples):
        sample_img, sample_mask = next(iter(loader))
        samples.append((sample_img, sample_mask))
    return samples

# Lấy 5 samples từ loader
samples = get_samples_from_loader(loader, 5)

# 2) Khởi tạo GA, truyền seg_model và compute_iou
ga = GeneticAlgorithm(
    img_shape=(32,32,3),
    seg_model=seg_model,
    compute_iou=compute_iou,      
    pop_size=10,
    generations=9,
    crossover_method="two_point",
    alpha = 0.5,
    beta = 2.0
)

# 3) Lặp qua các samples và chạy GA cho mỗi sample
for idx, (sample_img, sample_mask) in enumerate(samples):
    best = ga.run(sample_img.numpy().transpose(1, 2, 0) * 0.5 + 0.5,
                  sample_mask.numpy())
    best_chromosome = best

    # In chi tiết best_chromosome
    print(f"\nBest Chromosome found for sample {idx+1}:")
    print("Best length:", len(best.transformations))
    for i, sub in enumerate(best_chromosome.transformations, 1):
        print(f"\n Sub-vector #{i}:")
        print(f"   • Dropout:       active={sub['activate_dropout']}, rate={sub['dropout_rate']:.3f}")
        print(f"   • Gaussian:      active={sub['activate_gaussian']}, σ={sub['gaussian_sigma']:.2f}")
        print(f"   • Brightness:    active={sub['activate_brightness']}, shift={sub['brightness_shift']:.3f}")
        print(f"   • Channel Shift: active={sub['activate_channel_shift']}, shifts={sub['channel_shift_values']}")
        total_pixels = sub['indices_mask'].size
        affected     = int(np.sum(sub['indices_mask']))
        print(f"   • Pixels affected mask: {affected} / {total_pixels}")

    # 4) Hàm hiển thị ảnh gốc và ảnh perturbed cho mỗi sample
    def generate_and_show_image(img_tensor, chromosome, idx):
        img_np = img_tensor.numpy().transpose(1, 2, 0) * 0.5 + 0.5
        perturbed_img = apply_chromosome(img_np, chromosome)

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img_np)
        axs[0].set_title(f"Original Image {idx+1}")
        axs[0].axis('off')

        axs[1].imshow(perturbed_img)
        axs[1].set_title(f"Perturbed Image {idx+1}")
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    # 5) Hiển thị ảnh gốc và perturbed cho mỗi sample
    generate_and_show_image(sample_img, best_chromosome, idx)
