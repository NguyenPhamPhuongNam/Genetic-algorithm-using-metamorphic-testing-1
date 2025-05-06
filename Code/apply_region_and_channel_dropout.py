def apply_distortions(image, spatial_rate, channel_dropout):
    img_copy = image.copy()
    h, w, c = img_copy.shape

    # Region Dropout
    num_pixels = int(spatial_rate * h * w)
    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)
    img_copy[ys, xs, :] = np.min(img_copy)

    # Channel Dropout
    for idx, dropout in enumerate(channel_dropout):
        if dropout:
            img_copy[:, :, idx] = np.min(img_copy)

    return img_copy
