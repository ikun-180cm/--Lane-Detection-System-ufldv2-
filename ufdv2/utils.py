import cv2
import torch
import numpy as np


def preprocess(img, img_size=640):
    h0, w0 = img.shape[:2]

    img = cv2.resize(img, (img_size, img_size))
    img = img[:, :, ::-1]  # BGR → RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    return torch.from_numpy(img), (h0, w0)


def postprocess_lane(mask, original_shape):
    h, w = original_shape

    mask = mask.squeeze()
    mask = mask.cpu().numpy()

    mask = cv2.resize(mask, (w, h))
    mask = (mask > 0.5).astype(np.uint8) * 255

    return mask
