import cv2
import numpy as np

def run_kmeans_seg(img, k=3):
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    x_norm = (x_coords / w) * 0.2  # 0.2 是空间权重
    y_norm = (y_coords / h) * 0.2
    
    features = np.stack([
        lab[:,:,0]/100.0, 
        (lab[:,:,1]+128)/255.0, 
        (lab[:,:,2]+128)/255.0, 
        x_norm, 
        y_norm
    ], axis=-1).reshape(-1, 5).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(features, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    centers[:, 0] *= 100.0
    
    return labels.reshape((h, w)), centers