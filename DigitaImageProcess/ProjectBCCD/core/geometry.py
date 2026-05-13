import numpy as np
import cv2

def get_perspective_matrix(src, dst):
    """手动求解 8 参数透视变换矩阵"""
    A = []
    for i in range(4):
        u, v = src[i]
        x, y = dst[i]
        A.append([u, v, 1, 0, 0, 0, -u*x, -v*x])
        A.append([0, 0, 0, u, v, 1, -u*y, -v*y])
    A = np.array(A)
    B = np.array(dst).reshape(8)
    # 求解 A * h = B -> h = [a1, a2, a3, a4, a5, a6, a7, a8]
    h = np.linalg.solve(A, B)
    return np.append(h, 1.0).reshape((3, 3))

def bilinear_interpolate(img, x, y):
    """手动实现双线性插值采样"""
    h, w = img.shape[:2]
    x1, y1 = int(np.floor(x)), int(np.floor(y))
    x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
    
    dx, dy = x - x1, y - y1
    
    # 采样四个邻近像素
    val = (1-dx)*(1-dy)*img[y1, x1] + dx*(1-dy)*img[y1, x2] + \
          (1-dx)*dy*img[y2, x1] + dx*dy*img[y2, x2]
    return val