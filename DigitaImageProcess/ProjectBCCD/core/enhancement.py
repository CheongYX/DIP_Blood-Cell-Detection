import cv2
import numpy as np

def manual_pdf_cdf(channel):
    """手动计算 CDF 用于直方图重映射"""
    hist, _ = np.histogram(channel.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max() # 归一化用于可视化
    
    # 建立映射表 (Lookup Table)
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    lut = np.ma.filled(cdf_m, 0).astype('uint8')
    return lut[channel]

def blood_clahe_enhancement(img):
    """针对血液图像优化的增强"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 在亮度通道 L 上应用自适应均衡化，保护边缘
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)