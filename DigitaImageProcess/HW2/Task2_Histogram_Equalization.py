import cv2
import numpy as np

def manual_equalization(gray_img):
    h, w = gray_img.shape
    total_pixels = h * w
    
    hist = np.zeros(256, dtype=np.int32)
    for i in range(h):
        for j in range(w):
            hist[gray_img[i, j]] += 1
            
    cdf = np.zeros(256)
    current_sum = 0
    for i in range(256):
        current_sum += hist[i] / total_pixels
        cdf[i] = current_sum
        
    lut = np.round(cdf * 255).astype(np.uint8)
    
    equalized_img = lut[gray_img] 
    return equalized_img, hist

def draw_hist_panel(hist, width=256, height=100):
    hist_img = np.full((height, width, 3), 240, dtype=np.uint8)
    max_val = np.max(hist)
    for i in range(width):
        bin_h = int((hist[i] / max_val) * (height - 10))
        cv2.line(hist_img, (i, height), (i, height - bin_h), (80, 80, 80), 1)
    return hist_img

def main():
    img_color = cv2.imread('test3.jpg') 
    if img_color is None: return
    
    b, g, r = img_color[:,:,0], img_color[:,:,1], img_color[:,:,2]
    gray = (0.299*r + 0.587*g + 0.114*b).astype(np.uint8)
    
    equalized, old_hist = manual_equalization(gray)
    
    new_hist = np.zeros(256, dtype=np.int32)
    for val in range(256):
        new_hist[val] = np.sum(equalized == val)

    display_h = 450 
    
    col1 = cv2.resize(img_color, (int(img_color.shape[1] * display_h / img_color.shape[0]), display_h))
    
    gray_resized = cv2.resize(gray, (col1.shape[1], display_h - 100))
    gray_bgr = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
    hist_old_viz = cv2.resize(draw_hist_panel(old_hist), (col1.shape[1], 100))
    col2 = np.vstack((gray_bgr, hist_old_viz))
    
    res_resized = cv2.resize(equalized, (col1.shape[1], display_h - 100))
    res_bgr = cv2.cvtColor(res_resized, cv2.COLOR_GRAY2BGR)
    hist_new_viz = cv2.resize(draw_hist_panel(new_hist), (col1.shape[1], 100))
    col3 = np.vstack((res_bgr, hist_new_viz))
    comparison = np.hstack((col1, col2, col3))
    
    win_name = "DIP Project 2: Histogram Equalization Contrast"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow(win_name, comparison)
    
    print("展示完成：[彩色原图] | [灰度分布] | [均衡化增强]")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()