import cv2
import numpy as np

def segment_masterpiece(img, k=4, spatial_weight=0.3):
    h, w = img.shape[:2]
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    data = []
    for y in range(h):
        for x in range(w):
            l, a, b = lab[y, x]
            data.append([
                l / 100.0, 
                a / 128.0, 
                b / 128.0, 
                (x / w) * spatial_weight, 
                (y / h) * spatial_weight
            ])
    data = np.array(data, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    colors = np.array([
        [255, 170, 0], 
        [0, 255, 255], 
        [255, 0, 0],    
        [0, 255, 0],  
        [128, 0, 128]    
    ], dtype=np.uint8)
    
    if k > len(colors):
        colors = np.vstack((colors, np.random.randint(0, 255, (k-len(colors), 3), dtype=np.uint8)))

    res_data = colors[labels.flatten()]
    return res_data.reshape((h, w, 3))

def main():
    src = cv2.imread('test.jpg')
    if src is None: return

    lab_temp = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_temp)
    l_eq = cv2.equalizeHist(l)
    eq_bgr = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

    segmented = segment_masterpiece(src, k=4, spatial_weight=0.4)

    display_h = 450
    def fix_size(image, target_h):
        r = target_h / image.shape[0]
        return cv2.resize(image, (int(image.shape[1] * r), target_h))

    img_1 = fix_size(src, display_h)
    img_2 = fix_size(eq_bgr, display_h)
    img_3 = fix_size(segmented, display_h)

    comparison = np.hstack((img_1, img_2, img_3))

    win_name = " Original vs Equalized vs Segmented"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(win_name, comparison)
    
    print("Pipeline 执行成功！")
    print("原图展示了原始视角|均衡化展示了增强后的动态范围|分割图展示")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()