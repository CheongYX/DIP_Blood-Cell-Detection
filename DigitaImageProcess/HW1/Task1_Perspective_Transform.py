import cv2
import numpy as np

clicked_pts = []

def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pts.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select 4 Corners", param)

def convert_to_grayscale_manual(bgr_img):
    b, g, r = bgr_img[:, :, 0], bgr_img[:, :, 1], bgr_img[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)

def get_perspective_params(dst_pts, src_pts):
    A, B = [], []
    for i in range(4):
        u, v = dst_pts[i]
        x, y = src_pts[i]
        A.append([u, v, 1, 0, 0, 0, -u*x, -v*x])
        A.append([0, 0, 0, u, v, 1, -u*y, -v*y])
        B.append(x)
        B.append(y)
    return np.linalg.solve(np.array(A), np.array(B))

def warp_perspective_manual(img, h, output_size):
    a1, a2, a3, a4, a5, a6, a7, a8 = h
    tw, th = output_size
    src_h, src_w = img.shape
    dst_img = np.zeros((th, tw), dtype=np.uint8)
    for y_prime in range(th):
        for x_prime in range(tw):
            denom = a7 * x_prime + a8 * y_prime + 1
            src_x = (a1 * x_prime + a2 * y_prime + a3) / denom
            src_y = (a4 * x_prime + a5 * y_prime + a6) / denom
            if 0 <= src_x < src_w - 1 and 0 <= src_y < src_h - 1:
                x1, y1 = int(np.floor(src_x)), int(np.floor(src_y))
                x2, y2 = x1 + 1, y1 + 1
                u, v = src_x - x1, src_y - y1
                val = (1-u)*(1-v)*img[y1, x1] + u*(1-v)*img[y1, x2] + (1-u)*v*img[y2, x1] + u*v*img[y2, x2]
                dst_img[y_prime, x_prime] = int(val)
    return dst_img

def main():
    img_color = cv2.imread('test.jpg')
    if img_color is None: return

    h, w = img_color.shape[:2]
    max_display_dim = 800  
    scale = max_display_dim / max(h, w)
    display_w, display_h = int(w * scale), int(h * scale)

    img_display = img_color.copy()
    window_name = "Select 4 Corners"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_w, display_h) 
    cv2.setMouseCallback(window_name, mouse_handler, img_display)
    
    while len(clicked_pts) < 4:
        cv2.imshow(window_name, img_display)
        if cv2.waitKey(1) & 0xFF == 27: break
    cv2.destroyAllWindows()

    if len(clicked_pts) == 4:
        gray_img = convert_to_grayscale_manual(img_color)
        side = 300
        dst_coords = np.array([(0, 0), (side-1, 0), (side-1, side-1), (0, side-1)], dtype="float32")
        h_params = get_perspective_params(dst_coords, np.array(clicked_pts, dtype="float32"))
        result = warp_perspective_manual(gray_img, h_params, (side, side))

        res_h = display_h
        res_w_original = display_w
        img_1 = cv2.resize(img_color, (res_w_original, res_h))
        
        img_2_gray = cv2.resize(gray_img, (res_w_original, res_h))
        img_2 = cv2.cvtColor(img_2_gray, cv2.COLOR_GRAY2BGR) 
        
        img_3_res = cv2.resize(result, (res_h, res_h)) 
        img_3 = cv2.cvtColor(img_3_res, cv2.COLOR_GRAY2BGR)
        
        comparison = np.hstack((img_1, img_2, img_3))
        
        cv2.namedWindow("DIP Project 1: Perspective Transformation", cv2.WINDOW_NORMAL)
        cv2.imshow("DIP Project 1: Perspective Transformation", comparison)
        print("处理完成！展示顺序：原图 -> 灰度图 -> 校正结果")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()