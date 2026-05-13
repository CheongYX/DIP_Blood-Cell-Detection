import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

IMG_DIR = "BCCD_Dataset-master/BCCD/JPEGImages"
OUTPUT_XML_DIR = "output/Annotations_pred"
os.makedirs(OUTPUT_XML_DIR, exist_ok=True)

def feature_driven_pipeline(image_filename):
    img_path = os.path.join(IMG_DIR, image_filename)
    img = cv2.imread(img_path)
    wbc_exclusion_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    if img is None:
        return None

    result_img = img.copy()
    detection_data = []

    wbc_exclusion_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_purple = np.array([100, 40, 40])
    upper_purple = np.array([170, 255, 255])
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel_small)

    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    purple_mask_closed = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel_large)

    contours, _ = cv2.findContours(purple_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area > 8000:
            x, y, w, h = cv2.boundingRect(cnt)
            pad = 15
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(img.shape[1], x+w+pad), min(img.shape[0], y+h+pad)
            
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(result_img, "WBC", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            detection_data.append({"label": "WBC", "bbox": [x1, y1, x2-x1, y2-y1]})
            
            cv2.rectangle(wbc_exclusion_mask, (x, y), (x+w, y+h), 255, -1)

    kernel_30 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    wbc_halo_mask = cv2.dilate(wbc_exclusion_mask, kernel_30)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
        
    #     if 400 < area < 3500:
            
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         ratio = w / (float(h) + 1e-5)
    #         if not (0.6 < ratio < 2.5): 
    #             continue
                
    #         hull = cv2.convexHull(cnt)
    #         hull_area = cv2.contourArea(hull)
    #         if hull_area == 0: 
    #             continue
    #         solidity = area / hull_area
    #         if solidity <= 0.6: 
    #             continue
                
    #         M = cv2.moments(cnt)
    #         if M["m00"] == 0: continue
    #         cx = int(M["m10"] / M["m00"])
    #         cy = int(M["m01"] / M["m00"])
    #         if cy >= img.shape[0] or cx >= img.shape[1]: continue
    #         if wbc_halo_mask[cy, cx] == 255: 
    #             continue
                
    #         peri = cv2.arcLength(cnt, True)
    #         circularity = 4 * np.pi * area / (peri * peri + 1e-5)
    #         if not (0.45 < circularity < 1.4): 
    #             continue
                
    #         mask_single = np.zeros(img.shape[:2], dtype=np.uint8)
    #         cv2.drawContours(mask_single, [cnt], -1, 255, -1)
            
    #         mean_h = cv2.mean(hsv[:,:,0], mask=mask_single)[0]
    #         mean_s = cv2.mean(hsv[:,:,1], mask=mask_single)[0]
    #         mean_gray = cv2.mean(gray, mask=mask_single)[0]
            
    #         if (110 < mean_h < 160) and (mean_s > 40) and (mean_gray < 170):
                
    #             cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
    #             cv2.putText(result_img, "Platelets", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    #             detection_data.append({"label": "Platelets", "bbox": [x, y, w, h]})

    params = cv2.SimpleBlobDetector_Params()
    
    params.filterByColor = True
    params.blobColor = 0
    
    params.filterByArea = True
    params.minArea = 400
    params.maxArea = 3500
    
    params.filterByCircularity = True
    params.minCircularity = 0.45
    
    params.filterByConvexity = True
    params.minConvexity = 0.5
    
    params.filterByInertia = True
    params.minInertiaRatio = 0.2

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)

    for kp in keypoints:
        cx, cy = int(kp.pt[0]), int(kp.pt[1])
        diameter = kp.size
        radius = int(diameter / 2)
        
        if cy >= img.shape[0] or cx >= img.shape[1]: 
            continue
            
        if wbc_halo_mask[cy, cx] == 255: 
            continue
        mask_single = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask_single, (cx, cy), radius, 255, -1)
        
        mean_h = cv2.mean(hsv[:,:,0], mask=mask_single)[0]
        mean_gray = cv2.mean(gray, mask=mask_single)[0]
        
        if (110 < mean_h < 160) and (mean_gray < 170):
            
            x = max(0, cx - radius)
            y = max(0, cy - radius)
            w = int(diameter)
            h = int(diameter)
            
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(result_img, "Platelets", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            detection_data.append({"label": "Platelets", "bbox": [x, y, w, h]})
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    blurred = cv2.medianBlur(enhanced_gray, 5)

    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=50, 
        param1=50, 
        param2=40, 
        minRadius=20, 
        maxRadius=70
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center_x, center_y, radius = i[0], i[1], i[2]
            
            if center_y < img.shape[0] and center_x < img.shape[1]:
                if wbc_exclusion_mask[center_y, center_x] == 255:
                    continue
                
            x = max(0, center_x - radius)
            y = max(0, center_y - radius)
            w = radius * 2
            h = radius * 2
            
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(result_img, "RBC", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            detection_data.append({"label": "RBC", "bbox": [x, y, w, h]})

    return result_img, purple_mask, detection_data

def save_to_pascal_voc(filename, img_shape, detections, output_dir):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "JPEGImages"
    ET.SubElement(root, "filename").text = filename
    
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(img_shape[1])
    ET.SubElement(size, "height").text = str(img_shape[0])
    ET.SubElement(size, "depth").text = str(img_shape[2])
    ET.SubElement(root, "segmented").text = "0"
    
    for det in detections:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = det['label']
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        
        bndbox = ET.SubElement(obj, "bndbox")
        x, y, w, h = det['bbox']
        ET.SubElement(bndbox, "xmin").text = str(x)
        ET.SubElement(bndbox, "ymin").text = str(y)
        ET.SubElement(bndbox, "xmax").text = str(x + w)
        ET.SubElement(bndbox, "ymax").text = str(y + h)
        
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    xml_filename = filename.rsplit('.', 1)[0] + '.xml'
    xml_path = os.path.join(output_dir, xml_filename)
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

if __name__ == "__main__":
    IMG_DIR = "BCCD_Dataset-master/BCCD/JPEGImages" 
    
    image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg'))]
    total_imgs = len(image_files)
    
    print(f"开始批量处理，共 {total_imgs} 张图片，正在生成 XML...")
    
    for i, img_file in enumerate(image_files):
        print(f"[{i+1}/{total_imgs}] 正在处理: {img_file} ...", end="", flush=True)
        img_path = os.path.join(IMG_DIR, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print("读取失败！")
            continue
        res = feature_driven_pipeline(img_file)
        
        if res:
            final_img, p_mask, data = res 
            
            save_to_pascal_voc(img_file, img.shape, data, OUTPUT_XML_DIR)
            
            print(f"成功 (框出 {len(data)} 个目标)")
        else:
            print("异常。")
            
    print("批量处理和 XML 导出全部完成！")
