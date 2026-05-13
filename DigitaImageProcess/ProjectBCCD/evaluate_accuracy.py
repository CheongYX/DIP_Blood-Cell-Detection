import os
import xml.etree.ElementTree as ET
import numpy as np

GT_DIR = "BCCD_Dataset-master/BCCD/Annotations" 
PRED_DIR = "output/Annotations_pred"         

IOU_THRESHOLD = 0.5

def calculate_iou(boxA, boxB):
    # box 格式: [xmin, ymin, xmax, ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def parse_xml(xml_path):
    if not os.path.exists(xml_path):
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append({'name': name, 'bbox': [xmin, ymin, xmax, ymax], 'matched': False})
    return boxes

def evaluate_dataset():
    pred_files = [f for f in os.listdir(PRED_DIR) if f.endswith('.xml')]
    
    metrics = {
        'WBC': {'TP': 0, 'FP': 0, 'FN': 0},
        'RBC': {'TP': 0, 'FP': 0, 'FN': 0},
        'Platelets': {'TP': 0, 'FP': 0, 'FN': 0}
    }
    
    print(f"正在对比评估 {len(pred_files)} 张图片的检测结果...\n")

    for xml_file in pred_files:
        gt_path = os.path.join(GT_DIR, xml_file)
        pred_path = os.path.join(PRED_DIR, xml_file)
        
        gt_boxes = parse_xml(gt_path)
        pred_boxes = parse_xml(pred_path)
        
        for p_box in pred_boxes:
            p_cls = p_box['name']
            p_coords = p_box['bbox']
            
            best_iou = 0
            best_gt_idx = -1
            
            for i, g_box in enumerate(gt_boxes):
                if g_box['name'] == p_cls and not g_box['matched']:
                    iou = calculate_iou(p_coords, g_box['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
            
            if best_iou >= IOU_THRESHOLD and best_gt_idx != -1:
                metrics[p_cls]['TP'] += 1
                gt_boxes[best_gt_idx]['matched'] = True
            else:
                if p_cls in metrics:
                    metrics[p_cls]['FP'] += 1
                
        for g_box in gt_boxes:
            if not g_box['matched']:
                if g_box['name'] in metrics:
                    metrics[g_box['name']]['FN'] += 1

    print("="*60)
    print(f"{'Cell Type':<15} | {'Precision (精准率)':<20} | {'Recall (召回率)':<20} | {'F1-Score'}")
    print("-" * 60)
    
    for cls in ['WBC', 'RBC', 'Platelets']:
        tp = metrics[cls]['TP']
        fp = metrics[cls]['FP']
        fn = metrics[cls]['FN']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{cls:<15} | {precision*100:>18.2f}% | {recall*100:>18.2f}% | {f1*100:>8.2f}%")
        
    print("="*60)

if __name__ == "__main__":
    evaluate_dataset()