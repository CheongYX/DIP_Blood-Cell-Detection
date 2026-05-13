import os
import xml.etree.ElementTree as ET
import numpy as np

GT_DIR = "BCCD_Dataset-master/BCCD/Annotations"

def analyze_dataset():
    stats = {
        'RBC': {'areas': [], 'widths': [], 'heights': [], 'aspect_ratios': []},
        'Platelets': {'areas': [], 'widths': [], 'heights': [], 'aspect_ratios': []}
    }

    if not os.path.exists(GT_DIR):
        print(f"❌ 找不到路径: {GT_DIR}")
        return

    print("🔍 正在全量扫描 XML 标注文件，双通道提取中...")

    for xml_file in os.listdir(GT_DIR):
        if not xml_file.endswith('.xml'): continue
        tree = ET.parse(os.path.join(GT_DIR, xml_file))

        for obj in tree.getroot().findall('object'):
            name = obj.find('name').text
            
            if name in stats:
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                w, h = xmax - xmin, ymax - ymin
                area = w * h

                stats[name]['areas'].append(area)
                stats[name]['widths'].append(w)
                stats[name]['heights'].append(h)
                
                stats[name]['aspect_ratios'].append(max(w, h) / (min(w, h) + 1e-5))

    for cell_type in ['Platelets', 'RBC']: 
        areas = stats[cell_type]['areas']
        if not areas:
            continue

        widths = stats[cell_type]['widths']
        heights = stats[cell_type]['heights']
        aspect_ratios = stats[cell_type]['aspect_ratios']
        mean_area = np.mean(areas)

        print("\n" + "=" * 50)
        print(f"🦠 {cell_type} 全局数据统计 (共 {len(areas)} 个)")
        print("=" * 50)
        print(f"【面积 (Area)】")
        print(f" 绝对极限: {np.min(areas):.0f} px (最小) ~ {np.max(areas):.0f} px (最大)")
        print(f" 核心分布: {np.percentile(areas, 5):.0f} px (5%) ~ {np.percentile(areas, 95):.0f} px (95%)")
        print(f" 典型均值: {mean_area:.0f} px | 中位数: {np.median(areas):.0f} px")
        
        print(f"【形态 (Shape)】")
        print(f" 平均尺寸: {np.mean(widths):.0f} x {np.mean(heights):.0f} px")
        print(f" 长 宽 比: {np.mean(aspect_ratios):.2f} (越接近1越方/圆)")

        if cell_type == 'RBC':
            print("-" * 50)
            print(f"RBC 霍夫圆 (HoughCircles) 物理反推建议：")
            min_r = np.sqrt(np.percentile(areas, 5) / np.pi)
            max_r = np.sqrt(np.percentile(areas, 95) / np.pi)
            mean_r = np.sqrt(mean_area / np.pi)
            
            print(f" -> 建议 minRadius = {int(min_r)}")
            print(f" -> 建议 maxRadius = {int(max_r)}")
            print(f" -> 典型中心半径  = {int(mean_r)}")

if __name__ == "__main__":
    analyze_dataset()