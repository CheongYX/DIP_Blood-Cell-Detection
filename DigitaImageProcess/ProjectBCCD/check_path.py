import os

# 这里的路径要和你 main_detector.py 里的一模一样
BASE_DIR = "BCCD-dataset-master/BCCD"
IMG_DIR = os.path.join(BASE_DIR, "JPEGImages")
test_file = "BloodImage_00000.jpg"
full_path = os.path.join(IMG_DIR, test_file)

print(f"当前工作目录: {os.getcwd()}")
print(f"预期查找路径: {full_path}")
print(f"文件夹是否存在: {os.path.exists(IMG_DIR)}")
print(f"文件是否存在: {os.path.exists(full_path)}")

# 顺便看看当前目录下都有啥
print(f"当前目录下的内容: {os.listdir('.')}")