import torch.nn as nn
import sys
import cv2
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from torchvision import transforms
from units import Conv2d, Stem, Inception_ResNet_A, Reduction_A, Inception_ResNet_B, Reduction_B, Inception_ResNet_C

# 定义 Inception_ResNetv2 模型
class Inception_ResNetv2(nn.Module):
    def __init__(self, in_channels=3, classes=1000, k=256, l=256, m=384, n=384):
        super(Inception_ResNetv2, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for i in range(10):
            blocks.append(Inception_ResNet_A(320, 0.17))
        blocks.append(Reduction_A(320, k, l, m, n))
        for i in range(20):
            blocks.append(Inception_ResNet_B(1088, 0.10))
        blocks.append(Reduction_B(1088))
        for i in range(9):
            blocks.append(Inception_ResNet_C(2080, 0.20))
        blocks.append(Inception_ResNet_C(2080, activation=False))
        self.features = nn.Sequential(*blocks)
        self.conv = Conv2d(2080, 1536, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

# 数据预处理
IMAGE_SIZE = 299
TEST_MEAN = [0.4640921, 0.50851228, 0.49258145]
TEST_STD = [0.18034702, 0.16389939, 0.16225852]
test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(TEST_MEAN, TEST_STD)
])

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'inception_resnetv2_cub.pth'
model = Inception_ResNetv2(classes=1010).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

def load_class_names(file_path):
    class_names = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                class_id, class_name = parts
                class_names[int(class_id)] = class_name
    return class_names

class_names_file = 'classes.txt'
class_names = load_class_names(class_names_file)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = '鸟类鉴别'
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.label = QLabel('请选择一张图像进行预测', self)
        layout.addWidget(self.label)

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.button = QPushButton('导入图像', self)
        self.button.clicked.connect(self.open_image)
        layout.addWidget(self.button)

        self.setLayout(layout)

    def open_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "Images (*.png *.xpm *.jpg);;All Files (*)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(299, 299))
            self.predict_image(file_name)

    def predict_image(self, image_path):
        image = cv2.imread(image_path)
        image = test_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            predicted_class = class_names[preds.item() + 1]
            self.label.setText(f'预测类别: {predicted_class}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
