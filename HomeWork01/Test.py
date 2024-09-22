from ImageSegmentationDataset import *
from UNet import *
from Train import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score
from skimage.transform import resize
from PIL import Image
import torch

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型评估函数
def predict(model, image_path):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output).cpu().squeeze().numpy()
    return output

# 计算评估指标函数
def compute_metrics(pred, gt):
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)
    iou = jaccard_score(gt.flatten(), pred.flatten(), average='binary')
    dice = f1_score(gt.flatten(), pred.flatten(), average='binary')
    return iou, dice

# 加载并处理地面真实图像
def load_gt_image(gt_image_path, pred_shape):
    gt_image = Image.open(gt_image_path).convert('L')
    gt = np.array(gt_image)
    gt = resize(gt, pred_shape, mode='constant', preserve_range=True)
    return gt

# 显示结果函数
def display_results(original_image_path, prediction, gt):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(Image.open(original_image_path))

    plt.subplot(1, 3, 2)
    plt.title('Prediction')
    plt.imshow(prediction, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Ground Truth')
    plt.imshow(gt, cmap='gray')

    plt.show()

# 主函数
def main():
    # 图像路径
    test_image_path = r'archive/images/2C29D473-CCB4-458C-926B-99D0042161E6.jpg'
    gt_image_path = r'archive/labels/2C29D473-CCB4-458C-926B-99D0042161E6.jpg'

    # 加载模型并进行预测
    prediction = predict(model, test_image_path)

    # 加载并处理地面真实图像
    gt = load_gt_image(gt_image_path, prediction.shape)

    # 计算并打印评估指标
    iou, dice = compute_metrics(prediction, gt)
    print(f"IOU: {iou:.4f}, Dice: {dice:.4f}")

    # 显示结果
    display_results(test_image_path, prediction, gt)

if __name__ == "__main__":
    main()