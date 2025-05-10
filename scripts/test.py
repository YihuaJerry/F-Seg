import os
import sys
import argparse
import torch
from torchvision import transforms
from PIL import Image
from core.utils.visualize import get_color_pallete
from core.models import get_model  # 获取模型的自定义函数

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

# 设置命令行参数
parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fcn32s_vgg16_voc',
                    help='model name (default: fcn32_vgg16)')
parser.add_argument('--dataset', type=str, default='pascal_aug', choices=['pascal_voc', 'pascal_aug', 'ade20k', 'citys'],
                    help='dataset name (default: pascal_voc)')
parser.add_argument('--save-folder', default='~/.torch/models',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str, default='../datasets/voc/VOC2012/JPEGImages/2007_000032.jpg',
                    help='path to the input picture')
parser.add_argument('--outdir', default='./eval', type=str,
                    help='path to save the predict result')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

# 定义模型加载的函数
def demo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建输出目录
    if not os.path.exists(config.outdir):
        os.makedirs(config.outdir)

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # 加载测试图片
    image = Image.open(config.input_pic).convert('RGB')
    images = transform(image).unsqueeze(0).to(device)

    # 加载模型并将权重加载到模型中
    model = get_model(args.model, local_rank=args.local_rank, pretrained=False, root=args.save_folder).to(device)
    print('Finished loading model!')

    # 加载模型的状态字典（state_dict）
    model.load_state_dict(torch.load("fdlnet_deeplab.pth"))  # 使用您的模型权重文件路径

    # 设置为评估模式
    model.eval()
    
    # 执行推理
    with torch.no_grad():
        output = model(images)

    # 处理输出
    pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
    
    # 使用颜色调色板进行可视化
    mask = get_color_pallete(pred, args.dataset)
    outname = os.path.splitext(os.path.split(args.input_pic)[-1])[0] + '.png'
    mask.save(os.path.join(args.outdir, outname))

if __name__ == '__main__':
    demo(args)
