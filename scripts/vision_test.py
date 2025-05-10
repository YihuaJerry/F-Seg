#origin_vision

# import os
# import sys
# import argparse
# import torch
# from glob import glob

# cur_path = os.path.abspath(os.path.dirname(__file__))
# root_path = os.path.split(cur_path)[0]
# sys.path.append(root_path)

# from torchvision import transforms
# from PIL import Image
# from core.utils.visualize import get_color_pallete
# from core.models import get_segmentation_model

# parser = argparse.ArgumentParser(
#     description='Predict segmentation results from images')
# parser.add_argument('--model', type=str, default='yoloworld',
#                     help='model name (default: yoloworld)')
# parser.add_argument('--dataset', type=str, default='night',
#                     help='dataset name (default: night)')
# parser.add_argument('--save-folder', default='~/.torch/models',
#                     help='Directory for saving checkpoint models')
# parser.add_argument('--input-dir', type=str, 
#                     default='/home/ma-user/work/ymxwork/NIPS/YOLO-World/datasets/NightCity/images/val/',
#                     help='path to input directory')
# parser.add_argument('--outdir', type=str,
#                     default='/home/ma-user/work/ymxwork/NIPS/YOLO-World/FDLNet/image_vision_test',
#                     help='path to save predict results')
# parser.add_argument('--local_rank', type=int, default=0)
# args = parser.parse_args()

# def process_image(model, transform, device, img_path, output_dir):
#     # Load and process image
#     image = Image.open(img_path).convert('RGB')
#     images = transform(image).unsqueeze(0).to(device)
    
#     # Inference
#     with torch.no_grad():
#         output = model(images)
    
#     # Process output
#     output = output.permute(0, 2, 1, 3)
#     pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
#     mask = get_color_pallete(pred, args.dataset)
    
#     # Save result
#     outname = os.path.basename(img_path).split('.')[0] + '_mask.png'
#     mask.save(os.path.join(output_dir, outname))
#     print(f'Processed: {os.path.basename(img_path)} -> {outname}')

# def demo(config):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Create output directory
#     if not os.path.exists(config.outdir):
#         os.makedirs(config.outdir)
    
#     # Get first 20 images
#     image_paths = sorted(glob(os.path.join(config.input_dir, '*.png')))[:100]
#     if not image_paths:
#         print(f"No images found in {config.input_dir}")
#         return
    
#     # Prepare transforms
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ])
    
#     # Load model
#     model = get_segmentation_model(args.model).to(device)
#     checkpoint = torch.load(
#         '/home/ma-user/work/ymxwork/NIPS/train/runs/ckpt/last_yoloworld_resnet50_night_epoch_780_mean_iu_0.52015.pth',
#         map_location=device
#     )
#     model.load_state_dict(checkpoint['state_dict'])
#     model.eval()
#     print('Model loaded successfully')
    
#     # Process images
#     for img_path in image_paths:
#         process_image(model, transform, device, img_path, config.outdir)

# if __name__ == '__main__':
#     demo(args)




import os
import sys
import argparse
import torch
from glob import glob
import matplotlib.pyplot as plt


cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from core.utils.visualize import get_color_pallete
from core.models import get_segmentation_model

parser = argparse.ArgumentParser(
    description='Predict segmentation results from images')
parser.add_argument('--model', type=str, default='yoloworld',
                    help='model name (default: yoloworld)')
parser.add_argument('--dataset', type=str, default='night',
                    help='dataset name (default: night)')
parser.add_argument('--save-folder', default='~/.torch/models',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-dir', type=str, 
                    default='/home/ma-user/work/ymxwork/NIPS/YOLO-World/datasets/NightCity/images/val/',
                    help='path to input directory')
parser.add_argument('--outdir', type=str,
                    default='/home/ma-user/work/ymxwork/NIPS/YOLO-World/FDLNet/image_vision_test',
                    help='path to save predict results')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()



class ActivationHook:
    def __init__(self):
        self.pre_features = {}  # 存储输入特征
        self.post_features = {}  # 存储输出特征
        self.handles = []
    
    def hook_fn(self, name):
        def _hook(module, input, output):
            # 打印输入形状
            if isinstance(input, tuple):
                print(f"\n{name} input shapes:")
                for idx, i in enumerate(input):
                    if isinstance(i, torch.Tensor):
                        print(f"  input[{idx}]: {i.shape}")
            else:
                if isinstance(input, torch.Tensor):
                    print(f"\n{name} input shape: {input.shape}")

            # 打印输出形状
            if isinstance(output, tuple):
                print(f"{name} output shapes:")
                for idx, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        print(f"  output[{idx}]: {o.shape}")
            else:
                if isinstance(output, torch.Tensor):
                    print(f"{name} output shape: {output.shape}")

            # 存储特征
            if isinstance(input, tuple):
                self.pre_features[name] = [i.detach().cpu() for i in input]
            else:
                self.pre_features[name] = input.detach().cpu()

            if isinstance(output, tuple):
                self.post_features[name] = [o.detach().cpu() for o in output]
            else:
                self.post_features[name] = output.detach().cpu()
        return _hook
    
    def register(self, model, layer_dict):
        """注册hook层示例
        layer_dict = {
            'backbone.stage1': model.backbone.stage1,
            'decoder.block1': model.decoder[0]
        }
        """
        for name, layer in layer_dict.items():
            handle = layer.register_forward_hook(self.hook_fn(name))
            self.handles.append(handle)
    
    def release(self):
        """释放所有hook"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.post_features={}
        self.pre_features = {}


def save_features(pre_features, post_features, save_dir):
    """保存输入和输出特征"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建输入和输出特征的子目录
    pre_dir = os.path.join(save_dir, 'input_features')
    post_dir = os.path.join(save_dir, 'output_features')
    os.makedirs(pre_dir, exist_ok=True)
    os.makedirs(post_dir, exist_ok=True)
    
    # 保存输入特征
    for name, tensor in pre_features.items():
        _save_feature(tensor, name, pre_dir)
    
    # 保存输出特征
    for name, tensor in post_features.items():
        _save_feature(tensor, name, post_dir)

def _save_feature(tensor, name, save_dir):
    """辅助函数：保存单个特征"""
    # 处理列表类型的特征
    if isinstance(tensor, list):
        for i, t in enumerate(tensor):
            # 保存原始数据
            save_path = os.path.join(save_dir, f"{name}_part{i}.pt")
            torch.save(t, save_path)
            
            # 对每个部分都生成可视化
            if t.dim() == 4:  # 如果是4D张量
                mean_feat = t.mean(dim=1)[0]
                plt.imsave(os.path.join(save_dir, f"{name}_part{i}_heatmap.png"), 
                          mean_feat.numpy(), cmap='viridis')
            elif t.dim() == 2:  # 如果是2D张量
                plt.figure()
                plt.hist(t[0].numpy(), bins=50)
                plt.title(f"{name}_part{i}")
                plt.savefig(os.path.join(save_dir, f"{name}_part{i}_dist.png"))
                plt.close()
        return
            
    # 处理4D张量（卷积特征）
    if tensor.dim() == 4:  # [B,C,H,W]
        # 保存原始特征
        torch.save(tensor, os.path.join(save_dir, f"{name}_full.pt"))
        
        # 生成通道平均热力图
        mean_feat = tensor.mean(dim=1)[0]
        plt.imsave(os.path.join(save_dir, f"{name}_heatmap.png"), 
                  mean_feat.numpy(), cmap='viridis')
        
        # 额外生成每个通道的热力图（取前16个通道）
        for i in range(min(16, tensor.size(1))):
            channel_feat = tensor[0, i].cpu().numpy()
            plt.imsave(os.path.join(save_dir, f"{name}_channel_{i}.png"),
                      channel_feat, cmap='viridis')
        
        print(f"保存特征: {name}，大小: {tensor.size()}，均值: {tensor.mean().item()}, 方差: {tensor.var().item()}")
            
    # 处理2D张量（线性特征）
    elif tensor.dim() == 2:  # [B,D]
        torch.save(tensor, os.path.join(save_dir, f"{name}.pt"))
        plt.figure(figsize=(10, 6))
        plt.hist(tensor[0].numpy(), bins=50)
        plt.title(f"{name} Distribution")
        plt.savefig(os.path.join(save_dir, f"{name}_dist.png"))
        plt.close()
        print(f"保存特征: {name}，大小: {tensor.size()}，均值: {tensor.mean().item()}, 方差: {tensor.var().item()}")
            
    # 处理3D张量（序列特征）
    elif tensor.dim() == 3:  # [B,L,D]
        torch.save(tensor[:, :10], os.path.join(save_dir, f"{name}_top10.pt"))
        # 生成序列特征的热力图
        plt.figure(figsize=(12, 8))
        plt.imshow(tensor[0].cpu().numpy(), aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f"{name} Sequence Features")
        plt.savefig(os.path.join(save_dir, f"{name}_seq_heatmap.png"))
        plt.close()
    else:
        print(f"无法处理的特征维度: {tensor.dim()} for {name}")

def process_image(model, transform, device, img_path, output_dir,hook):
    # Load and process image
    hook.pre_features.clear()
    hook.post_features.clear()
    image = Image.open(img_path).convert('RGB')
    images = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(images)

    #保存中间特征测试
    base_name = os.path.basename(img_path).split('.')[0]
    save_features(hook.pre_features,hook.post_features, os.path.join(output_dir, "intermediate_features", base_name))


    # Process output
    output = output.permute(0, 2, 1, 3)
    pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
    mask = get_color_pallete(pred, args.dataset)
    
    # Save result
    outname = os.path.basename(img_path).split('.')[0] + '_mask.png'
    mask.save(os.path.join(output_dir, outname))
    print(f'Processed: {os.path.basename(img_path)} -> {outname}')

def demo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    if not os.path.exists(config.outdir):
        os.makedirs(config.outdir)
    
    #创建hook特征目录
    feature_dir = os.path.join(config.outdir, "intermediate_features")
    os.makedirs(feature_dir, exist_ok=True)

    # Get first 20 images
    image_paths = sorted(glob(os.path.join(config.input_dir, '*.png')))[:1]
    if not image_paths:
        print(f"No images found in {config.input_dir}")
        return
    
    # Prepare transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Load model
    model = get_segmentation_model(args.model).to(device)
    checkpoint = torch.load(
        '/home/ma-user/work/ymxwork/NIPS/train/runs/ckpt/last_yoloworld_resnet50_night_epoch_780_mean_iu_0.52015.pth',
        map_location=device
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    hook = ActivationHook()
    hook.register(model, {
        # 骨干网络特征
        'backbone': model.detr.backbone.image_model,
        # 文本编码投影层
        'text_projection': model.detr.backbone.text_model.model.text_projection,
        'fam0':model.detr.fam0,
        'fam1':model.detr.fam1,
        'fam2':model.detr.fam2,
        'lfe0':model.detr.lfe0,
        'lfe1':model.detr.lfe1,
        'lfe2':model.detr.lfe2,
        # 分割头最终层
        #'seg_head.final': model.detr.final_seg[-1]
    })
 
    #hook.release()



    # print("\n可用层列表:")
    # for name, module in model.named_modules():
    #     if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):  # 过滤关键层
    #         print(f"层名: {name:30} | 类型: {str(module.__class__)[8:-2]}")
    #     #print('Model loaded successfully')
    
    # Process images
    for img_path in image_paths:
        process_image(model, transform, device, img_path, config.outdir,hook)

if __name__ == '__main__':
    demo(args)