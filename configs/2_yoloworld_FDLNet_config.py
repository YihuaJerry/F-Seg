# 训练集中的类别数量
num_training_classes = 19

# 测试集中的类别数量
num_test_classes = 19

# 文本特征通道的数量
text_channels = 512

# YOLOWorld 模型中 neck 部分的嵌入通道
neck_embed_channels = [128, 256, 128]

# YOLOWorld 模型中 neck 部分的头数
neck_num_heads = [4, 8, 4]

# 训练过程中是否允许掩膜重叠
mask_overlap = False
