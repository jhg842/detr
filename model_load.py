# import torch
# import torch.nn as nn

# from models import build_model
# import argparse
# import time
# from PIL import Image
# import requests
# import matplotlib.pyplot as plt

# import torch
# from torch import nn
# from torchvision.models import resnet50
# import torchvision.transforms as T
# torch.set_grad_enabled(False);
# def get_args_parser():
#     parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
#     parser.add_argument('--lr', default=1e-4, type=float)
#     parser.add_argument('--lr_backbone', default=1e-5, type=float)
#     parser.add_argument('--batch_size', default=2, type=int)
#     parser.add_argument('--weight_decay', default=1e-4, type=float)
#     parser.add_argument('--epochs', default=300, type=int)
#     parser.add_argument('--lr_drop', default=200, type=int)
#     parser.add_argument('--clip_max_norm', default=0.1, type=float,
#                         help='gradient clipping max norm')

#     # Model parameters
#     parser.add_argument('--frozen_weights', type=str, default=None,
#                         help="Path to the pretrained model. If set, only the mask head will be trained")
#     # * Backbone
#     parser.add_argument('--backbone', default='resnet50', type=str,
#                         help="Name of the convolutional backbone to use")
#     parser.add_argument('--dilation', action='store_true',
#                         help="If true, we replace stride with dilation in the last convolutional block (DC5)")
#     parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
#                         help="Type of positional embedding to use on top of the image features")

#     # * Transformer
#     parser.add_argument('--enc_layers', default=6, type=int,
#                         help="Number of encoding layers in the transformer")
#     parser.add_argument('--dec_layers', default=6, type=int,
#                         help="Number of decoding layers in the transformer")
#     parser.add_argument('--dim_feedforward', default=2048, type=int,
#                         help="Intermediate size of the feedforward layers in the transformer blocks")
#     parser.add_argument('--hidden_dim', default=256, type=int,
#                         help="Size of the embeddings (dimension of the transformer)")
#     parser.add_argument('--dropout', default=0.1, type=float,
#                         help="Dropout applied in the transformer")
#     parser.add_argument('--nheads', default=8, type=int,
#                         help="Number of attention heads inside the transformer's attentions")
#     parser.add_argument('--num_queries', default=100, type=int,
#                         help="Number of query slots")
#     parser.add_argument('--pre_norm', action='store_true')

#     # * Segmentation
#     parser.add_argument('--masks', action='store_true',
#                         help="Train segmentation head if the flag is provided")

#     # Loss
#     parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
#                         help="Disables auxiliary decoding losses (loss at each layer)")
#     # * Matcher
#     parser.add_argument('--set_cost_class', default=1, type=float,
#                         help="Class coefficient in the matching cost")
#     parser.add_argument('--set_cost_bbox', default=5, type=float,
#                         help="L1 box coefficient in the matching cost")
#     parser.add_argument('--set_cost_giou', default=2, type=float,
#                         help="giou box coefficient in the matching cost")
#     # * Loss coefficients
#     parser.add_argument('--mask_loss_coef', default=1, type=float)
#     parser.add_argument('--dice_loss_coef', default=1, type=float)
#     parser.add_argument('--bbox_loss_coef', default=5, type=float)
#     parser.add_argument('--giou_loss_coef', default=2, type=float)
#     parser.add_argument('--eos_coef', default=0.1, type=float,
#                         help="Relative classification weight of the no-object class")

#     # dataset parameters
#     parser.add_argument('--dataset_file', default='coco')
#     parser.add_argument('--coco_path', type=str)
#     parser.add_argument('--coco_panoptic_path', type=str)
#     parser.add_argument('--remove_difficult', action='store_true')

#     parser.add_argument('--output_dir', default='',
#                         help='path where to save, empty for no saving')
#     parser.add_argument('--device', default='cuda',
#                         help='device to use for training / testing')
#     parser.add_argument('--seed', default=42, type=int)
#     parser.add_argument('--resume', default='', help='resume from checkpoint')
#     parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
#                         help='start epoch')
#     parser.add_argument('--eval', action='store_true')
#     parser.add_argument('--num_workers', default=2, type=int)

#     # distributed training parameters
#     parser.add_argument('--world_size', default=1, type=int,
#                         help='number of distributed processes')
#     parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
#     return parser

# parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
# args = parser.parse_args()

# detr, _, _ = build_model(args)

# check_point = torch.load('/mnt/d/Users/jhg84/hospital_nii/label_selection/detr-r50.pth', map_location='cpu')
# detr.load_state_dict(check_point['model'])

# # for param in detr.backbone.parameters():
# #     param.requires_grad = False

# for name, param in detr.named_parameters():
#     if not name == 'transformer.encoder.layers.5.self_attn':
#         print(name)

# CLASSES = [
#     'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
#     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
#     'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
#     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
#     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
#     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#     'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
#     'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
#     'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
#     'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
#     'toothbrush'
# ]
# COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
#           [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
# transform= T.Compose([
#     T.Resize(800), # * 이유.
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # dataset을 확인하면 된다
# ])

# # # output bounding box 후처리
# def box_cxcywh_to_xyxy(x):
#     x_c, y_c, w, h = x.unbind(1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#          (x_c + 0.5 * w), (y_c + 0.5 * h)]
    
#     return torch.stack(b, dim=1)

# def rescale_bboxes(out_bbox, size):
#     img_w, img_h = size
#     b = box_cxcywh_to_xyxy(out_bbox)
#     b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
#     return b

# def detect(im, model, transform):
    
#     # input image를 정규화해줍니다. (batch-size : 1)
#     img = transform(im).unsqueeze(0)
    
#     # demo의 경우 aspect ratio를 0.5와 2사이만 지원합니다.
#     # 이 범위 밖의 이미지를 사용하고 싶다면 maximum size을 1333이하로 rescaling해야 합니다.
#     assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600
    
#     # model을 통과시킵니다. 
#     outputs = model(img)
    
#     # 70 % 이상의 정확도를 가진 예측만 남깁니다.
#     probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
#     keep = probas.max(-1).values > 0.7
    
#     # 0과 1사이의 boxes 값을 image scale로 확대합니다.
#     bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
#     return probas[keep], bboxes_scaled

# # # url = 'https://news.imaeil.com/inc/photos/2020/11/02/2020110216374231552_l.jpg'
# im = Image.open('/mnt/d/Users/jhg84/hospital_nii/label_selection/cross_ko.jpg')

# start = time.time()

# scores, boxes = detect(im, detr, transform)

# print("Inference time :", round(time.time()-start, 3), 'sec')


# def plot_results(pil_img, prob, boxes):
#     plt.figure(figsize=(16,10))
#     plt.imshow(pil_img)
#     ax = plt.gca()
#     for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
#         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                    fill=False, color=c, linewidth=3))
#         cl = p.argmax()
#         text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
#         ax.text(xmin, ymin, text, fontsize=15,
#                 bbox=dict(facecolor='yellow', alpha=0.5))
#     plt.axis('off')
#     plt.savefig('/mnt/d/Users/jhg84/hospital_nii/label_selection/result_ko.jpg')
    
# plot_results(im, scores, boxes)

# img = transform(im).unsqueeze(0)

# # 모델 통과

# outputs = detr(img)
# probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
# keep = probas.max(-1).values > 0.7
# conv_features, enc_attn_weights, dec_attn_weights = [], [], []
# bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
# # 모델을 통과시킬 때 값을 저장할 수 있게끔.
# hooks = [
#     detr.backbone[-2].register_forward_hook(
#         lambda self, input, output: conv_features.append(output)
#     ),
#     detr.transformer.encoder.layers[-1].self_attn.register_forward_hook( 
#         lambda self, input, output: enc_attn_weights.append(output[1]) # encoder : 여섯 개의 layer 중 last lyaer
#     ),
#     detr.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
#         lambda self, input, output: dec_attn_weights.append(output[1]) # decoder : 여섯 개의 layer 중 last layer
#     ),
# ]

# outputs = detr(img)

# for hook in hooks:
#     hook.remove()
    
# conv_features = conv_features[0]
# enc_attn_weights = enc_attn_weights[0]
# dec_attn_weights = dec_attn_weights[0]
# h, w = conv_features['0'].tensors.shape[-2:]

# fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
# colors = COLORS * 100
# for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
#     ax = ax_i[0]
#     ax.imshow(dec_attn_weights[0, idx].view(h, w))
#     ax.axis('off')
#     ax.set_title(f'query id: {idx.item()}')
#     ax = ax_i[1]
#     ax.imshow(im)
#     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                fill=False, color='blue', linewidth=3))
#     ax.axis('off')
#     ax.set_title(CLASSES[probas[idx].argmax()])
# fig.tight_layout()
# plt.savefig('/mnt/d/Users/jhg84/hospital_nii/label_selection/attention_map_cross.jpg')
# plt.close('all')


# f_map = conv_features['0']
# print("Encoder attention:      ", enc_attn_weights[0].shape)
# print("Feature map:            ", f_map.tensors.shape)

# shape = f_map.tensors.shape[-2:]
# # more interpretable shape으로 reshape해준다. (encoder token의 길이는 HxW 이기 때문에 아래와 같이 reshape한다.)
# sattn = enc_attn_weights[0].reshape(shape + shape)
# print("Reshaped self-attention:", sattn.shape)
# # downsampling factor for the CNN(32 for DETR, 16 for DETR DC5)
# fact = 32

# # 시각화를 위해 4개의 reference points를 고른다. ()
# idxs = [(700, 900), (550, 50), (500, 500), (700, 1050),]

# # canvas 생성
# fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
# # and we add one plot per reference point
# gs = fig.add_gridspec(2, 4)
# axs = [
#     fig.add_subplot(gs[0, 0]),
#     fig.add_subplot(gs[1, 0]),
#     fig.add_subplot(gs[0, -1]),
#     fig.add_subplot(gs[1, -1]),
# ]

# # reference point 마다 self-attention plot.
# # for that point
# for idx_o, ax in zip(idxs, axs):
#     idx = (idx_o[0] // fact, idx_o[1] // fact)
#     ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
#     ax.axis('off')
#     ax.set_title(f'self-attention{idx_o}')

# # central image를 더해준다.
# # reference points : 빨간 원
# fcenter_ax = fig.add_subplot(gs[:, 1:-1])
# fcenter_ax.imshow(im)
# for (y, x) in idxs:
#     scale = im.height / img.shape[-2]
#     x = ((x // fact) + 0.5) * fact
#     y = ((y // fact) + 0.5) * fact
#     fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
#     fcenter_ax.axis('off')
# plt.savefig('/mnt/d/Users/jhg84/hospital_nii/label_selection/attention_map_cross_4.jpg')