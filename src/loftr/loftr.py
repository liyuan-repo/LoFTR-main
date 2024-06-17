import time
import torch
import torch.nn as nn
from einops.einops import rearrange
from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching
from thop import profile


class LoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)              # output resolution are 1/8 and 1/2.
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],                    # 'd_model': 256,  'temp_bug_fix': False
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()
        #    'coarse': {'d_model': 256, 'd_ffn': 256, 'nhead': 8,
        #               'layer_names': ['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross'],
        #               'attention': 'linear', 'temp_bug_fix': False},

    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1.----------------------------------------Local Feature CNN---------------------------------------------------
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            # 返回 1/8的粗略特征图feats_c, 1/2的精细特征图feats_f
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

            # feats_c={Tensor:(2, 256, 64, 64)}, feats_f={Tensor:(2, 128, 256, 256)}
            # feat_c0={Tensor:(1, 256, 64, 64)}, feat_c1={Tensor:(1, 256, 64, 64)}
            # feat_f0={Tensor:(1, 128, 256, 256)}, feat_f1={Tensor:(1, 128, 256, 256)}

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })
        # 'hw0_c'={Tensor:([64, 64])}, 'hw1_c'={Tensor:([64, 64])}
        # 'hw0_f'=torch.Size([256, 256]),'hw1_f'=torch.Size([256, 256])

        # 2. --------------------------------------coarse-level loftr module--------------------------------------------
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = self.pos_encoding(feat_c0)
        feat_c1 = self.pos_encoding(feat_c1)

        feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')  # 经过位置编码的特征图
        feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')
        # 位置编码维度==输入的图像维度(必须)，相同维度的向量才能相加

        # feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        # feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        # feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
        # feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)   # run time:1.632
        # flops, params = profile(self.loftr_coarse, inputs=(feat_c0, feat_c1))
        # print("loftr_coarse参数量：", "%.2f" % (params / (1000 ** 2)), "M")     # loftr_coarse参数量： 5.25 M
        # print("GFLOPS：", "%.2f" % (flops / (1000 ** 3)))                      # GFLOPS： 43.08

        # 3.------------------------------------------match coarse-level------------------------------------------------

        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4.-----------------------------------------fine-level refinement----------------------------------------------
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        # tif = time.time()
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # flops, params = profile(self.loftr_fine, inputs=(feat_f0_unfold, feat_f1_unfold))
        # print("loftr_fine参数量：", "%.2f" % (params / (1000 ** 2)), "M")  # loftr_fine参数量： 0.33 M
        # print("GFLOPS：", "%.2f" % (flops / (1000 ** 3)))   # GFLOPS： 10.06

        # 5. ------------------------------------------match fine-level-------------------------------------------------
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
    # 在pytorch中构建好一个模型后，一般需要将torch.load()的预训练权重加载到自己的模型重。
    # torch.load_state_dict()函数就是用于将预训练的参数权重加载到新的模型之中，操作方式如下所示：
    # 读取模型参数
    # weights_dict = torch.load("./w32_256x192.pth")
    # 加载模型参数到模型中
    # model.load_state_dict(weights_dict, strict=False)
