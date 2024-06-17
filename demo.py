import time
import cv2
import torch
import demo_utils
import noise_color
import numpy as np
from thop import profile
from src.loftr import LoFTR, default_cfg
from sklearn.metrics import mean_squared_error


def test_image_matching():
    matcher = LoFTR(config=default_cfg)
    # matcher.load_state_dict(torch.load("weights/loftr.ckpt")['state_dict'])
    matcher.cuda()
    matcher = matcher.eval()

    # Load example images
    img0_path = "assets/1DSMsets/pair1-2.jpg"
    img1_path = "assets/1DSMsets/pair1-1.jpg"

    # -----------------------origin image ------------------------
    output_path = "./output/1DSMsets/pair1r.jpg"
    img0 = cv2.imread(img0_path)
    img0 = demo_utils.resize(img0, 512)

    # --------------------Additive noise image ------------------
    # output_path = "./output/1DSMsets/pair1+snr0.jpg"
    # img0 = noise_color.Additive_noise(img0_path, SNR=0)

    # --------------------stripe noise image --------------------
    # output_path = "output/1DSMsets/pair1+0p101S.jpg"
    # img0 = noise_color.stripe_noise(img0_path, 0.1)

    img1 = cv2.imread(img1_path)
    img1 = demo_utils.resize(img1, 512)

    img0_g = cv2.imread(img0_path, 0)
    img1_g = cv2.imread(img1_path, 0)  # 读入灰度图片，可用0作为实参替代

    img0_g, img1_g = demo_utils.resize(img0_g, 512), demo_utils.resize(img1_g, 512)
    batch = {'image0': torch.from_numpy(img0_g / 255.)[None, None].cuda().float(),
             'image1': torch.from_numpy(img1_g / 255.)[None, None].cuda().float()}

    start = time.time()
    with torch.no_grad():  # 不需要进行网络参数的更新就不用反向传播

        matcher(batch)

        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()

    # --------------------------RANSAC Outlier Removal----------------------------------
    # F_hat, mask_F = cv2.findFundamentalMat(mkpts0, mkpts1, method=cv2.USAC_FAST,
    #                                        ransacReprojThreshold=1, confidence=0.999)
    # F_hat, mask_F = cv2.findFundamentalMat(mkpts0, mkpts1, method=cv2.USAC_ACCURATE,
    #                                        ransacReprojThreshold=1, confidence=0.999)
    F_hat, mask_F = cv2.findFundamentalMat(mkpts0, mkpts1, method=cv2.USAC_MAGSAC,
                                           ransacReprojThreshold=1, confidence=0.999)

    end = time.time()
    tsum = end - start

    if mask_F is not None:
        mask_F = mask_F[:, 0].astype(bool)
    else:
        mask_F = np.zeros_like(mkpts0[:, 0]).astype(bool)
        # 得到mask_F是一个bool类型的N维数组，{ndarray:(255)},dtype=bool   mask_F=[False ... True ... False... True]

    # --------------------------------visualize match----------------------------------
    # display = demo_utils.draw_match(img0, img1, mkpts0, mkpts1)
    display = demo_utils.draw_match(img0, img1, mkpts0[mask_F], mkpts1[mask_F])

    putative_num = len(mkpts0)
    correct_num = len(mkpts0[mask_F])
    inliner_ratio = correct_num / putative_num
    # -------------------------------RMSE计算---------------------------------

    text1 = "putative_num:{}".format(putative_num)
    text2 = 'correct_num:{}'.format(correct_num)
    text3 = 'inliner ratio:%.3f' % inliner_ratio
    text4 = 'run time: %.3fs' % tsum

    print('putative_num:{}'.format(putative_num), '\ncorrect_num:{}'.format(correct_num),
          '\ninliner ratio:%.3f' % inliner_ratio, '\nrun time: %.3f' % tsum)

    cv2.putText(display, str(text1), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(display, str(text2), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(display, str(text3), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(display, str(text4), (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(output_path, display)

    flops, params = profile(matcher, inputs=(batch,))
    print("Params：", "%.2f" % (params / (1000 ** 2)), "M")
    print("GFLOPS：", "%.2f" % (flops / (1000 ** 3)))


if __name__ == '__main__':
    test_image_matching()

# config={'backbone_type': 'ResNetFPN',
#            'resolution': (8, 2),
#            'fine_window_size': 5,
#            'fine_concat_coarse_feat': True,

#            'resnetfpn': {'initial_dim': 128,'block_dims': [128, 196, 256]},
#
#            'coarse': {'d_model': 256, 'd_ffn': 256, 'nhead': 8,
#            'layer_names': ['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross'],
#            'attention': 'linear', 'temp_bug_fix': False},
#
#            'match_coarse': {'thr': 0.2, 'border_rm': 2, 'match_type': 'dual_softmax',
#            'dsmax_temperature': 0.1, 'skh_iters': 3, 'skh_init_bin_score': 1.0, 'skh_prefilter': True,
#            'train_coarse_percent': 0.4, 'train_pad_num_gt_min': 200},
#
#            'fine': {'d_model': 128, 'd_ffn': 128, 'nhead': 8, 'layer_names': ['self', 'cross'],
#            'attention': 'linear'}}
#
