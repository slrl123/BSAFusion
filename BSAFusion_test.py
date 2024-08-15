import os
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.BrainDataset_2D import TestData
from utils_2d.warp import Warper2d, warp2D
from modal_2d.RegFusion_lite import Encoder, ModelTransfer_lite, RegNet_lite, FusionNet_lite
import torch

from utils_2d.utils import project, rgb2ycbcr, ycbcr2rgb
import warnings

modal = 'PET'

warnings.filterwarnings('ignore') # 忽略警告

image_warp = Warper2d()
device = torch.device('cpu')
checkpoint_path = './checkpoint'


checkpoint = torch.load(os.path.join(checkpoint_path, f'BSAFusion_{modal}.pkl'), map_location=torch.device('cpu'))
# checkpoint = torch.load(os.path.join(checkpoint_path, f'Reg_0317_{modal}_reg.pkl'), map_location=torch.device('cpu'))

encoder = Encoder().to(device)
transfer = ModelTransfer_lite(num_vit=2, num_heads=4, img_size=[256, 256]).to(device)
reg_net = RegNet_lite().to(device)
fusion_net = FusionNet_lite().to(device)

encoder.load_state_dict(checkpoint['encoder_state_dict'])
transfer.load_state_dict(checkpoint['transfer_state_dict'])
reg_net.load_state_dict(checkpoint['reg_net_state_dict'])
fusion_net.load_state_dict(checkpoint['fusion_net_state_dict'])


if modal == 'CT':
    val_dataset = TestData(
        img1_folder=f'./data/testData/{modal}/MRI',
        img2_folder=f'./data/testData/{modal}/{modal}',
        modal=modal
    )
else:
    val_dataset = TestData(
        img1_folder=f'./data/testData/{modal}/MRI',
        img2_folder=f'./data/testData/{modal}/{modal}_RGB',
        modal=modal
    )

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=1,
    pin_memory=True,
    shuffle=False,
    num_workers=14)

def validate_mask(encoder, transfer, reg_net, fusion_net, dataloader, modal):
    # torch.cuda.empty_cache()
    # torch.backends.cudnn.benchmark = True
    epoch_iterator = tqdm(dataloader, desc='Val (X / X Steps) (loss=X.X)', ncols=150, leave=True, position=0)
    encoder.eval()
    transfer.eval()
    reg_net.eval()
    fusion_net.eval()

    figure_save_path = f"./{modal}_result"
    if not os.path.exists(figure_save_path):
        os.makedirs(os.path.join(figure_save_path, "MRI"))
        os.makedirs(os.path.join(figure_save_path, f"{modal}"))
        os.makedirs(os.path.join(figure_save_path, f"Fusion"))
        os.makedirs(os.path.join(figure_save_path, f"{modal}_align"))
        os.makedirs(os.path.join(figure_save_path, f"{modal}_label"))

    with torch.no_grad():
        for i, batch in enumerate(epoch_iterator):
            img1, img2, file_name = batch
            H, W = img1.shape[2], img1.shape[3]
            if modal == 'CT':
                img1, img2 = img1.to(device), img2.to(device)
            else:
                img1, img2 = img1.to(device), img2.to(device)
                img2 = rgb2ycbcr(img2)
                img2_cbcr = img2[:,1:3,:,:]
                img2 = img2[:,0:1,:,:]
            AS_F, feature1 = encoder(img1)
            BS_F, feature2 = encoder(img2)
            pre1, pre2, feature_pred1, feature_pred2, feature1, feature2, AU_F, BU_F = transfer(feature1, feature2)

            feature1 = project(feature1, [H, W]).to(device)
            feature2 = project(feature2, [H, W]).to(device)
            AU_F = project(AU_F, [H, W])
            BU_F = project(BU_F, [H, W])
            img1, img2 = img1.to(device), img2.to(device)
            _, _, flows, flow, _, _ = reg_net(feature1, feature2)
            warped_image2 = image_warp(flow, img2)
            fusion_img = fusion_net(AS_F, BS_F, AU_F, BU_F, flow)

            """save as png"""
            if modal == 'CT':
                pass
            else:
                fusion_cbcr = warp2D()(img2_cbcr, flow)
                fusion_img = torch.cat((fusion_img, fusion_cbcr), dim=1)
                fusion_img = ycbcr2rgb(fusion_img)

            save_image(img1, os.path.join(figure_save_path, f"MRI/{file_name[0]}"))
            save_image(img2, os.path.join(figure_save_path, f"{modal}/{file_name[0]}"))
            save_image(fusion_img, os.path.join(figure_save_path, f"Fusion/{file_name[0]}"))
            save_image(warped_image2, os.path.join(figure_save_path, f"{modal}_align/{file_name[0]}"))

    return


validate_mask(encoder, transfer, reg_net, fusion_net, val_dataloader, modal)



