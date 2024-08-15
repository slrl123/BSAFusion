import os
import warnings
import torch
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.BrainDataset_2D import RegDataset_F
from modal_2d.RegFusion_lite import Encoder, ModelTransfer_lite, RegNet_lite, FusionNet_lite
from utils_2d.loss import regFusion_loss
from colorama import Fore, Style
from utils_2d.warp import warp2D
from einops import rearrange

warnings.filterwarnings('ignore')
print(f'{Style.RESET_ALL}')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
multiGPUs = False
device = torch.device('cuda:0')
image_warp = warp2D()

def project(x, image_size):
    """将 torch.Size([1, 512, 768]) 转换成图像形状[channel, W/p, H/p]"""
    W, H = image_size[0], image_size[1]
    x = rearrange(x, 'b (w h) hidden -> b w h hidden', w=W // 16, h=H // 16)
    x = x.permute(0, 3, 1, 2)
    return x

def train(modal,
          train_batch_size,
          lr,
          num_epoch,
          beta1,
          beta2,
          resume):


    checkpoint_path = './checkpoint'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    torch.backends.cudnn.benchmark = True

    train_dataset = RegDataset_F(
        root='./data',
        mode='train',
        model = modal,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=64)


    """create network"""
    encoder = Encoder().to(device)
    transfer = ModelTransfer_lite(num_vit=2, num_heads=4, img_size=[256, 256]).to(device)
    reg_net = RegNet_lite().to(device)
    fusion_net = FusionNet_lite().to(device)
    for par in transfer.modal_dis.parameters():  # freeze modal_dis
        par.requires_grad = False


    optimizer = Adam(encoder.parameters(), lr=lr, betas=(beta1, beta2))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr * 1e-2)

    optimizer1 = Adam(transfer.parameters(), lr=lr, betas=(beta1, beta2))
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=num_epoch, eta_min=lr * 1e-2)

    optimizer2 = Adam(reg_net.parameters(), lr=lr, betas=(beta1, beta2))
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=num_epoch, eta_min=lr * 1e-2)

    optimizer3 = Adam(fusion_net.parameters(), lr=lr, betas=(beta1, beta2))
    scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=num_epoch, eta_min=lr * 1e-2)

    epoch_loss_values = []
    cls_loss_value = []
    transfer_loss_value = []
    flow_loss_value = []
    fusion_loss_value = []
    reg_loss_values = []


    start_epoch = 0
    sum_ssim1 = 1
    sum_ssim2 = 1

    """resume training"""
    if resume:
        checkpoint = torch.load(os.path.join('./checkpoint/checkpoint', 'checkpoint_999.pkl'), map_location=torch.device('cpu'))  # load文件
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        transfer.load_state_dict(checkpoint['transfer_state_dict'])
        reg_net.load_state_dict(checkpoint['reg_net_state_dict'])
        fusion_net.load_state_dict(checkpoint['fusion_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer1.load_state_dict(checkpoint['optimizer1_state_dict'])
        optimizer2.load_state_dict(checkpoint['optimizer2_state_dict'])
        optimizer3.load_state_dict(checkpoint['optimizer3_state_dict'])
        start_epoch = checkpoint['epoch']
        scheduler.last_epoch = start_epoch
        epoch_loss_values = checkpoint['epoch_loss_values']
        cls_loss_value = checkpoint['cls_loss_value']
        reg_loss_values = checkpoint['reg_loss_values']
        transfer_loss_value = checkpoint['transfer_loss_value']
        flow_loss_value = checkpoint['flow_loss_value']
        fusion_loss_value = checkpoint['fusion_loss_value']


    """start training"""
    for iter_num in range(start_epoch, num_epoch):
        encoder.train()
        transfer.train()
        reg_net.train()
        fusion_net.train()

        step = 0
        epoch_loss = 0.0
        cls_losses = 0.0
        transfer_losses = 0.0
        flow_losses = 0.0
        fusion_losses = 0.0
        reg_losses = 0.0
        parameter = (sum_ssim2) / (sum_ssim1)
        sum_ssim1=0.0
        sum_ssim2=0.0

        epoch_iterator = tqdm(
            train_dataloader, desc='Train (X / X Steps) (loss=X.X)', ncols=150, leave=True, position=0)

        for step, batch in enumerate(epoch_iterator):
            step += 1

            img1, img1_2, img2, flow_GT, label1, label2  = batch
            if modal == 'CT':
                img1, img2, img1_2, flow_GT, label1, label2 = img1.to(device), img2.to(device), img1_2.to(device), flow_GT.to(device), label1.to(
                    device), label2.to(device)
            else:
                img1, img2, img1_2, flow_GT, label1, label2 = img1.to(device), img2[:, 0, :, :].unsqueeze(1).to(
                    device), img1_2[:, 0, :, :].unsqueeze(1).to(device), flow_GT.to(device), label1.to(device), label2.to(device)

            AS_F, feature1 = encoder(img1)
            BS_F, feature2 = encoder(img2)

            pre1, pre2, feature_pred1, feature_pred2, feature1, feature2, AU_F, BU_F = transfer(feature1, feature2) # 分类器， 模态鉴别器， 转换后特征， 转换前特征

            feature1 = project(feature1, [256, 256]).to(device)
            feature2 = project(feature2, [256, 256]).to(device)
            AU_F = project(AU_F, [256, 256])
            BU_F = project(BU_F, [256, 256])

            _, _, flows, flow, _, _ = reg_net(feature1, feature2)
            fusion_img = fusion_net(AS_F, BS_F, AU_F, BU_F, flow)

            warped_img2 = image_warp(img2, flow)


            """generate loss"""
            cls_loss, transfer_loss, flow_loss, fusion_loss, reg_loss, ssim1, ssim2 = regFusion_loss(
                label1, label2,
                pre1, pre2,
                feature_pred1, feature_pred2,
                flow, flows, warped_img2, flow_GT,
                img1, img1_2, fusion_img,
                parameter
            )

            loss = cls_loss + transfer_loss + flow_loss + fusion_loss + reg_loss

            sum_ssim1 += ssim1.item()
            sum_ssim2 += ssim2.item()
            epoch_loss += loss.item()
            cls_losses += cls_loss.item()
            transfer_losses += transfer_loss.item()
            flow_losses += flow_loss.item()
            fusion_losses += fusion_loss.item()
            reg_losses += reg_loss.item()

            optimizer.zero_grad()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            loss.backward()

            optimizer.step()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()

            epoch_iterator.set_description(
                "Train(%d/%d Steps) (l=%2.3f)" %
                (iter_num + 1, num_epoch, loss))

        scheduler.step()
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
        transfer.modal_dis.load_state_dict(transfer.classifier.state_dict())  # 转换器参数转移

        epoch_loss_values.append((epoch_loss / step))
        cls_loss_value.append((cls_losses / step))
        transfer_loss_value.append((transfer_losses / step))
        flow_loss_value.append((flow_losses / step))
        fusion_loss_value.append((fusion_losses / step))
        reg_loss_values.append(reg_losses/ step)
        print(f'{Fore.RED}total:{epoch_loss / step:.4f}{Style.RESET_ALL}')

        """保存断点"""

        if (iter_num + 1) % 5 ==0:
            checkpoint = {
                "encoder_state_dict": encoder.state_dict(),
                "transfer_state_dict": transfer.state_dict(),
                "reg_net_state_dict": reg_net.state_dict(),
                "fusion_net_state_dict": fusion_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "optimizer1_state_dict": optimizer1.state_dict(),
                "optimizer2_state_dict": optimizer2.state_dict(),
                "optimizer3_state_dict": optimizer3.state_dict(),
                "epoch": iter_num,
                "epoch_loss_values":epoch_loss_values,
                "cls_loss_value":cls_loss_value,
                "transfer_loss_value":transfer_loss_value,
                "flow_loss_value":flow_loss_value,
                'reg_loss_values':reg_loss_values,
                "fusion_loss_value":fusion_loss_value,
            }
            path_checkpoint = "./checkpoint/checkpoint/checkpoint_999.pkl"
            if not os.path.exists('./checkpoint/checkpoint'):
                os.mkdir('./checkpoint/checkpoint')
            torch.save(checkpoint, path_checkpoint)

    checkpoint_ = {
        "encoder_state_dict": encoder.state_dict(),
        "transfer_state_dict": transfer.state_dict(),
        "reg_net_state_dict": reg_net.state_dict(),
        "fusion_net_state_dict": fusion_net.state_dict()}
    torch.save(checkpoint_, os.path.join(checkpoint_path, f'BSAFusion_999_{modal}.pkl'))
    print("Training is complected")


if __name__ == '__main__':
    modal = 'CT'
    train_batch_size = 32
    lr = 5e-5
    num_epoch = 3000
    bata = 0.1
    beta1 = 0.9
    beta2 = 0.999
    resume = False
    a = 1.0  # recon损失平衡参数
    b = 3.0  # sim损失平衡参数
    c = 3.0 # flow平滑损失
    d = 2.0  # flow 平衡参数
    e = 0.0  # dice 平衡参数
    f = 1.0  # ncc 平衡参数

    train(modal,
          train_batch_size,
          lr,
          num_epoch,
          beta1,
          beta2,
          resume)
