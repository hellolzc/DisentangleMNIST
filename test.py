import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from torchvision import datasets
import torchvision.utils as vutils

from dataset.mnist_m import MNISTM
from model.device_funcs import to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_val_dataloader(batch_size = 64, image_size = 28):
    ###################
    # load data       #
    ###################

    img_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # gray2rgb_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.repeat(3,1,1)),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])   # 修改的位置

    
    # if name == 'mnist':
    #     mode = 'source'
    #     image_root = './data/'
    #     dataset = datasets.MNIST(
    #         root=image_root,
    #         train=False,
    #         transform=gray2rgb_transform
    #     )

    #     dataloader = torch.utils.data.DataLoader(
    #         dataset=dataset,
    #         batch_size=batch_size,
    #         shuffle=False,
    #         num_workers=8
    #     )

    # elif name == 'mnist_m':

    # mode = 'target'
    image_root = './data/'
    dataset = MNISTM(
        root=image_root,
        transform=img_transform,
        train=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )
    return dataloader



# def tr_image(img):
#     img_new = (img + 1) / 2
#     return img_new


def test(model, criterion, epoch, step, name='MINIST-M', logger=None):

    ###################
    # params          #
    ###################
    cuda = True
    cudnn.benchmark = True
    ckpt_root = './log/'

    dataloader = prepare_val_dataloader()


    ####################
    # load model       #
    ####################

    model.eval()
    # model = model.to(device)

    ####################
    # transform image  #
    ####################


    len_dataloader = len(dataloader)
    data_iter = iter(dataloader)

    with torch.no_grad():
        n_total = 0
        for i in  range(len_dataloader):
            data_input = data_iter.next()
            data_input = to_device(data_input, device=device)
            input_img, class_label = data_input

            batch_size = len(class_label)


            result = model(input_data=input_img)
            ref_code, rec_img = result

            loss, target_mse = criterion(data_input, result)

            if i == len_dataloader - 2:
                vutils.save_image(input_img, ckpt_root + 'Epoch_%d_ori_image_all.png' % epoch, nrow=8)
                vutils.save_image(rec_img, ckpt_root + 'Epoch_%d_rec_image_all.png' % epoch, nrow=8)

            n_total += batch_size

    print('Epoch: %d, Step %d, dataset: %s, total_loss: %f, rec_mse: %f' % (epoch, step, name, 
            loss.data.cpu().numpy(),
            target_mse.data.cpu().numpy(),
        )
    )

    if logger is not None:
        logger.log_validation(loss, model, step)
