import random
import os
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

from dataset.mnist_m import MNISTM
from model.model_ed import ModelED
from model.loss import EncoderDecoderLoss
from model.logger import ExpLogger
from common.device_funcs import to_device
from common.log_util import get_host_ip, get_hostname, get_cuda_version, get_python_version, log_summary
from common.model_summary import model_summary
from common.hparams import hparams_debug_string
from test import test

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True
cudnn.benchmark = True


def prepare_dataloader(source_image_root, batch_size = 64, image_size = 28, ):
    #######################
    # load data           #
    #######################

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

    # dataset_source = datasets.MNIST(
    #     root=source_image_root,
    #     train=True,
    #     transform=gray2rgb_transform
    # )

    # dataloader_source = torch.utils.data.DataLoader(
    #     dataset=dataset_source,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=8
    # )

    dataset_target = MNISTM(
        root=source_image_root,
        train=True,
        transform=img_transform,
        download=False,
    )

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    return dataloader_target



def lr_scheduler_update(optimizer, step, init_lr=0.001, lr_decay_step=1000, step_decay_weight=0.9):

    # Decay learning rate by a factor of step_decay_weight every lr_decay_step
    current_lr = init_lr * (step_decay_weight ** (step // lr_decay_step))

    if step % lr_decay_step == 0:
        print( 'learning rate is set to %f' % current_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer, current_lr


def get_model(config):
    model_name = config['model']
    if model_name == 'ModelED':
        model = ModelED(config)
    elif model_name == 'ModelNTI':
        from model.model_ed import ModelNTI
        model = ModelNTI(config)
    elif model_name == 'ModelST':
        from model.model_st import ModelST
        model = ModelST(config)
    elif model_name == 'ModelSVC':
        from model.model_st import ModelSVC
        model = ModelSVC(config)
    elif model_name == 'ModelSVB':
        from model.model_st import ModelSVB
        model = ModelSVB(config)
    else:
        raise ValueError()

    return model

def get_loss_fn(config):
    model_name = config['model']
    if model_name in ['ModelED', 'ModelNTI', 'ModelST', 'ModelSVC', 'ModelSVB']:
        criterion = EncoderDecoderLoss()
    else:
        raise ValueError()
    return criterion


def main(
    config,
    log_dir = './log/',
    ckpt_dir = './ckpt/',
):
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    image_root = os.path.join('./data/')
    image_size=28

    lr = 1e-2

    n_epoch = 50
    step_decay_weight = 0.95
    lr_decay_step = 2000
    weight_decay = 1e-6
    grad_clip_thresh = 1.0

    momentum = 0.9
    ##################################
    #  load model, setup optimizer   #
    ##################################

    my_net = get_model(config).to(device)
    criterion = get_loss_fn(config).to(device)

    optimizer = optim.SGD(my_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    logger = ExpLogger(log_dir)

    my_net.train()

    # print(my_net)
    log_summary(
        os.path.join(log_dir, "summary.log"),
        {
            '\nHost Name': get_hostname(),
            'Host IP': get_host_ip(),
            'Python Version': get_python_version(),
            'CUDA Version': get_cuda_version(),
            'PyTorch Version': torch.__version__,
            '\nModel': model_summary(my_net),
            '\nConfig': hparams_debug_string(config),
        }
    )


    #############################
    # training network          #
    #############################
    dataloader = prepare_dataloader(image_root, batch_size=64, image_size=image_size)

    len_dataloader = len(dataloader)

    current_step = 0
    for epoch in range(n_epoch):
        data_iter = iter(dataloader)
        for i in range(len_dataloader):
            # Update LR
            optimizer, current_lr = lr_scheduler_update(optimizer=optimizer, step=current_step,
                init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight)

            data_target = data_iter.next()
            data_target = to_device(data_target, device=device)
            input_img, class_label = data_target

            # Forward Backward
            my_net.zero_grad()
            result = my_net(input_data=input_img, number=class_label)
            ref_code, rec_img = result

            loss, target_mse = criterion(data_target, result)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(my_net.parameters(), grad_clip_thresh)

            optimizer.step()

            logger.log_training(loss.data.cpu().numpy(), grad_norm.data.cpu().numpy(), current_lr, current_step)

            current_step += 1

        print('Step: %d, Epoch %d, loss total: %f,  mse: %f' % ( 
                current_step,
                epoch,
                loss.data.cpu().numpy(),
                target_mse.data.cpu().numpy()
            )
        )

        # print('step: %d, loss: %f' % (current_step, loss.cpu().data.numpy()))
        torch.save(my_net.state_dict(), ckpt_dir + '/sv_mnist_' + str(epoch) + '.pth')

        test(my_net, criterion, epoch, current_step, name='mnist_m', logger=logger, log_dir=log_dir)


if __name__ == '__main__':
    ######################
    # params             #
    ######################
    parser = argparse.ArgumentParser(
        description="Train Disentangle (See detail in train.py)."
    )
    parser.add_argument("--ckpt_dir", type=str, default="exp/untitled/ckpt",
        required=False, help="Directory to save checkpoint"
    )
    parser.add_argument("--log_dir", type=str, default="exp/untitled/log",
        required=False, help="Directory to save logs"
    )
    parser.add_argument("--model", type=str, default="ModelED",
        required=False, help="model to train"
    )

    args = parser.parse_args()
    config = {
        'model':args.model,
        'code_size': 128,
        'n_class': 10,
        'token_num': 5,
    }
    main(config, args.log_dir, args.ckpt_dir)
    print('Done!')




