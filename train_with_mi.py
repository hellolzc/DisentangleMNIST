import random
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from model.logger import ExpLogger
from common.device_funcs import to_device
from common.hparams import create_hparams
from utils import prepare_dataloader, get_model, get_loss_fn, log_exp
from model.mi_estimators import CLUBSample
from test_with_mi import test

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True
cudnn.benchmark = True


def lr_scheduler_update(optimizer, step, init_lr=0.001, lr_decay_step=1000, step_decay_weight=0.9):

    # Decay learning rate by a factor of step_decay_weight every lr_decay_step
    current_lr = init_lr * (step_decay_weight ** (step // lr_decay_step))

    if step % lr_decay_step == 0:
        print( 'learning rate is set to %f' % current_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer, current_lr




def mi_first_forward(model, mi_net, optimizer_mi_net, data_target):
    optimizer_mi_net.zero_grad()

    input_img, class_label = data_target

    with torch.no_grad():
        weights, scores, style_embs, text_embs = model.encode(input_data=input_img, number=class_label)

    x = style_embs.detach()
    y = text_embs.detach()

    lld_loss = mi_net.negative_loglikeli(x, y)
    lld_loss.backward()
    optimizer_mi_net.step()

    return lld_loss



def mi_second_forward(my_net, optimizer, criterion, mi_net, data_target, config, grad_clip_thresh=1.0):
    use_mi=config['use_mi']
    mi_loss_weight= config['loss_weight_mi']
    # Forward Backward
    input_img, class_label = data_target
    my_net.zero_grad()
    result = my_net(input_data=input_img, number=class_label)
    # ref_code, rec_img = result

    loss, loss_dict = criterion(data_target, result)


    _, _, weights, scores, style_embs, text_embs = result
    mi_loss = mi_net.mi_est(style_embs, text_embs)
    loss_dict['mi_loss'] = mi_loss
    if use_mi:
        loss += mi_loss_weight * mi_loss

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(my_net.parameters(), grad_clip_thresh)

    optimizer.step()
    return loss, loss_dict, grad_norm


def get_mi_net(config):
    code_size=config['code_size']
    mi_net = CLUBSample(x_dim=code_size, y_dim=code_size, hidden_size=256)
    return mi_net



def main(
    config,
    log_dir = './log/',
    ckpt_dir = './ckpt/',
    save_epoch = 2,
    synth_epoch = 2,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    image_root = os.path.join('./data/')
    image_size=28

    lr = 1e-2

    n_epoch = 100
    step_decay_weight = 0.95
    lr_decay_step = 2000
    weight_decay = 1e-6
    grad_clip_thresh = 1.0

    momentum = 0.9
    ##################################
    #  load model, setup optimizer   #
    ##################################

    my_net = get_model(config).to(device)
    mi_net = get_mi_net(config).to(device)
    criterion = get_loss_fn(config).to(device)

    optimizer = optim.SGD(my_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer_mi = optim.SGD(mi_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    logger = ExpLogger(log_dir)

    my_net.train()

    # print(my_net)
    log_exp(log_dir+'/../', my_net, config)


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

            optimizer_mi, current_lr = lr_scheduler_update(optimizer=optimizer_mi, step=current_step,
                init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight)

            data_target = data_iter.next()
            data_target = to_device(data_target, device=device)
            

            if config["model"] in ['ModelSV',]:
                my_net.set_step(current_step)

            lld_scalar = [0 for _ in range(config["mi_iters"])]
            for j in range(config["mi_iters"]):
                lld_loss = mi_first_forward(my_net, mi_net, optimizer_mi, data_target)
                lld_scalar[j] = lld_loss.item()
            # else: # config['use_mi'] = False
            #     lld_loss = torch.tensor(0.)
            # with open(log_dir+'/lld.txt', 'a') as logf:
            #     logf.write('Step: %d, Epoch: %d'% (current_step,  epoch) + str(lld_scalar) + '\n')
            # print(lld_scalar)
            
            loss, loss_dict, grad_norm = mi_second_forward(
                my_net, optimizer, criterion, mi_net, data_target, config,
                grad_clip_thresh=grad_clip_thresh)


            loss_dict = {
                k: v.item() for k, v in loss_dict.items()
            }
            loss_dict['lld_loss'] = lld_loss.item()
            logger.log_training(
                loss.item(), grad_norm.item(), current_lr, current_step,
                scalar_dict=loss_dict
            )

            current_step += 1

        print('Step: %d, Epoch: %d, total_loss: %.6f, ' % ( 
                current_step,
                epoch,
                loss.data.cpu().numpy(),
            ),
            ' '.join(['%s: %.6f' % (k,v) for k, v in loss_dict.items()]),
        )

        # print('step: %d, loss: %f' % (current_step, loss.cpu().data.numpy()))
        if epoch % save_epoch == 0:
            torch.save(my_net.state_dict(), ckpt_dir + '/sv_mnist_' + str(epoch) + '.pth')
            torch.save(mi_net.state_dict(), ckpt_dir + '/sv_minet_' + str(epoch) + '.pth')

        if epoch % synth_epoch == 0:
            test(my_net, criterion, mi_net, epoch, current_step, 
                name='mnist_m', logger=logger, log_dir=log_dir, mi_loss_weight=config['loss_weight_mi'])


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
    # parser.add_argument("--model", type=str, default="ModelED",
    #     required=False, help="model to train"
    # )
    parser.add_argument("--hparams", type=str, default="",
        required=False, help="yaml style dict to update config"
    )

    args = parser.parse_args()
    config = create_hparams(
        yaml_hparams_string=args.hparams,
        debug_print=True,
        allow_add=True,
    )

    print("LogDir:", args.log_dir)
    print("CheckPointDir:", args.ckpt_dir)
    print("Start Training.\n")
    main(config, args.log_dir, args.ckpt_dir)
    print('Done!')
    print('Logs are saved at:', args.log_dir)


