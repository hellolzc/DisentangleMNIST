import random
import os
import argparse
import torch

import torch.backends.cudnn as cudnn
import torch.optim as optim
from common.device_funcs import to_device
from common.hparams import create_hparams
from utils import prepare_dataloader, get_model, get_loss_fn, log_exp
from model.logger import ExpLogger
from model.adv_classifier import EmbClassifier, CLFLoss
from test_clf import test

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




def clf_forward(model, clf_net, clf_criterion, optimizer_clf_net, data_target, grad_clip_thresh=1.0):
    optimizer_clf_net.zero_grad()

    input_img, class_label = data_target

    with torch.no_grad():
        weights, scores, style_embs, text_embs, ref_embs = model.encode(input_data=input_img, number=class_label)

    x = style_embs.detach()
    y = class_label.detach()

    clf_output = clf_net(x)
    loss, loss_dict = clf_criterion(data_target, clf_output)

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(clf_net.parameters(), grad_clip_thresh)

    optimizer_clf_net.step()

    return loss, loss_dict, grad_norm


def get_clf_net(config):
    code_size=config['code_size']
    n_class=config['n_class']

    clf_net = EmbClassifier(x_dim=code_size, y_dim=n_class, hidden_size=256)
    return clf_net



def train_clf(
    encoder_model,
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

    my_net = encoder_model.to(device)
    clf_net = get_clf_net(config).to(device)
    clf_criterion = CLFLoss(config).to(device)

    # optimizer_clf = optim.SGD(clf_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer_clf = optim.Adam(clf_net.parameters(), lr=lr, weight_decay=weight_decay)

    logger = ExpLogger(log_dir)

    my_net.eval()

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
            optimizer_clf, current_lr = lr_scheduler_update(optimizer=optimizer_clf, step=current_step,
                init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight)

            data_target = data_iter.next()
            data_target = to_device(data_target, device=device)

            # TODO: fix this bug
            if config["model"] in ['ModelSV',]:
                assert False, 'Not support now'
                my_net.set_step(current_step)

            loss, loss_dict, grad_norm = clf_forward(my_net, clf_net, clf_criterion, optimizer_clf, data_target, 
                                                    grad_clip_thresh=grad_clip_thresh)

            loss_dict = {
                k: v.item() for k, v in loss_dict.items()
            }

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
            torch.save(clf_net.state_dict(), ckpt_dir + '/clf_minet_' + str(epoch) + '.pth')

        if epoch % synth_epoch == 0:
            test(my_net, clf_net, clf_criterion, epoch, current_step, 
                name='mnist_m', logger=logger, log_dir=log_dir)


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
    parser.add_argument("--load_model", type=str, default="exp/untitled/ckpt/sv_mnist_48.pth",
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
    print("Load Model...\n")
    my_net = get_model(config)
    checkpoint = torch.load(args.load_model)
    my_net.load_state_dict(checkpoint)
    my_net.eval()

    my_net = my_net.to(device)

    print("Start Training.\n")
    train_clf(my_net, config, args.log_dir, args.ckpt_dir)

    print('Done!')
    print('Logs are saved at:', args.log_dir)

