import os
import torch
import torch.backends.cudnn as cudnn


from common.device_funcs import to_device
from utils import prepare_val_dataloader, save_batch_results


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, criterion, mi_net, epoch, step, 
        use_mi=True,
        mi_loss_weight=1.0,
        name='MINIST-M',
        logger=None,
        log_dir = './log/',
    ):

    ###################
    # params          #
    ###################
    cuda = True
    cudnn.benchmark = True

    dataloader = prepare_val_dataloader()

    model.eval()

    len_dataloader = len(dataloader)
    data_iter = iter(dataloader)

    with torch.no_grad():
        n_total = 0
        total_loss_sum = 0.0
        loss_dict_sum = None
        for i in  range(len_dataloader):
            data_input = data_iter.next()
            data_input = to_device(data_input, device=device)
            input_img, class_label = data_input
            batch_size = len(class_label)

            result = model(input_data=input_img, number=class_label)
            loss, loss_dict = criterion(data_input, result)

            _, _, weights, scores, style_embs, text_embs = result
            mi_loss = mi_net.mi_est(style_embs, text_embs)
            loss_dict['mi_loss'] = mi_loss
            if use_mi:
                loss += mi_loss_weight * mi_loss

            total_loss_sum += loss * batch_size
            if loss_dict_sum is None:
                loss_dict_sum = {
                    k: v.item() * batch_size for k, v in loss_dict.items()
                }
            else:
                for k in loss_dict_sum:
                    loss_dict_sum[k] += loss_dict[k].item() * batch_size

            if i == len_dataloader - 2:
                save_batch_results(log_dir, 'Epoch_%d' % epoch, data_input, result)

            n_total += batch_size

    loss_mean = total_loss_sum / n_total
    loss_dict_mean = {
        k: v / n_total for k, v in loss_dict_sum.items()
    }
    print(
        'Step: %d, Epoch: %d, Validation on %d samples. total_loss: %.6f, ' % (step, epoch, n_total, loss_mean),
        ' '.join(['%s: %.6f' % (k,v) for k, v in loss_dict_mean.items()]),
    )

    if logger is not None:
        logger.log_validation(loss_mean, model, step, scalar_dict=loss_dict_mean)

    model.train()
