import os
import torch
import torch.backends.cudnn as cudnn


from common.device_funcs import to_device
from utils import prepare_val_dataloader, save_batch_results


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def clf_test_forward(model, clf_net, clf_criterion, data_target):

    input_img, class_label = data_target

    with torch.no_grad():
        weights, scores, style_embs, text_embs = model.encode(input_data=input_img, number=class_label)

    x = style_embs.detach()
    y = class_label.detach()

    clf_output = clf_net(x)
    loss, loss_dict = clf_criterion(data_target, clf_output)

    return clf_output, loss, loss_dict


def test(encoder_model, clf_net, clf_criterion, epoch, step, 
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

    clf_net.eval()

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

            result, loss, loss_dict = clf_test_forward(encoder_model, clf_net, clf_criterion, data_input)

            # y, y_linear = result

            # Sum Loss
            total_loss_sum += loss * batch_size
            if loss_dict_sum is None:
                loss_dict_sum = {
                    k: v.item() * batch_size for k, v in loss_dict.items()
                }
            else:
                for k in loss_dict_sum:
                    loss_dict_sum[k] += loss_dict[k].item() * batch_size

            #if i == len_dataloader - 2:
            #    save_batch_results(log_dir, 'Epoch_%d' % epoch, data_input, result)

            n_total += batch_size

    # Calcuate Loss mean
    loss_mean = total_loss_sum / n_total
    loss_dict_mean = {
        k: v / n_total for k, v in loss_dict_sum.items()
    }
    print(
        'Step: %d, Epoch: %d, Validation on %d samples. total_loss: %.6f, ' % (step, epoch, n_total, loss_mean),
        ' '.join(['%s: %.6f' % (k,v) for k, v in loss_dict_mean.items()]),
    )

    if logger is not None:
        logger.log_validation(loss_mean, clf_net, step, scalar_dict=loss_dict_mean)

    clf_net.train()
