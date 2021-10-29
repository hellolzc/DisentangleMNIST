import os
import torch
import torch.backends.cudnn as cudnn


from common.device_funcs import to_device
from utils import prepare_val_dataloader, save_batch_results


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, criterion, epoch, step, 
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
        for i in  range(len_dataloader):
            data_input = data_iter.next()
            data_input = to_device(data_input, device=device)
            input_img, class_label = data_input

            batch_size = len(class_label)

            result = model(input_data=input_img, number=class_label)
            # ref_code, rec_img = result

            loss, target_mse = criterion(data_input, result)

            if i == len_dataloader - 2:
                save_batch_results(log_dir, 'Epoch_%d' % epoch, data_input, result)

            n_total += batch_size

    print('Validation Epoch: %d, Step %d, dataset: %s, total_loss: %f, rec_mse: %f' % (epoch, step, name, 
            loss.data.cpu().numpy(),
            target_mse.data.cpu().numpy(),
        )
    )

    if logger is not None:
        logger.log_validation(loss.data.cpu().numpy(), model, step)

    model.train()
