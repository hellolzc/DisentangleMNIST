import torch
import torch.nn as nn
import torch.nn.functional as F


def to_categorical_torch(y, num_classes):
    """ 1-hot encodes a tensor """
    return F.one_hot(y, num_classes)

def binary_accuracy(Y_true, prediction):
    """calculate accuracy
    Y_true: (L,)
    prediction: (L,)
    """
    Y_true = torch.round(Y_true)
    prediction = torch.round(prediction)
    total_num = len(Y_true)
    correct = (Y_true == prediction.long()).sum()
    acc = correct / total_num
    return acc

def categorical_accuracy(Y_true, prediction):
    """calculate accuracy
    Y_true: (L, dim)
    prediction: (L, dim)
    """
    max_vals, max_indices = torch.max(prediction, dim=-1)
    _, Y_indices = torch.max(Y_true, dim=-1)
    total_num = len(Y_true)
    correct = (max_indices == Y_indices).sum(dtype=torch.float32)
    acc = correct / total_num

    return acc


class CLFLoss(nn.Module):
    """ EmbClassifier Loss """

    def __init__(self, config):
        super(CLFLoss, self).__init__()
        self.n_class = config['n_class']
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, targets, predictions):
        t_img, t_label = targets[:2]
        pred_label, pred_linear = predictions[:2]

        t_label_categorical = to_categorical_torch(t_label, self.n_class)

        ce_loss = self.bce_loss(pred_linear, t_label_categorical.float())
        accuracy = categorical_accuracy(t_label_categorical, pred_label)

        total_loss = (
            ce_loss + 0.0
        )

        return (
            total_loss,
            {
                'ce_loss': ce_loss,
                'accuracy': accuracy,
            },
        )


class EmbClassifier(nn.Module):
    '''
        This class provides the adversarial classifier to embeddings
        Method:
            forward() :      provides the estimation with input samples  
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(EmbClassifier, self).__init__()
        self.clf = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))


    def forward(self, x_samples, y_samples='unused'):
        y_linear = self.clf(x_samples)
        y = torch.softmax(y_linear, dim=1)
        return y, y_linear
