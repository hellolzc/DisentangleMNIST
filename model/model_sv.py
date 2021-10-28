import torch.nn as nn
from functions import ReverseLayerF




class RefEncoder(nn.Module):
    """refference encoder"""
    def __init__(self, code_size=100):
        super(RefEncoder, self).__init__()
        
        self.code_size = code_size
        ################################
        # refference encoder (dann_mnist)
        ################################
        self.ref_encoder_conv = nn.Sequential()
        self.ref_encoder_conv.add_module('conv_se1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,
                                                                  padding=2))
        self.ref_encoder_conv.add_module('ac_se1', nn.ReLU(True))
        self.ref_encoder_conv.add_module('pool_se1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.ref_encoder_conv.add_module('conv_se2', nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5,
                                                                  padding=2))
        self.ref_encoder_conv.add_module('ac_se2', nn.ReLU(True))
        self.ref_encoder_conv.add_module('pool_se2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.ref_encoder_fc = nn.Sequential()
        self.ref_encoder_fc.add_module('fc_se3', nn.Linear(in_features=7 * 7 * 48, out_features=code_size))
        self.ref_encoder_fc.add_module('ac_se3', nn.ReLU(True))

    def forward(self, input_data):
        ref_feat = self.ref_encoder_conv(input_data)
        ref_feat = ref_feat.view(-1, 48 * 7 * 7)
        ref_code = self.ref_encoder_fc(ref_feat)
        return ref_code



class Encoder(nn.Module):
    """ Encoder """

    def __init__(self,
        n_symbol=10,
        hidden_size=64,
        code_size=100,
    ):
        super(Encoder, self).__init__()

        self.src_word_emb = nn.Embedding(
            n_symbol, hidden_size
        )

        self.encoder_fc = nn.Sequential()
        self.encoder_fc.add_module('fc_sd1', nn.Linear(in_features=hidden_size, out_features=hidden_size))
        self.encoder_fc.add_module('relu_sd1', nn.ReLU(True))
        self.encoder_fc.add_module('fc_sd2', nn.Linear(in_features=hidden_size, out_features=code_size))
        self.encoder_fc.add_module('relu_sd2', nn.ReLU(True))

    def forward(self, src_seq):
        batch_size = src_seq.shape[0]

        # -- Forward
        enc_output = self.src_word_emb(src_seq)

        enc_output = self.encoder_fc(enc_output)

        return enc_output



class Decoder(nn.Module):
    """decoder"""
    def __init__(self, code_size=100):
        super(Decoder, self).__init__()

        self.decoder_fc = nn.Sequential()
        self.decoder_fc.add_module('fc_sd1', nn.Linear(in_features=code_size, out_features=588))
        self.decoder_fc.add_module('relu_sd1', nn.ReLU(True))

        self.decoder_conv = nn.Sequential()
        self.decoder_conv.add_module('conv_sd2', nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5,
                                                                  padding=2))
        self.decoder_conv.add_module('relu_sd2', nn.ReLU())

        self.decoder_conv.add_module('conv_sd3', nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5,
                                                                  padding=2))
        self.decoder_conv.add_module('relu_sd3', nn.ReLU())

        self.decoder_conv.add_module('us_sd4', nn.Upsample(scale_factor=2))

        self.decoder_conv.add_module('conv_sd5', nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                                                                  padding=1))
        self.decoder_conv.add_module('relu_sd5', nn.ReLU(True))

        self.decoder_conv.add_module('conv_sd6', nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3,
                                                                  padding=1))

    def forward(self, union_code):
        rec_vec = self.decoder_fc(union_code)
        rec_vec = rec_vec.view(-1, 3, 14, 14)
        rec_img = self.decoder_conv(rec_vec)
        return rec_img



class ModelED(nn.Module):
    def __init__(self, config):
        super(ModelED, self).__init__()
        code_size= config['code_size']  # 100, 
        n_class  = config['n_class']  # 10

        self.ref_encoder = RefEncoder(code_size)
        self.decoder = Decoder(code_size)


    def forward(self, input_data=None, number='unused'):
        # ref encoder
        ref_code = self.ref_encoder(input_data)
        # decoder
        union_code = ref_code
        rec_img = self.decoder(union_code)

        return ref_code, rec_img


class ModelNTI(nn.Module):
    """Number to image"""
    def __init__(self, config):
        super(ModelNTI, self).__init__()
        code_size= config['code_size']  # 100, 
        n_class  = config['n_class']  # 10

        self.encoder = Encoder(n_class, 64, code_size)
        self.decoder = Decoder(code_size)

    def forward(self, input_data='unused', number=None):
        # encoder
        emb_code = self.encoder(number)
        # decoder
        union_code = emb_code
        rec_img = self.decoder(union_code)

        return emb_code, rec_img


