import torch
import torch.nn as nn

from .transformer_encoder import TransformerEncoder as TE

"""
From paper HMCAN: hirarchical Multi-Modal Contextual Attention Network
plus the predictor
"""
class HMCAN(nn.Module):

    def __init__(self,
                 feature_size=384,
                 d_ff=512,
                 output_size=6,
                 k=2):
        super(HMCAN, self).__init__()
        # transformers
        self.transformer1 = Contextual_Transformer(d_model=feature_size, d_ff=d_ff)
        self.transformer2 = Contextual_Transformer(d_model=feature_size, d_ff=d_ff)

        # predictor
        self.fc = nn.Linear((1+k)*feature_size, output_size, bias=True)

    def forward(self, S, C):
        """
        inputs
        ------
        S: N * d, N: number of sentences
        C: T * d, T: number of comments
        """
        assert(len(S.shape) <= 4 and len(C.shape) <= 4)

        # for word-embedding (there are "max_token_num" tokens), take average
        if (len(S.shape) == 4):
            S = torch.mean(S, dim=-2)
        if (len(C.shape) == 4):
            C = torch.mean(C, dim=-2)

        C_TI = self.transformer1(S, C) # [B, 1, 384]
        S1 = C_TI.shape
        C_TI = C_TI.reshape(S1[0], S1[1] * S1[2])

        C_IT = self.transformer2(C, S) # [B, k, 1024]
        S2 = C_IT.shape
        C_IT = C_IT.reshape(S2[0], S2[1] * S2[2])

        concat_out = torch.concatenate((C_TI, C_IT), axis=-1)
        output = self.fc(concat_out)
        return output

class Contextual_Transformer(nn.Module):
    """
    From paper HMCAN: Contextual transformer network
    consists of a normal transformer encoder, and another encoder with K, V input value from first encoder
    """
    def __init__(self, 
                 d_model=1024, # size of learnable matrix in tanh(C^T * W * S)
                 d_ff=512
        ):
        super(Contextual_Transformer, self).__init__()
        # normal transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=1)
        self.single_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)

        # query input from input2 ; key, value from input 1
        self.co_encoder = TE(d_model=d_model, d_ff=d_ff)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
    def forward(self, S, C):
        """
        inputs
        ------
        S: N * d, N: number of sentences
        C: T * d, T: number of comments
        """
        encoder1_outputs = self.single_encoder(S) # [B, 1, 1024]
        pool1_outputs = self.pool1(encoder1_outputs) # [B, 1, 512]

        co_encoder_out = self.co_encoder(x=encoder1_outputs, x2=C, mask=None)
        pool2_outputs = self.pool2(co_encoder_out)
        
        outputs = torch.concatenate((pool1_outputs, pool2_outputs), axis=-1)
        return outputs
    

class FC_model(nn.Module):
    """
    simply a FC layer with concat input
    """
    def __init__(self,
                 feature_size=384,
                 output_size=6,
                 k=2):
        super(FC_model, self).__init__()
        # predictor
        self.fc = nn.Linear((1+k)*feature_size, output_size, bias=True)

    def forward(self, S, C):
        """
        inputs
        ------
        S: N * d, N: number of sentences
        C: T * d, T: number of comments
        """
        # if word-embedding, take average (for now)
        if (len(S.shape) == 4):
            S = torch.mean(S, dim=-2)
        if (len(C.shape) == 4):
            C = torch.mean(C, dim=-2)

        # and concat together (B, 1, 3*384)
        S = S.squeeze()
        C_shape = C.shape
        C = C.reshape(C_shape[0], C_shape[1]*C_shape[2])
        concat_out = torch.concat((S, C), dim=-1)
        
        output = self.fc(concat_out)
        output = output.squeeze()
        return output
    
if __name__ == '__main__':
    s = torch.rand(10, 1, 20, 1024)
    c = torch.rand(10, 2, 20, 1024)
    model = FC_model(feature_size=1024)
    ans = model(s, c)
    print(ans.shape)
    pass