import torch
import torch.nn as nn
import math

class self_attn(nn.Module):
    """
    known as scaled dot-product mechanism
    utilized three weight matrix for Key, Value, Query
    """
    def __init__(self, 
                 input_dim=20, 
                 q_dim=20, # dimension of Q or K
                 use_multi=False, num_heads=4):
        super(self_attn, self).__init__()
        self.input_dim = input_dim
        self.use_multi = use_multi
        self.d = q_dim

        if (self.use_multi):
            self.multi = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        else:
            self.query = nn.Linear(input_dim, q_dim) # [B, sentence_len, token (input_dim), BERT_dim]
            self.key = nn.Linear(input_dim, q_dim)
            self.value = nn.Linear(input_dim, q_dim)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        input shape be like [2, 20, 1024] 
        we want to do the second dim
        """
        # merge batch size with sentence len
        x_shape = x.shape

        if (self.use_multi):
            weighted, _ = self.multi(x, x, x)
            # now weight.shape = [B, seq_length, input_dim]
            weighted = weighted.sum(dim=-1) # [2*B, 1024]
            weighted = weighted.reshape(x_shape[0], x_shape[1], x_shape[2]) # [B, 2, 1024]
            return weighted

        else:
            Q = self.query(x)
            K = self.key(x)
            V = self.value(x)

            ## TODO: there should be an attention mask
            scores = torch.matmul(Q, K.transpose(-1, -2))/math.sqrt(self.d)
            attention = self.softmax(scores)
            # weight_score
            weighted = torch.matmul(attention, V) # [B*sent_length, max_token, model_dim]
            return {'atten_prob': attention, 'atten_score': weighted}
        
class AttNetwork(nn.Module):
    """
    (for word embedding only)
    Measure the importance level between word embedding
    u_t = tanh(Wc * h_t + bc)
    B_t = exp(u_t * uc) / \sum_k exp(u_k * uc)
    """
    def __init__(self,
                 model_dim=1024, # e.g. [2, 20, 1024] 
                 attention_dim=32):
        
        super(AttNetwork, self).__init__()
        self.attention_dim = attention_dim
        # uit = tanh(W^T * h_t + b)
        self.linear_u = nn.Linear(model_dim, attention_dim, bias=True)
        # beta = exp(uit * u)
        self.u = nn.Parameter(torch.rand(self.attention_dim, 1))
        self.epsilon = 1e-10

    def forward(self, S):
        uit = torch.tanh(self.linear_u(S)) # [B, word_token, atten_dim]
        ait = torch.einsum('Bbwa,al->Bbwl', uit, self.u) # [B, word_token, 1]
        ait = torch.squeeze(ait, dim=-1) # [B, word_token]

        # Torch: https://discuss.pytorch.org/t/in-place-operation-error-in-backward-call/171058
        ait = torch.exp(ait).clone()

        ait /= torch.sum(ait, dim=2, keepdim=True) + self.epsilon
        
        # beta_t * h_t (model_dim)
        # (2, 5, 1) * (2, 5, 3) --> (2, 5, 3)
        # scalar product to last axis
        ait = torch.unsqueeze(ait, dim=-1)
        weighted_input = S * ait
        output = torch.sum(weighted_input, dim=2)

        return output

class CoAttentionNetwork(nn.Module):
    """
    dEFEND paper implementation:
    
    Co-attention layer which accept content and comment states and computes co-attention between them and returns the
     weighted sum of the content and the comment states

    ---
    
    argument
    - setting: experiement settings in dEFEND paper, including
        - defend_co: self-attention on sentences and comments instead of Co-Attention   
    """
    def __init__(self, 
                 output_size=6,
                 feature_size=384, # size of learnable matrix in tanh(C^T * W * S)
                 k=2, # top-k
                 word_attn=False, # add wird-attention network before CoAttention network or not
                 affinity='expo', # 'expo', 'cosine' or 'gaussian'
                 setting=None, self_attn_input_dim=20,   # experiment settings
        ):
        super(CoAttentionNetwork, self).__init__()

        self.output_size = output_size
        self.feature_size = feature_size
        self.k = k
        self.word_attn = word_attn
        self.affinity = affinity
        self.setting = setting

        self.no_co_mode = "defend_co"

        # refer to dEFEND method
        self.Wl = nn.Parameter(torch.rand(feature_size, feature_size))
        # weight construct 2D attetion network weights, along with affinity matrix
        self.Ws = nn.Parameter(torch.rand(self.k, feature_size))
        self.Wc = nn.Parameter(torch.rand(self.k, feature_size))
        # weight construct the attetion weight
        self.Whs = nn.Parameter(torch.rand(1, self.k))
        self.Whc = nn.Parameter(torch.rand(1, self.k))

        # weight to construct predicted softmax output: softmax([s_, c_] * Wf + bf)
        self.fc = nn.Linear(
            in_features = 2 * self.feature_size, 
            out_features = self.output_size, 
            bias=True
        )


        # word-attention network
        self.S_attn = None
        self.C_attn = None

        # use exponential-based attention network
        if (word_attn):
            self.S_attn = AttNetwork(model_dim=feature_size)
            self.C_attn = AttNetwork(model_dim=feature_size)

        if (setting == self.no_co_mode):
            self.S_self_attn = self_attn(input_dim=self_attn_input_dim, q_dim=1)
            self.C_self_attn = self_attn(input_dim=self_attn_input_dim, q_dim=1)
            
            # fc-layer for other setting
            self.fc_no_co = nn.Linear(
                in_features = (1 + self.k) * self.feature_size,
                out_features = self.output_size  
            )

    def forward(self, S, C):
        """
        inputs
        ------
        S: N * d, N: number of sentences
        C: T * d, T: number of comments
        """
        if (self.setting == self.no_co_mode):
            ## TODO: self_attn instead of co-attention
            S_shape = S.shape
            S = S.transpose(-1, -2)
            S = self.S_self_attn(S)["atten_score"] # [B*sent, 1, model_dim]
            S = S.transpose(-1, -2)
            S = S.reshape(S_shape[0], S_shape[1] * S_shape[-1])

            C_shape = C.shape
            C = C.transpose(-1, -2)
            C = self.C_self_attn(C)["atten_score"]
            C = C.transpose(-1, -2)
            C = C.reshape(C_shape[0], C_shape[1] * C_shape[-1])

            sc = torch.cat((S, C), dim=1)

            # Linear with no softmax (due to nn.CE loss)
            outputs = self.fc_no_co(sc)
            return outputs


        if (self.word_attn):
            S = self.S_attn(S)
            C = self.C_attn(C)

        S = S.transpose(1, 2)
        C = C.transpose(1, 2)

        # affinity matrix
        if (self.affinity == "consine"):
            F = cosine_affinity(C.transpose(1, 2), S.transpose(1, 2))
        elif (self.affinity == "gaussian"):
            F = gaussian_affinty(C.transpose(1, 2), S.transpose(1, 2))
        else:
            # tanh(C^T * W * S) --> (B, C.sent_len, S.sent_len)
            F = torch.tanh(torch.einsum("btd,dD,bDn->btn", C.transpose(1, 2), self.Wl, S))

        # Hs = tanh(WsS + (WcC)F)
        Hs = torch.tanh(
            torch.einsum('kd,bdn->bkn', self.Ws, S) + 
            torch.einsum('kd,bdt,btn->bkn', self.Wc, C, F)
        )
        
        # Hc = tanh(WcC + (WsS)F^T)
        Hc = torch.tanh(
            torch.einsum('kd,bdt->bkt', self.Wc, C) +
            torch.einsum('kd,bdn,bnt->bkt', self.Ws, S, F.transpose(1, 2))
        )

        # As = softmax(Whs * Hs) --> 1 * N
        As = torch.nn.functional.softmax(
            torch.einsum('yk,bkn->bn', self.Whs, Hs),
            dim=-1
        )

        # Ac = softmax(Whc * Hc)
        Ac = torch.nn.functional.softmax(
            torch.einsum('yk,bkt->bt', self.Whc, Hc),
            dim=-1
        )

        # comment feature and sentence feature
        # s_ : 1 * d
        co_s = torch.einsum('bdn,bn->bd', S, As)
        co_c = torch.einsum('bdt,bt->bd', C, Ac)

        # softmax([s_, c_] * Wf + bf)
        co_sc = torch.cat((co_s, co_c), dim=1)

        # Linear with no softmax (due to nn.CE loss)
        outputs = self.fc(co_sc)
        # outputs = torch.nn.functional.softmax(outputs, dim=-1)

        return outputs
    
def gaussian_affinty(x, y, gamma=0.001):
    # broadcasting
    x = x.unsqueeze(1)
    y = y.unsqueeze(2)

    # matrix operation
    s = torch.sum((x - y) ** 2, dim=-1)
    s = s.transpose(-2, -1) # [B, X_sentlen, Y_sentlen]
    return torch.exp(-gamma * s)

def cosine_affinity(s1, s2):
    # dot product btwn s1, s2
    dot_product = torch.matmul(s1, s2.transpose(-2, -1)) # (2, 3, 2)

    # norm of s1, s2
    s1_n = torch.linalg.norm(s1, dim=-1, keepdim=True) # (B, sent, 1)
    s2_n = torch.linalg.norm(s2, dim=-1, keepdim=True) 

    cos_sim = dot_product / (s1_n * s2_n.transpose(-2, -1))
    return cos_sim # (B, s1.sent_len, s2.sent_len)

if __name__ == '__main__':
    # model = CoAttentionNetwork(feature_size=1024, setting="defend_co", word_attn=True)
    s1 = torch.rand(2, 1, 20, 1024)
    s2 = torch.rand(2, 2, 20, 1024)
    model = self_attn(input_dim=20, q_dim=1)

    s = model(s1.transpose(-1, -2))
    print(s["atten_score"].shape)
