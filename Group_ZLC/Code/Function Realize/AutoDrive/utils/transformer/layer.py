from torch import nn

from utils.transformer.layer_norm import LayerNorm
from utils.transformer.multi_head_attention import MultiHeadAttention
from utils.transformer.position_wise_feed_forward import PositionwiseFeedForward


class Layer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, device):
        super(Layer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head).to(device)
        self.norm1 = LayerNorm(d_model=d_model).to(device)
        self.dropout1 = nn.Dropout(p=drop_prob).to(device)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob).to(device)
        self.norm2 = LayerNorm(d_model=d_model).to(device)
        self.dropout2 = nn.Dropout(p=drop_prob).to(device)

    def forward(self, dec, mask):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x