import math

import dgl.function as fn
import torch
from torch import nn


class CompGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout, bias=True):
        super(CompGCNLayer, self).__init__()
        self.weight_msg = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_msg.size(1))
        self.weight_msg.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def msg_func(self, edges):
        rel_ids = edges.data['rel']
        rel_embs = self.r[rel_ids]
        node_embs = edges.src['h']
        msg = node_embs - rel_embs
        msg = torch.mm(msg, self.weight_msg)
        # rel_type = edges.data['type_rel']
        # type_weight = self.weight_msg[rel_type]
        # msg = torch.bmm(msg.unsqueeze(1), type_weight).squeeze()
        return {'msg': msg}
            
    def forward(self, h, r, g):
        self.r = r
        if self.dropout:
            h = self.dropout(h)
        # h = torch.mm(h, self.weight)
        h = h * g.ndata['norm']
        g.ndata['h'] = h
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h0'))
        agg_msg = g.ndata.pop('h0') * g.ndata['norm']
        h = torch.mm(g.ndata.pop('h'), self.weight)
        h = h + agg_msg
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h

class NewRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, need_neighbor_weight=True,
                 need_loop_weight=True, need_skip_weight=True, bias=None, activation=None,
                 self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
        super(NewRGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.need_neighbor_weight = need_neighbor_weight
        self.need_loop_weight = need_loop_weight
        self.need_skip_weight = need_skip_weight
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.rel_emb = None

        if need_neighbor_weight:
            self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        else:
            self.register_parameter("weight_neighbor", None)

        if self_loop and need_loop_weight:  # 有 self_loop 并且 需要权重
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        elif self_loop and need_loop_weight is False:  # 有 self_loop 但是 会自己传权重进来
            self.register_parameter("loop_weight", None)
            self.register_parameter("evolve_loop_weight", None)
        else:
            pass

        if skip_connect and need_skip_weight:  # 有 skip_connect 并且需要权重
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))  # 和self-loop不一样，是跨层的计算
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
        elif skip_connect and need_skip_weight is False:  # 有 skip_connect 但是会自己传权重进来
            self.register_parameter("skip_connect_weight", None)
            self.register_parameter("skip_connect_bias", None)
        else:
            pass

        self.reset_parameters()

        if dropout:
            self.dropout = nn.Dropout(dropout)  # 没有可训练的参数
        else:
            self.dropout = None

    def reset_parameters(self):
        r"""
        Reinitilize learnable parameters
        """
        if self.weight_neighbor is not None:
            nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))
        if self.self_loop and self.loop_weight is not None:
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))
        if self.skip_connect and self.skip_connect_weight is not None:
            nn.init.xavier_uniform_(self.skip_connect_weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

    def propagate(self, g, weight_neighbor):
        g.update_all(lambda x: self.msg_func(x, weight_neighbor), fn.sum(msg='msg', out='h'), self.apply_func)

    def forward(self, g, prev_h, emb_rel, weight_neighbor=None, loop_weight=None, evolve_loop_weight=None,
                skip_connect_weight=None, skip_connect_bias=None):  # g: 当前历史子图; []; self.h_0 边的嵌入 (num_rels*2, h_dim)
        if self.need_neighbor_weight:  # 模型初始化了参数
            if weight_neighbor is not None:
                raise NotImplementedError
            else:
                weight_neighbor = self.weight_neighbor
        else:  # 模型未初始化参数，需要自己传入
            if weight_neighbor is None:
                raise NotImplementedError

        if self.self_loop:
            if self.need_loop_weight:  # 模型初始化了参数
                if loop_weight is not None or evolve_loop_weight is not None:
                    raise NotImplementedError
                else:
                    loop_weight = self.loop_weight
                    evolve_loop_weight = self.evolve_loop_weight
            else:
                if loop_weight is None or evolve_loop_weight is None:
                    raise NotImplementedError

        if self.skip_connect:
            if self.need_skip_weight:  # 模型初始化了参数
                if skip_connect_weight is not None or skip_connect_bias is not None:
                    raise NotImplementedError
                else:
                    skip_connect_weight = self.skip_connect_weight
                    skip_connect_bias = self.skip_connect_bias
            else:
                if skip_connect_weight is None or skip_connect_bias is None:
                    raise NotImplementedError

        self.rel_emb = emb_rel
        if self.self_loop:
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),  # 全体node编号中筛选
                (g.in_degrees(range(g.number_of_nodes())) > 0))  # 筛选当前历史子图中入度不为0的所有node节点编号，返回一维张量
            loop_message = torch.mm(g.ndata['h'],
                                    evolve_loop_weight)  # g.ndata['h']: node embedding (g_num_nodes, h_dim) (h_dim. h_dim)
            loop_message[masked_index, :] = torch.mm(g.ndata['h'], loop_weight)[masked_index,
                                            :]  # 更新loop_message中入度不为0的node节点
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, skip_connect_weight) + skip_connect_bias)  # 使用sigmoid，让值在0~1

        # calculate the neighbor message with weight_neighbor
        self.propagate(g, weight_neighbor)
        node_repr = g.ndata['h']  # node embedding

        # print(len(prev_h))
        if len(prev_h) != 0 and self.skip_connect:  # 两次计算loop_message的方式不一样，前者激活后再加权
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation:  # 激活函数
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr  # 返回的是更新的节点表示g.ndata['h']

    def msg_func(self, edges, weight_neighbor):
        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        edge_type = edges.data['type']
        edge_num = edge_type.shape[0]
        node = edges.src['h'].view(-1, self.out_feat)

        msg = node + relation
        msg = torch.mm(msg, weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}
