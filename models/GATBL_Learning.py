import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.layers import GraphAttentionLayer

class GATBL(nn.Module):
    def __init__(self, args):
        super(GATBL, self).__init__()
        self.args = args
        feat_dim = args.previous_hours * args.indicators_num
        gat_hidden_size = args.gat_hidden_size
        nheads = args.nheads
        alpha = args.alpha
        self.dropout = args.dropout
        nclass = args.city_num

        self.input_dropout = nn.Dropout(args.dropout)
        self.gat_dropout = nn.Dropout(args.dropout)
        
        self.attentions = [GraphAttentionLayer(feat_dim, gat_hidden_size, dropout=self.dropout, 
                                               alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(gat_hidden_size * nheads, nclass, dropout=self.dropout, alpha=alpha, concat=False)

        self.encoder = nn.LSTM(args.indicators_num, args.lstm_hidden_size, args.lstm_layers, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(args.lstm_hidden_size*2, args.lstm_hidden_size, args.lstm_layers, batch_first=True)

        self.fully_connect = nn.Linear(args.lstm_layers, 1)
    
    def forward(self, input_x, adj):
        # # -----------------------------------
        """ GAT Part """
        input_x = input_x.squeeze(0)
        input_x = input_x.transpose(0, 1).contiguous()
        input_x = input_x.view(self.args.city_num, -1)
        adj = adj.squeeze(0)
        x = self.input_dropout(input_x)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.gat_dropout(x)
        x = F.elu(self.out_att(x, adj))
        gat = F.log_softmax(x, dim=1)
        # # ------------------------------------

        # # ------------------------------------
        """ LSTM encoder-decoder Part """
        input_lstm = gat.unsqueeze(0)
        encoder_seq, _ = self.encoder(input_lstm)
        decoder_seq, (out_h, out_c) = self.decoder(encoder_seq)
        # # ------------------------------------

        # # ------------------------------------
        """ Fully Connect Part """
        input_tail = out_h.transpose(0, 2)
        output = self.fully_connect(input_tail).squeeze(2)
        # # ------------------------------------
        
        return output