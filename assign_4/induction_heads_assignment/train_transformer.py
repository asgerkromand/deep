from typing import Optional

import fire
import torch
import torch as t
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import dtype, Tensor
from torch.optim import Adam
from data import InductionData
import hconfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=t.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()* (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Transformer):
    def __init__(self, ntoken, ninp, nhead, nff, nlayers, dropout=0.1):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nff, num_encoder_layers=nlayers)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return output


def main(n_epoch=1000, epoch_sz=8192, report_every=100, out_file='transformer_model.ckpt'):
    hps = hconfig.small
    data = InductionData(hps.batch, hps.n_vocab, hps.train_len)

    # Initialize transformer model
    model = TransformerModel(ntoken=hps.n_vocab + 1, ninp=hps.d_model, nhead=8, nff=2 * hps.d_model,
                             nlayers=hps.n_layer, dropout=0.1).to('cuda')

    it = iter(data)
    xent = t.nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=hps.learn_rate)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    step = 0
    for epoch in range(n_epoch):
        loss_sum = 0
        total_correct = 0
        total_count = 0
        for b in range(epoch_sz):
            batch = next(it)
            tokens = batch['tokens'].to('cuda')
            opt.zero_grad()
            out = model(tokens[:, :-1].transpose(0,1))  # Forward pass through transformer
            pred = out.transpose(0,1)[:, -1, :]
            targ = tokens[:, -1]
            loss = xent(pred, targ)
            loss.backward()
            loss_sum += loss
            opt.step()

            predicted_tokens = pred.argmax(dim=-1)
            correct = (predicted_tokens == targ).sum().item()
            total_correct += correct
            total_count += targ.size(0)
            step += 1
            if step % report_every == 0:
                accuracy = total_correct / total_count if total_count > 0 else 0
                print(f'{epoch=}, {step=}, {loss.item():3.3f}, Accuracy: {accuracy:.3f}')

    state = model.state_dict()
    print(f'Saving to {out_file}')
    t.save(state, out_file)


if __name__ == '__main__':
    fire.Fire(main)
