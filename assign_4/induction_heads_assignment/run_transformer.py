import torch
import fire
from data import InductionData
import hconfig
from train_transformer import TransformerModel


def main(ckpt_file='transformer_model.ckpt', **kwargs):
    hps = hconfig.small
    hps.update(kwargs)

    evals = {}
    eval_batch = 50
    print('Generating eval data...', end='')
    for p in range(6, 12):
        l = 2**p
        eval_data = InductionData(eval_batch, hps.n_vocab, l)
        evals[p] = next(iter(eval_data))
    print('done.')

    xent = torch.nn.CrossEntropyLoss()
    model = TransformerModel(ntoken=hps.n_vocab + 1, ninp=hps.d_model, nhead=8, nff=2*hps.d_model,
                             nlayers=hps.n_layer, dropout=0.1).to('cuda')
    state = torch.load(ckpt_file)
    model.load_state_dict(state)
    print(f'Restored model from {ckpt_file}')

    eval_loss = 0
    total_correct = 0
    total_count = 0
    for p, eval_data in evals.items():
        tokens = eval_data['tokens'].to('cuda')
        out = model(tokens[:, :-1].transpose(0, 1))
        pred = out.transpose(0, 1)[:, -1, :]
        targ = tokens[:, -1]
        eval_loss += xent(pred, targ)
        predicted_tokens = pred.argmax(dim=-1)
        correct = (predicted_tokens == targ).sum().item()
        total_correct += correct
        total_count += targ.size(0)
        accuracy = correct / targ.size(0) if total_count > 0 else 0 # print for every p
        eval_loss /= len(evals) # print for every p
        print(f'p: {p}')
        print(f'Eval: {eval_loss.item():3.3f}')
        print(f'Accuracy: {accuracy:.3f}')
    eval_loss /= len(evals)
    accuracy = total_correct/ total_count if total_count > 0 else 0
    print(f'Overall Eval: {eval_loss.item():3.3f}')
    print(f'Overall Accuracy: {accuracy:.3f}')

if __name__ == '__main__':
    fire.Fire(main)
