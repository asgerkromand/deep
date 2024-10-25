# InductionHeads Experiment

The code for data creation is based on [this repository](https://github.com/hrbigelow/mamba-recall).

## Setup

To run the experiment, use the following commands:
cd induction_heads_assignment  
python train_transformer.py --n_epoch=25  
python run_transformer.py  

## Training

The script `train_transformer.py` will train a Transformer model for 25 epochs with `--n_epoch=25`.

You can modify the code in `train_transformer.py` to stop training when a certain accuracy, such as 99%, is achieved. This allows for early stopping based on performance rather than a fixed number of epochs.

## Inference

After training, the script `run_transformer.py` will load the pretrained checkpoint and run inference on sequences of different lengths. The lengths of the input sequences will vary from 2^6 to 2^12. The script will then output the model's performance for each sequence length.

## Additional Notes

- Analyze the results to see how well the model generalizes across different input sequence lengths.

