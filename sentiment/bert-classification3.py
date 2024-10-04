"""
A basic classifier based on the transformers
(https://github.com/huggingface/transformers) library. It loads a masked
language model (by default bert-base-cased), and adds a linear layer for
prediction. Needs the path to a training file and a development file. Example
usage:

python3 bert-classification.py train sentiment/sst.train
python3 bert-classification.py evaluate path/to/test/file
"""
from typing import List, Dict
import torch
import sys
import myutils
from transformers import AutoModel, AutoTokenizer
import os

# Set some constants
torch.manual_seed(8446)
MLM = 'bert-base-cased'
BATCH_SIZE = 16
LEARNING_RATE = 0.000005
EPOCHS = 5
PAD_TOKEN = "[PAD]"
MAX_LEN = 128
MAX_TRAIN_SENTS=10000
# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = 'sentiment_model.pth'

class ClassModel(torch.nn.Module):
    def __init__(self, nlabels: int, mlm: str):
        """
        Model for classification with transformers.

        The architecture of this model is simple, we just have a transformer
        based language model, and add one linear layer to converts it output
        to our prediction.
    
        Parameters
        ----------
        nlabels : int
            Vocabulary size of output space (i.e. number of labels)
        mlm : str
            Name of the transformers language model to use, can be found on:
            https://huggingface.co/models
        """
        super().__init__()

        # The transformer model to use
        self.mlm = AutoModel.from_pretrained(mlm)

        # Find the size of the output of the masked language model
        if hasattr(self.mlm.config, 'hidden_size'):
            self.mlm_out_size = self.mlm.config.hidden_size
        elif hasattr(self.mlm.config, 'dim'):
            self.mlm_out_size = self.mlm.config.dim
        else: # if not found, guess
            self.mlm_out_size = 768

        # Create prediction layer
        self.hidden_to_label = torch.nn.Linear(self.mlm_out_size, nlabels)

    def forward(self, input: torch.tensor, attention_mask: torch.tensor):
        """
        Forward pass
    
        Parameters
        ----------
        input : torch.tensor
            Tensor with wordpiece indices. shape=(batch_size, max_sent_len).
        mask : torch.tensor
            Mask corresponding to input. shape=(batch_size, max_sent_len)
        Returns
        -------
        output_scores : torch.tensor
            ?. shape=(?,?)
        """
        # Run transformer model on input
        mlm_out = self.mlm(input, attention_mask=attention_mask)

        # Keep only the last layer: shape=(batch_size, max_len, DIM_EMBEDDING)
        mlm_out = mlm_out.last_hidden_state
        # Keep only the output for the first ([CLS]) token: shape=(batch_size, DIM_EMBEDDING)
        mlm_out = mlm_out[:,:1,:].squeeze()

        # Matrix multiply to get scores for each label: shape=(batch_size, num_labels)
        output_scores = self.hidden_to_label(mlm_out)

        return output_scores

    def run_eval(self, text_batched: List[torch.tensor], text_batched_mask: List[torch.tensor], labels_batched: List[torch.tensor]):
        """
        Run evaluation: predict and score
    
        Parameters
        ----------
        text_batched : List[torch.tensor]
            list with batches of text, containing wordpiece indices.
        text_batched_mask : List[torch.tensor]
            mask corresponding to text_batched
        labels_batched : List[torch.tensor]
            list with batches of labels (converted to ints).
        model : torch.nn.module
            The model to use for prediction.
    
        Returns
        -------
        score : float
            accuracy of model on labels_batches given feats_batches
        """
        self.eval()
        match = 0
        total = 0
        for sents, mask, labels in zip(text_batched, text_batched_mask, labels_batched):
            output_scores = self.forward(sents, mask)
            pred_labels = torch.argmax(output_scores, 1)
            for gold_label, pred_label in zip(labels, pred_labels):
                total += 1
                if gold_label.item() == pred_label.item():
                    match+= 1
        return(match/total)

def train(train_file):
    print('reading data...')
    train_text, train_labels = myutils.read_data(train_file)
    train_text = train_text[:MAX_TRAIN_SENTS]
    train_labels = train_labels[:MAX_TRAIN_SENTS]
    
    id2label, label2id = myutils.labels2lookup(train_labels, PAD_TOKEN)
    NLABELS = len(id2label)
    train_labels = [label2id[label] for label in train_labels]
    
    print('tokenizing...')
    tokzr = AutoTokenizer.from_pretrained(MLM, clean_up_tokenization_spaces=True)
    train_tokked = myutils.tok(train_text, tokzr, MAX_LEN)
    PAD_ID = tokzr.pad_token_id
    
    print('converting to batches...')
    train_text_batched, train_text_batched_mask, train_labels_batched = myutils.to_batch(train_tokked, train_labels, BATCH_SIZE, PAD_ID, DEVICE)
    
    print('initializing model...')
    model = ClassModel(NLABELS, MLM)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    
    print('training...')
    for epoch in range(EPOCHS):
        print('=====================')
        print('starting epoch ' + str(epoch))
        model.train() 
    
        loss = 0
        for batch_idx in range(0, len(train_text_batched)):
            optimizer.zero_grad()
            output_scores = model.forward(train_text_batched[batch_idx], attention_mask=train_text_batched_mask[batch_idx])
            batch_loss = loss_function(output_scores, train_labels_batched[batch_idx])
            loss += batch_loss.item()
    
            batch_loss.backward()

            optimizer.step()
    
        print('Loss: {:.2f}'.format(loss))
        print()

    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'id2label': id2label,
        'label2id': label2id,
    }, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')

def evaluate(test_file):
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found. Please train the model first.")
        return

    # Load the saved model
    checkpoint = torch.load(MODEL_PATH)
    id2label = checkpoint['id2label']
    label2id = checkpoint['label2id']
    NLABELS = len(id2label)

    model = ClassModel(NLABELS, MLM)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    print('reading test data...')
    test_text, test_labels = myutils.read_data(test_file)
    test_labels = [label2id[label] for label in test_labels]

    print('tokenizing...')
    tokzr = AutoTokenizer.from_pretrained(MLM, clean_up_tokenization_spaces=True)
    test_tokked = myutils.tok(test_text, tokzr, MAX_LEN)
    PAD_ID = tokzr.pad_token_id

    print('converting to batches...')
    test_text_batched, test_text_batched_mask, test_labels_batched = myutils.to_batch(test_tokked, test_labels, BATCH_SIZE, PAD_ID, DEVICE)

    test_score = model.run_eval(test_text_batched, test_text_batched_mask, test_labels_batched)
    print('Acc(test): {:.2f}'.format(100*test_score))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 bert-classification.py [train|evaluate] [file_path]')
    elif sys.argv[1] == 'train':
        train(sys.argv[2])
    elif sys.argv[1] == 'evaluate':
        evaluate(sys.argv[2])
    else:
        print('Invalid command. Use either "train" or "evaluate".')
