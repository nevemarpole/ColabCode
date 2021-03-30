"""
Created on Tue Mar 23 18:24:59 2021

@author: nevem
"""

import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt


def prepareData(dataFrame):
    sentences = dataFrame.prompt.values
    
    i = 0
    for this in sentences:
        sentences[i] = str(sentences[i])
        i = i + 1

    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]

    return(sentences)




def prepareLabels(dataFrame):
    dataFrame['context'].replace({"surprised": "0", "excited": "1", "angry": "2", "proud": "3", 
                             "sad": "4", "annoyed": "5", "grateful": "6", "lonely": "7", 
                             "afraid": "8", "terrified": "9", "guilty": "10", "impressed": "11",
                             "disgusted": "12", "hopeful": "13", "confident": "14", 
                             "furious": "15", "anxious": "16", "anticipating": "17",
                             "joyful": "18", "nostalgic": "19", "disappointed": "20",
                             "prepared": "21", "jealous": "22", "content": "23",
                             "devastated": "24", "embarrassed": "25", "caring": "26",
                             "sentimental": "27", "trusting": "28", "ashamed": "29",
                             "apprehensive": "30", "faithful": "31",}, inplace=True)
    
    labels = dataFrame.context.values
    labels = np.array(labels, dtype='float32')
    
    return(labels)




def tokenizeData(data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    MAX_LEN = 128
    
    tokenized = [tokenizer.tokenize(section) for section in data]
    
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    return(input_ids)
        



def applyMasks(input_ids):
    attention_masks = []
    
    for seq in input_ids_train:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    
    return(attention_masks)
    
  
    

# Function to calculate the accuracy of predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
    



df_train = pd.read_csv("train.csv", delimiter=',', usecols=('conv_id', 'utterance_idx', 'context', 'prompt', 'speaker_idx', 'utterance')) 
#df_valid = pd.read_csv("valid.csv", delimiter=',', usecols=('conv_id', 'utterance_idx', 'context', 'prompt', 'speaker_idx', 'utterance')) 
#df_test = pd.read_csv("test.csv", delimiter=',', usecols=('conv_id', 'utterance_idx', 'context', 'prompt', 'speaker_idx', 'utterance'))

print("Data read in")


train_data = prepareData(df_train)
#valid_data = prepareData(df_valid)
#test_data = prepareData(df_test)

print("Data prepared")


train_labels = prepareLabels(df_train)
#valid_labels = prepareLabels(df_valid)
#test_labels = prepareLabels(df_test)

print("Labels prepared")


input_ids_train = tokenizeData(train_data)
#input_ids_valid = tokenizeData(valid_data)
#input_ids_test = tokenizeData(test_data)

print("Input ID's configured")


attention_masks_train = applyMasks(input_ids_train)
#attention_masks_valid = applyMasks(input_ids_valid)
#attention_masks_test = applyMasks(input_ids_test)

print("Masks applied")

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids_train, train_labels, 
                                                            random_state=2018, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks_train, input_ids_train,
                                             random_state=2018, test_size=0.1)

train_labels = np.array(train_labels, dtype='float32')
validation_labels = np.array(validation_labels, dtype='float32')

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

print("Tensors created")

# Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
batch_size = 32

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)



model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=32)
model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

# This variable contains all of the hyperparemeter information our training loop needs
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=.1)



##BIG CODE BLOCK##

t = [] 

# Store our loss and accuracy for plotting
train_loss_set = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 4

# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc="Epoch"):
  
  
  # Training
  
  # Set our model to training mode (as opposed to evaluation mode)
  model.train()
  
  # Tracking variables
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0
  
  # Train the data for one epoch
  for step, batch in enumerate(train_dataloader):
    # Add batch to GPU
    batch = tuple(t.to(dtype=torch.int64) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Clear out the gradients (by default they accumulate)
    optimizer.zero_grad()
    # Forward pass
    loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
    train_loss_set.append(loss.item())    
    # Backward pass
    loss.backward()
    # Update parameters and take a step using the computed gradient
    optimizer.step()
    
    
    # Update tracking variables
    tr_loss += loss.item()
    nb_tr_examples += b_input_ids.size(0)
    nb_tr_steps += 1

  print("Train loss: {}".format(tr_loss/nb_tr_steps))
    
    
  # Validation

  # Put model in evaluation mode to evaluate loss on the validation set
  model.eval()

  # Tracking variables 
  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0

  # Evaluate data for one epoch
  for batch in validation_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(dtype=torch.int64) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up validation
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

  print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))


