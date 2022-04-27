import functools
import sys

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
from model import LSTM
import time
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

seed = 0

torch.manual_seed(seed)
train_data, test_data = datasets.load_dataset('imdb', split=['train', 'test'])
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

def tokenize_data(example, tokenizer, max_length):
    tokens = tokenizer(example['text'])[:max_length]
    length = len(tokens)
    return {'tokens': tokens, 'length': length}

def numericalize_data(example, vocab):
    ids = [vocab[token] for token in example['tokens']]
    return {'ids': ids}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

def collate(batch, pad_index):
    batch_ids = [i['ids'] for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_length = [i['length'] for i in batch]
    batch_length = torch.stack(batch_length)
    batch_label = [i['label'] for i in batch]
    batch_label = torch.stack(batch_label)
    batch = {'ids': batch_ids,
             'length': batch_length,
             'label': batch_label}
    return batch


def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy

def train(dataloader, model, criterion, optimizer, device):

    model.train()
    epoch_losses = []
    epoch_accs = []
    count = 0
    total = 0
    for batch in tqdm.tqdm(dataloader, desc='training...', file=sys.stdout):
        start_time = time.time()
        ids = batch['ids'].to(device)
        length = batch['length']
        label = batch['label'].to(device)
        prediction = model(ids, length)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
        end_time = time.time()
        elapsed = end_time - start_time
        print("Time for batch", count, " is:", elapsed)
        if count > 0:
            total += elapsed
        if count == 40:
            break
        count += 1
    print("Avg time:", (total / 40))
    return epoch_losses, epoch_accs

def evaluate(dataloader, model, criterion, device):
    
    model.eval()
    epoch_losses = []
    epoch_accs = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='evaluating...', file=sys.stdout):
            ids = batch['ids'].to(device)
            length = batch['length']
            label = batch['label'].to(device)
            prediction = model(ids, length)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())

    return epoch_losses, epoch_accs

def train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, device):
    n_epochs = 1
    best_valid_loss = float('inf')

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    for epoch in range(n_epochs):
        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(valid_dataloader, model, criterion, device)

        train_losses.extend(train_loss)
        train_accs.extend(train_acc)
        valid_losses.extend(valid_loss)
        valid_accs.extend(valid_acc)
        
        epoch_train_loss = np.mean(train_loss)
        epoch_train_acc = np.mean(train_acc)
        epoch_valid_loss = np.mean(valid_loss)
        epoch_valid_acc = np.mean(valid_acc)
        
        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            torch.save(model.state_dict(), 'lstm.pt')
        
        print(f'epoch: {epoch+1}')
        print(f'train_loss: {epoch_train_loss:.3f}, train_acc: {epoch_train_acc:.3f}')
        print(f'valid_loss: {epoch_valid_loss:.3f}, valid_acc: {epoch_valid_acc:.3f}')
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(train_losses, label='train loss')
    ax.plot(valid_losses, label='valid loss')
    plt.legend()
    ax.set_xlabel('updates')
    ax.set_ylabel('loss')
    plt.savefig('loss.png')
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(train_accs, label='train accuracy')
    ax.plot(valid_accs, label='valid accuracy')
    plt.legend()
    ax.set_xlabel('updates')
    ax.set_ylabel('accuracy')
    plt.savefig('accuracy.png')

max_length = 256

train_data = train_data.map(tokenize_data, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})
test_data = test_data.map(tokenize_data, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})

test_size = 0.25

train_valid_data = train_data.train_test_split(test_size=test_size)
train_data = train_valid_data['train']
valid_data = train_valid_data['test']

min_freq = 5
special_tokens = ['<unk>', '<pad>']

vocab = torchtext.vocab.build_vocab_from_iterator(train_data['tokens'],
                                                  min_freq=min_freq,
                                                  specials=special_tokens)

unk_index = vocab['<unk>']
pad_index = vocab['<pad>']
collate = functools.partial(collate, pad_index=pad_index)

vocab.set_default_index(unk_index)

train_data = train_data.map(numericalize_data, fn_kwargs={'vocab': vocab})
valid_data = valid_data.map(numericalize_data, fn_kwargs={'vocab': vocab})
test_data = test_data.map(numericalize_data, fn_kwargs={'vocab': vocab})

train_data = train_data.with_format(type='torch', columns=['ids', 'label', 'length'])
valid_data = valid_data.with_format(type='torch', columns=['ids', 'label', 'length'])
test_data = test_data.with_format(type='torch', columns=['ids', 'label', 'length'])

def main():

    parser = argparse.ArgumentParser(description='Run the neural network')
    parser.add_argument('--master-ip', type=str, nargs=1, metavar="IP",
                    help='IP address of master node')
    parser.add_argument('--num-nodes', type=int, nargs=1, metavar='N',
                    help='Number of total nodes')
    parser.add_argument('--rank', type=int, nargs=1, metavar='N',
                    help='Rank of device')
    parser.add_argument('--model', type=str, nargs=1, metavar='filename', 
                    help="Pretrained model")
    args = parser.parse_args(sys.argv[1:])
    args = vars(args)

    vocab_size = len(vocab)
    embedding_dim = 300
    hidden_dim = 300
    output_dim = len(train_data.unique('label'))
    n_layers = 2
    bidirectional = True
    dropout_rate = 0.5
    if args['model'] == None:
        print("Creating new model")
        model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate, 
                    pad_index)
        model.apply(initialize_weights)
    else:
        print('Loading existing model')
        model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate, 
                    pad_index)
        model.load_state_dict(torch.load(args['model'][0]))
    print(model)
    vectors = torchtext.vocab.FastText()
    pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
    model.embedding.weight.data = pretrained_embedding

    lr = 0.01

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    batch_size = 512

    if args['master_ip'] != None:
        ip = args['master_ip'][0]
        nodes = args['num_nodes'][0]
        rankNum = args['rank'][0]
        dist.init_process_group('gloo',init_method=ip, world_size=nodes, rank=rankNum)
        ddp_model = DDP(model)
        ddp_model.to(device)
        train_sampler = DistributedSampler(train_data)
        train_loader = torch.utils.data.DataLoader(train_data,sampler=train_sampler, batch_size=batch_size, 
                                                collate_fn=collate, 
                                                shuffle=True)
        valid_sampler = DistributedSampler(valid_data)
        valid_loader = torch.utils.data.DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size, 
                                                collate_fn=collate, 
                                                shuffle=True)
        train_model(ddp_model, train_loader, valid_loader, criterion, optimizer, device)

    else:
        train_dataloader = torch.utils.data.DataLoader(train_data, 
                                                batch_size=batch_size, 
                                                collate_fn=collate, 
                                                shuffle=True)

        valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, collate_fn=collate)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, collate_fn=collate)
        train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, device)


main()