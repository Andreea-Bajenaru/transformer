import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import torch.nn.utils.rnn as rnn_utils
import wandb
import random
import torch.distributions as dist
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gzip
import os
import time
import tqdm
import sys
import wget
import zipfile

ENWIK8_URL = 'https://mattmahoney.net/dc/enwik8.zip'
ENWIK8_ZIP ='enwik8.zip'
ENWIK8_FILE = 'enwik8'

def download_if_not_exists(url, filename):
    """
    Download the file from the given URL if it does not exist locally.
    
    :param url: URL to download the file from
    :param filename: Local filename to save the downloaded file
    """
    if not os.path.exists(filename):
        wget.download(url, filename)

def extract_zip(zip_path, extract_to='.'):
    """
    Extract the contents of a zip file.
    
    :param zip_path: Path to the zip file
    :param extract_to: Directory to extract the files to
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def enwik8(path=None, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    """
    Load the enwik8 dataset from the Hutter challenge.
    
    :param path: Path to the dataset file
    :param n_train: Number of training samples
    :param n_valid: Number of validation samples
    :param n_test: Number of test samples
    :return: Training, validation, and test datasets as PyTorch tensors
    """
    if path is None:
        download_if_not_exists(ENWIK8_URL, ENWIK8_ZIP)
        extract_zip(ENWIK8_ZIP)
        path = ENWIK8_FILE

    with open(path, 'rb') as file:
        X = np.frombuffer(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

def enwik8_bytes(path=None, split=(90, 5, 5)):
    """
    Load the enwik8 dataset from the Hutter challenge as a list of bytes.
    
    :param path: Path to the dataset file
    :param split: Tuple specifying the train, validation, and test split ratios
    :return: Training, validation, and test datasets as byte sequences
    """
    if path is None:
        download_if_not_exists(ENWIK8_URL, ENWIK8_ZIP)
        extract_zip(ENWIK8_ZIP)
        path = ENWIK8_FILE

    with gzip.open(path, 'r') if path.endswith('.gz') else open(path, 'rb') as file:
        all_data = file.read()

        split = tuple(s / sum(split) for s in split)
        split = tuple(int(s * len(all_data)) for s in split)

        train, val, test = all_data[:split[0]], all_data[split[0]:split[0] + split[1]], all_data[split[0] + split[1]:]
        return train, val, test

def enwik8_string(path=None, split=(90, 5, 5)):
    """
    Load the enwik8 dataset from the Hutter challenge as a string.
    
    :param path: Path to the dataset file
    :param split: Tuple specifying the train, validation, and test split ratios
    :return: Training, validation, and test datasets as strings
    """
    if path is None:
        download_if_not_exists(ENWIK8_URL, ENWIK8_ZIP)
        extract_zip(ENWIK8_ZIP)
        path = ENWIK8_FILE

    with gzip.open(path, 'rt', encoding='utf-8') if path.endswith('.gz') else open(path, 'r', encoding='utf-8') as file:
        all_data = file.read()

        split = tuple(s / sum(split) for s in split)
        split = tuple(int(s * len(all_data)) for s in split)

        train, val, test = all_data[:split[0]], all_data[split[0]:split[0] + split[1]], all_data[split[0] + split[1]:]
        return train, val, test

def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """

    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()

def sample_sequence(model, seed, max_context, length=600, temperature=0.5, verbose=False):
    """
    Sequentially samples a sequence from the model, token by token.

    :param model:
    :param seed: The sequence to start with.
    :param length: The total number of characters to sample.
    :param temperature: The sampling temperature.
    :param verbose: If true, the sampled sequence is also printed as it is sampled.

    :return: The sampled sequence, including the seed.
    """

    sequence = seed.detach().clone()

    if verbose: # Print the seed, surrounded by square brackets
        print('[', end='', flush=True)
        for c in seed:
            print(str(chr(c)), end='', flush=True)
        print(']', end='', flush=True)

    for _ in range(length):

        # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
        input = sequence[-max_context:]

        # Run the current input through the model
        output = model(input[None, :])

        # Sample the next token from the probabilitys at the last position of the output.
        c = sample(output[0, -1, :], temperature)

        if verbose:
            print(str(chr(max(32, c))), end='', flush=True)

        sequence = torch.cat([sequence, c[None]], dim=0) # Append the sampled token to the sequence

    print()
    return seed

def sample_batch(data, length, batch_size):
    """
    Takes the data (a single sequence of tokens) and slices out a batch of subsequences to provide as input to the model.

    For each input instance, it also slices out the sequence that is shifted one position to the right, to provide as a
    target for the model.

    :param data: The (training) data. A single vector of tokens represented by integers
    :param length: The length of the subsequences in the batch.
    :param batch_size: The number of subsequences in the batch
    :return: A pair (input, target) of minteger matrices representing the input and target for the model.
    """

    # Sample the starting indices of the sequences to slice out.
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - length - 1)

    # Slice out the input sequences
    seqs_inputs  = [data[start:start + length] for start in starts]
    # -- the start index is the one we just sampled, and the end is exactly 'lentgh' positions after that.
    seqs_target = [data[start + 1:start + length + 1] for start in starts]
    # -- The target is the same sequence as input, except one character ahead (we are asking the model to predict the
    #    next character at each position)

    # We now have two lists of torch vectors, which we can concatenate into matrices of batch_size-by-length
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    # -- Note that we add a singleton dimenson to each vector, s[None.,:], and then concatenate along that dimension.

    return inputs, target

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false

    In place operation

    :param tns:
    :return:
    """

    h, w = matrices.size(-2), matrices.size(-1)

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[..., indices[0], indices[1]] = maskval

def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def here(subpath=None):
    """
    :return: the path in which the package resides (the directory containing the 'former' dir)
    """
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', subpath))

def contains_nan(tensor):
    return bool((tensor != tensor).sum() > 0)


tics = []


def tic():
    tics.append(time.time())

def toc():
    if len(tics)==0:
        return None
    else:
        return time.time()-tics.pop()

def slice_diag(matrix, l, dv=None):
    """
    Take a batch of attention matrices for relative position encodings and slice out the relevant attentions. These
    are the length l sequences starting at the diagonal

    :param matrix:
    :return:
    """
    if dv is None:
        dv = d(matrix)

    h, w = matrix.size(-2), matrix.size(-1)

    assert w == 2 * l -1, f'(h, w)= {(h, w)}, l={l}'

    rest = matrix.size()[:-2]

    matrix = matrix.view(-1, h, w)
    b, h, w = matrix.size()

    result = matrix.view(b, -1)
    result = torch.cat([result, torch.zeros(b, l, device=dv)], dim=1)
    assert result.size() == (b, 2 * l * l), f'result.size() {result.size()}'

    result = result.view(b, l, 2*l)
    result = result[:, :, :l]

    result = result.view(*rest, h, l)
    return result

# Used for converting between nats and bits
LOG2E = math.log2(math.e)
LOGE2 = math.log(2.0)

def compute_compression(model, data, context, batch_size, verbose=False,
                        tbw:SummaryWriter=None, tok=None, skip=0):


    """
    Compute the _compression_ of a dataset under a model. That is, given a model, in how many bits could we represent
    the dataset. This requires us to turn a given probability distribution into a code for the outcomes.

    See [this video](https://youtu.be/mSneVjDvzNQ) for an explanation.

    :param model: A sequence-to-sequence model that takes as input a (sub) sequence of integers and produces a probability
    distributuion on the output.
    :param data: A singe list of integers representing the  data
    :return: The result of the computation in "bits per byte". That is, how many bits does the compressed representation
    spend on each byte (=ASCII character) of the raw data.
    """

    bits, tot = 0.0, 0
    batch = []
    # Buffer, every time it fills up, we run it through the model
    # --- For the sake of speed we want to process the data in batches. For each token in the data, we make a
    #     prediction based on all the `context` tokens before it. This means that for each subsequence in the batch, we
    #     need to shift the start/end indices ahead by one token.
    #
    #     After we pass the batch through the model, we look at only the probabilities predicted for the last token.

    target_indices = []
    i, ic = 0, 0

    for current in tqdm.trange(skip, data.size(0)) if verbose else range(skip, data.size(0)):

        # `current` is the character which we will ultimately predict

        fr = max(0, current - context)
        to = current + 1

        instance = data[fr:to].to(torch.long) # the subsequence of the data to add to the batch
        # -- slice out an instance of size context + 1 (or shorter at the start of the data)

        # if tok is not None:
        #     print(instance[:-1], tok.decode(instance[:-1]))
        #     print(instance[-1:], tok.decode(instance[-1:]))

        target_indices.append(instance.size(0) - 2) # index of the last element of the input to the model

        if instance.size(0) < context + 1:
            assert skip < context # We shouldn't get here if we skip the first `context` characters

            # the index in the output tensor of the character we want to predict
            # -- It's context + 1, because we clip off the last token as a target

            pad = torch.zeros(size=(context + 1 - instance.size(0),), dtype=torch.long)
            instance = torch.cat([instance, pad], dim=0)
            # -- the first tokens don't have enough tokens preceding them, so we pad them to the right size.

            assert instance.size(0) == context + 1 # all instances should be `context` + 1 long

        if torch.cuda.is_available():
            instance = instance.cuda()

        batch.append(instance[None, :])
        # -- We add a singleton dimension to concatenate along later.

        if len(batch) == batch_size or current == data.size(0) - 1:
            # batch is full or we are at the last instance, run it through the model

            b = len(batch)

            ti = torch.tensor(target_indices) + 1
            all = torch.cat(batch, dim=0)
            inputs = all[:, :-1] # input
            target = all[torch.arange(b), ti]  # target values

            with torch.no_grad():
                if next(model.parameters()).is_cuda:
                    inputs = inputs.cuda()
                output = model(inputs)

            if type(output) != torch.Tensor:
                output = torch.log_softmax(output.logits, dim=2) # To make the method work for GPT2 models from Huggingface

            assert output.size()[:2] == (b, context), f'was: {output.size()}, should be {(b, context, -1)}'

            lnprobs = output[torch.arange(b, device=d()), target_indices, target]
            log2probs = lnprobs / LOGE2
            # -- The model produces natural logarithms of probabilities, but we need base-2 logarithms of the
            #    probabilities, since these give us bits.

            if tbw is not None:
                for j, lp in enumerate(log2probs):
                    i += 1
                    tbw.add_scalar('compression/bits-per-token', -lp, i)

                    if tok is not None:
                        nc = len(tok.decode(target[j]))
                        ic += nc
                        tbw.add_scalar('compression/bits-per-byte', -lp/nc, ic)

            bits += - log2probs.sum() # Add the bits for each character (the negative log_2 probabilities) to the running total
            batch, target_indices = [], []  # clear the buffer

    if isinstance(bits, torch.Tensor):
        bits = bits.item()

    return bits # total nr of bits used

def estimate_compression(model, data, nsamples, context, batch_size, verbose=False, model_produces_logits=False):
    
    """
    Estimates the compression by sampling random subsequences instead of predicting all characters.

    NB: This doesn't work for GPT-2 style models with super-character tokenization, since the tokens and number of
    characters are mismatched.

    :param model: A sequence-to-sequence model that takes as input a (sub) sequence of integers and produces a probability
    distributuion on the output.
    :param data: A singe list of integers representing the data
    :return: The result of the computation in "bits per byte". That is, how many bits does the compressed representation
    spend on each byte (=ASCII character) of the raw data.
    """

    bits, tot = 0.0, 0
    batch = []

    # indices of target characters in the data
    gtargets = random.sample(range(data.size(0)), k=nsamples)

    # Buffer, every time it fills up, we run it through the model
    # --- For the sake of speed we want to process the data in batches. For each token in the data, we make a
    #     prediction based on all the `context` tokens before it. This means that for each subsequence in the batch, we
    #     need to shift the start/end indices ahead by one token.
    #
    #     After we pass the batch through the model, we look at only the probabilities predicted for the last token.
    target_indices = []

    for i, current in enumerate(tqdm.tqdm(gtargets) if verbose else gtargets):
        # current is the character to be predicted

        fr = max(0, current - context)
        to = current + 1

        instance = data[fr:to].to(torch.long) # the subsequence of the data to add to the batch
        # -- slice out an instance of size context + 1 (or shorter at the start of the data)

        target_indices.append(instance.size(0) - 2) # index of the last element of the context

        if instance.size(0) < context + 1:
            # the index in the output tensor of the character we want to predict
            # -- It's context + 1, because we clip off the last token as a target

            pad = torch.zeros(size=(context + 1 - instance.size(0),), dtype=torch.long)
            instance = torch.cat([instance, pad], dim=0)
            # -- the first tokens don't have enough tokens preceding them, so we pad them to the right size.

            assert instance.size(0) == context + 1 # all instances should be `context` + 1 long

        if torch.cuda.is_available():
            instance = instance.cuda()

        batch.append(instance[None, :])
        # -- We add a singleton dimension to concatenate along later.

        if len(batch) == batch_size or i == len(gtargets) - 1:
            # batch is full, or we are at the last instance, run it through the model

            b = len(batch)

            all = torch.cat(batch, dim=0)
            inputs = all[:, :-1] # input
            target = all[:, -1]  # target values

            with torch.no_grad():
                if next(model.parameters()).is_cuda:
                    inputs = inputs.cuda()
                output = model(inputs)

                if model_produces_logits:
                    output = F.log_softmax(output, dim=-1)

            if type(output) != torch.Tensor:
                output = torch.log_softmax(output.logits, dim=2) # To make the method work for GPT2 models from Huggingface

            assert output.size()[:2] == (b, context), f'was: {output.size()}, should be {(b, context, -1)}'

            lnprobs = output[torch.arange(b, device=d()), target_indices, target]
            log2probs = lnprobs * LOG2E
            # -- The model produces natural logarithms of probabilities, but we need base-2 logarithms of the
            #    probabilities, since these give us bits.

            bits += - log2probs.sum() # Add the bits for each character (the negative log_2 probabilties) to the running total
            batch, target_indices = [], []  # clear the buffer

    return bits.item() / nsamples # total nr of bits used

def batcher(data, seq_len, batch_size, num_batches):
    x_batches, y_batches = [],[]
    
    for i in range(num_batches):
        start_idx = i * batch_size * (seq_len + 1)
        x_batch,y_batch = [],[]
        for _ in range(batch_size):
            x_batch.append(data[start_idx:start_idx + seq_len])
            y_batch.append(data[start_idx + 1: start_idx + seq_len + 1])

            start_idx += seq_len + 1
        
        x_batch = torch.cat([x[None,:] for x in x_batch], dim=0).to(torch.long)
        y_batch = torch.cat([y[None,:] for y in y_batch], dim=0).to(torch.long)

        x_batches.append(x_batch)
        y_batches.append(y_batch)

    return x_batches, y_batches

def byte_to_string(byte_data):
    return ''.join([chr(byte) for byte in byte_data])

def string_to_byte(string_data):
    return [ord(char) for char in string_data]

def integer_to_bits(integer, num_bits):
    binary_representation = bin(integer)[2:].zfill(num_bits)  
    return [int(bit) for bit in binary_representation]  


def slice_batches(data, length, batch_size):
    """
    Slice out batches of subsequences from the data with a fixed length,
    and create input-target pairs for the model.

    :param data: The (training) data. A single vector of tokens represented by integers.
    :param length: The length of the subsequences in the batch.
    :param batch_size: The number of subsequences in the batch.
    :return: A pair (input, target) of integer matrices representing the input and target for the model.
    """
    input_sequences = []
    target_sequences = []

    for _ in range(batch_size):
        # Randomly select a starting index for the subsequence
        start_index = random.randint(0, len(data) - length - 1)
        end_index = start_index + length

        # Extract the input subsequence
        input_seq = [integer_to_bits(token, 8) for token in data[start_index:end_index]]

        # Extract the target subsequence (shifted one position to the right)
        target_seq = [integer_to_bits(token, 8) for token in data[start_index + 1:end_index + 1]]

        # Append the input and target sequences to the batch
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)

    # Convert input_sequences and target_sequences to integer tensors
    input_tensor = torch.tensor(input_sequences)
    target_tensor = torch.tensor(target_sequences)

    # Stack input and target tensors along a new dimension to represent batches
    input_batches = input_tensor.permute(1, 0, 2)  # Shape: (sequence_length, batch_size, num_bits_per_token)
    target_batches = target_tensor.permute(1, 0, 2)  # Shape: (sequence_length, batch_size, num_bits_per_token)

    return input_batches, target_batches


api_key = 'b574a868036a98ebf558d9abaf60f7fed3738a31'
wandb.login(key=api_key)

# Initialize wandb
wandb.init(project="transformer_enwik8")

# Hyperparameters and configurations
config = {
    "total_batches": 10000,  # Define the total number of batches instead of epochs
    "batch_size": 64,
    "lr": 0.0001,
    "context": 256,
    "d_model": 512,
    "num_heads": 4,
    "d_ff": 2048,
    "num_layers": 12,
    "patience": 20,
    "print_interval": 1000,
    "validation_interval": 1000,
}

# Log hyperparameters
wandb.config.update(config)

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadedSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        output = self.out(context)

        return output, attention_weights

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedSelfAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, mask)
        x = self.layer_norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        return x

class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(num_tokens, k)
        self.pos_embedding = nn.Embedding(seq_length, k)  # Learnable positional embeddings
        self.layers = nn.ModuleList([TransformerBlock(k, heads, k * 4) for _ in range(depth)])
        self.fc_out = nn.Linear(k, num_classes)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.fc_out(x)
        return x

def validate(model, data, criterion, batch_size=64, sequence_length=256, num_batches=5):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = sample_batch(data, length=sequence_length, batch_size=batch_size)
            inputs, targets = inputs.to(d()), targets.to(d())

            outputs = model(inputs)
            outputs = outputs.view(-1, 256)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss

def generate_text(model, seed, max_context, length=500, temperature=0.5, verbose=False):

    generated_text = seed.clone().detach()

    if verbose:
        print('[', end='', flush=True)
        for c in seed:
            print(str(chr(c)), end='', flush=True)
        print(']', end='', flush=True)

    for _ in range(length):
        input_seq = generated_text[-max_context:].unsqueeze(0)

        with torch.no_grad():
            output = model(input_seq)

        # Sample the next character from the output distribution
        next_char = sample(output[0, -1, :], temperature)

        if verbose:
            print(str(chr(max(32, next_char))), end='', flush=True)

        generated_text = torch.cat([generated_text, next_char.unsqueeze(0)], dim=0)

    print()
    return generated_text

best_val_loss = float('inf')
counter = 0

# Load data
trX, vaX, teX = enwik8()
trX, vaX, teX = trX.to(d()), vaX.to(d()), teX.to(d())

# Initialize model, loss function, optimizer
model = Transformer(
    k=512,
    heads=4,
    depth=12,
    seq_length=256,
    num_tokens=256,
    num_classes=256
).to(d())

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"])

# Training loop
for current_batch in range(config["total_batches"]):
    model.train()
    inputs, targets = sample_batch(trX, config["context"], config["batch_size"])
    inputs, targets = inputs.to(d()), targets.to(d())

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs.view(-1, 256), targets.view(-1))
    loss.backward()
    optimizer.step()

    if (current_batch + 1) % config["print_interval"] == 0:
        print(f'Batch {current_batch + 1}: Training Loss = {loss.item()}')
        wandb.log({"batch": current_batch + 1, "train_loss": loss.item()})

    if (current_batch + 1) % config["validation_interval"] == 0:
        val_loss, val_accuracy = validate(model, vaX, criterion, config["batch_size"], config["context"], 5)
        print(f'Batch {current_batch + 1}: Validation Loss = {val_loss}, Accuracy = {val_accuracy}%')
        wandb.log({"batch": current_batch + 1, "val_loss": val_loss, "val_accuracy": val_accuracy})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model")
            counter = 0
        else:
            counter += 1
            if counter >= config["patience"]:
                print("Early stopping")
                break

seed_sequence = torch.randint(256, (config["context"],)).to(d())
sampled_sequence = sample_sequence(model, seed_sequence, max_context=config["context"], length=500, temperature=0.5)
print(f"Sampled sequence: {sampled_sequence}")

compression_bits_per_byte = estimate_compression(model, vaX, nsamples=1000, context=config["context"], batch_size=64, verbose=True)
print(f"Estimated compression: {compression_bits_per_byte} bits per byte")
wandb.log({"compression_bits_per_byte": compression_bits_per_byte})
