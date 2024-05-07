import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import wget, os, gzip, pickle, random, re, sys

IMDB_URL = 'http://dlvu.github.io/data/imdb.{}.pkl.gz'
IMDB_FILE = 'imdb.{}.pkl.gz'

PAD, START, END, UNK = '.pad', '.start', '.end', '.unk'

def load_imdb(final=False, val=5000, seed=0, voc=None, char=False):

    cst = 'char' if char else 'word'

    imdb_url = IMDB_URL.format(cst)
    imdb_file = IMDB_FILE.format(cst)

    if not os.path.exists(imdb_file):
        wget.download(imdb_url)

    with gzip.open(imdb_file) as file:
        sequences, labels, i2w, w2i = pickle.load(file)

    if voc is not None and voc < len(i2w):
        nw_sequences = {}

        i2w = i2w[:voc]
        w2i = {w: i for i, w in enumerate(i2w)}

        mx, unk = voc, w2i['.unk']
        for key, seqs in sequences.items():
            nw_sequences[key] = []
            for seq in seqs:
                seq = [s if s < mx else unk for s in seq]
                nw_sequences[key].append(seq)

        sequences = nw_sequences

    if final:
        return (sequences['train'], labels['train']), (sequences['test'], labels['test']), (i2w, w2i), 2

    # Make a validation split
    random.seed(seed)

    x_train, y_train = [], []
    x_val, y_val = [], []

    val_ind = set( random.sample(range(len(sequences['train'])), k=val) )
    for i, (s, l) in enumerate(zip(sequences['train'], labels['train'])):
        if i in val_ind:
            x_val.append(s)
            y_val.append(l)
        else:
            x_train.append(s)
            y_train.append(l)

    return (x_train, y_train), \
           (x_val, y_val), \
           (i2w, w2i), 2


def gen_sentence(sent, g):

    symb = '_[a-z]*'

    while True:

        match = re.search(symb, sent)
        if match is None:
            return sent

        s = match.span()
        sent = sent[:s[0]] + random.choice(g[sent[s[0]:s[1]]]) + sent[s[1]:]

def gen_dyck(p):
    open = 1
    sent = '('
    while open > 0:
        if random.random() < p:
            sent += '('
            open += 1
        else:
            sent += ')'
            open -= 1

    return sent

def gen_ndfa(p):

    word = random.choice(['abc!', 'uvw!', 'klm!'])

    s = ''
    while True:
        if random.random() < p:
            return 's' + s + 's'
        else:
            s+= word

def load_brackets(n=50_000, seed=0):
    return load_toy(n, char=True, seed=seed, name='dyck')

def load_ndfa(n=50_000, seed=0):
    return load_toy(n, char=True, seed=seed, name='ndfa')

def load_toy(n=50_000, char=True, seed=0, name='lang'):

    random.seed(0)

    if name == 'lang':
        sent = '_s'

        toy = {
            '_s': ['_s _adv', '_np _vp', '_np _vp _prep _np', '_np _vp ( _prep _np )', '_np _vp _con _s' , '_np _vp ( _con _s )'],
            '_adv': ['briefly', 'quickly', 'impatiently'],
            '_np': ['a _noun', 'the _noun', 'a _adj _noun', 'the _adj _noun'],
            '_prep': ['on', 'with', 'to'],
            '_con' : ['while', 'but'],
            '_noun': ['mouse', 'bunny', 'cat', 'dog', 'man', 'woman', 'person'],
            '_vp': ['walked', 'walks', 'ran', 'runs', 'goes', 'went'],
            '_adj': ['short', 'quick', 'busy', 'nice', 'gorgeous']
        }

        sentences = [ gen_sentence(sent, toy) for _ in range(n)]
        sentences.sort(key=lambda s : len(s))

    elif name == 'dyck':

        sentences = [gen_dyck(7./16.) for _ in range(n)]
        sentences.sort(key=lambda s: len(s))

    elif name == 'ndfa':

        sentences = [gen_ndfa(1./4.) for _ in range(n)]
        sentences.sort(key=lambda s: len(s))

    else:
        raise Exception(name)

    tokens = set()
    for s in sentences:

        if char:
            for c in s:
                tokens.add(c)
        else:
            for w in s.split():
                tokens.add(w)

    i2t = [PAD, START, END, UNK] + list(tokens)
    t2i = {t:i for i, t in enumerate(i2t)}

    sequences = []
    for s in sentences:
        if char:
            tok = list(s)
        else:
            tok = s.split()
        sequences.append([t2i[t] for t in tok])

    return sequences, (i2t, t2i)


(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

class SequenceModel(nn.Module):
    def __init__(self, vocab_size, embedding_size=300, hidden_size=300, num_classes=2):
        super(SequenceModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)

        # Linear layer
        self.linear = nn.Linear(embedding_size, hidden_size)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes)
        

    def forward(self, inputs):

        # Embedding layer
        embedded = self.embedding(inputs)

        # Linear layer
        linear_output = self.linear(embedded)

        # ReLu activation function
        activations = F.relu(linear_output)

        # Max pooling
        pooled_output = activations.mean(dim=1)

        # Output layer
        output = self.output_layer(pooled_output)
        
        return output


def fixed_batching_with_padding(sequences, labels, batch_size, pad_index):
    batches = []
    num_batches = (len(sequences) + batch_size - 1) // batch_size  # Calculate number of batches
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # Pad sequences to the length of the longest sequence in the batch
        max_length = max(len(seq) for seq in batch_sequences)
        padded_batch_sequences = []
        for seq in batch_sequences:
            padded_seq = seq + [pad_index] * (max_length - len(seq))
            padded_batch_sequences.append(padded_seq)
        
        # Convert to tensors
        batch_tensor = torch.tensor(padded_batch_sequences, dtype=torch.long)
        label_tensor = torch.tensor(batch_labels, dtype=torch.long)
        
        batches.append((batch_tensor, label_tensor))
    
    return batches

class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(SelfAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads if input_dim % num_heads == 0 else (input_dim // num_heads) + 1

        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(input_dim, input_dim, bias=False)
        self.W_k = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v = nn.Linear(input_dim, input_dim, bias=False)
        
        # Dropout layer
        #self.dropout = nn.Dropout(dropout)
        
        # Output projection
        #self.W_o = nn.Linear(input_dim, input_dim)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        

        #Linear projections of Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split tensors into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        #print("Shape of Q after reshaping:", Q.shape)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        #print("Shape of K after reshaping:", K.shape)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        #print("Shape of V after reshaping:", V.shape)

        # Scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        #attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        
        # Concatenate
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        #output = self.W_o(output)
        
        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, k, heads, dropout=0.1):
        super().__init__()

        self.attention = SelfAttention(k, heads, dropout)
        self.layer_norm = nn.LayerNorm(k)
        # Feedforward layer
        self.linear1 = nn.Linear(k, 4 * k)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(4 * k, k)
        # Layer normalization after feedforward
        self.layer_norm_ff = nn.LayerNorm(k)
    

    def forward(self, x):
        # Self-attention
        attention_output = self.attention(x)
        # Layer normalization after attention
        attention_output = self.layer_norm(attention_output)
        # Feedforward
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(attention_output))))
        # Layer normalization after feedforward
        output = self.layer_norm_ff(ff_output + attention_output)

        return output
    

class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
            self.tblocks = nn.Sequential(*tblocks)

        # Output layer
        self.output_layer = nn.Linear(k, num_classes)


    def forward(self, x):
        # Generate token embeddings
        tokens = self.token_emb(x)
        b, t, k = tokens.size()

        # Generate position embeddings
        positions = torch.arange(t)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        # Add positional embeddings to token embeddings
        x = tokens + positions

        # Pass through Transformer blocks
        x = self.tblocks(x)

        # Mean pooling
        x = x.mean(dim=1)

        # Compute log-probabilities using softmax
        return F.log_softmax(x, dim=1)
    

vocab_size = len(w2i)
model = Transformer(k=300, heads=4, depth=4, seq_length=50, num_tokens=vocab_size, num_classes=2)
loss_fn = F.cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5
batch_size = 32 
num_classes=2


# Training Loop
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in fixed_batching_with_padding(x_train, y_train, batch_size, pad_index=w2i[PAD]):
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in fixed_batching_with_padding(x_val, y_val, batch_size, pad_index=w2i[PAD]):
            output = model(inputs)
            predicted_labels = torch.argmax(output, dim=1)
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
    val_accuracy = correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy:.2%}")

