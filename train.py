import math
import gzip
import random
import tqdm
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from light_recurrent_unit_pytorch.light_recurrent_unit import LightRecurrentUnitBlock, RMSNorm

torch.set_float32_matmul_precision('high')

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 64
GENERATE_EVERY = 500
GENERATE_LENGTH = PRIME_LENGTH + 128
SEQ_LEN = 128

# helpers

def exists(v):
    return v is not None

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

def base_decoding(
    net,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
    filter_thres = 0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    for _ in range(sample_num_times):
        logits = net(out)
        logits = logits[:, -1]

        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample), dim = -1)

    return out[..., prompt_seq_len:]

# LRU based LM

class LRULM(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = nn.ModuleList([LightRecurrentUnitBlock(dim = dim, has_ff_block = True) for _ in range(depth)])

        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

    def forward(self, x, return_loss = False):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        for block in self.layers:
            x = block(x)

        logits = self.to_logits(x)

        if not return_loss:
            return logits

        return F.cross_entropy(logits.transpose(1, 2), labels)

# the LRULM char language model

model = LRULM(
    num_tokens = 256,
    dim = 384,
    depth = 6
).cuda()

model = torch.compile(model)

# prepare enwik8 data

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

# optimizer

optim = Adam(model.parameters(), lr = LEARNING_RATE)

train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
    model.train()

    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)

        loss = model(data, return_loss = True)

        (loss / GRAD_ACCUM_EVERY).backward()

    print(f"training loss: {loss.item():.3f}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            valid_data = next(val_loader)

            loss = model(valid_data, return_loss = True)
            print(f"validation loss: {loss.item():.3f}")

    if i % GENERATE_EVERY == 0:
        model.eval()

        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        inp = inp.cuda()

        prime = decode_tokens(inp)
        print(f"%s \n\n %s", (prime, "*" * 100))

        prompt = inp[None, ...]

        sampled = base_decoding(model, prompt, GENERATE_LENGTH)

        base_decode_output = decode_tokens(sampled[0])

        print("\n\n", base_decode_output, "\n")
