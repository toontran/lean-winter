#encoding: utf-8
'''
Run inference on GPTf-Neo

Author: Tung Tran
'''
import copy
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import random
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn.functional as F
from tqdm import trange

# Checkpoint model for resuming
import os
from os.path import isfile, join, expanduser
import re
import json

from model import GPT2LMHeadModel, GPT2Config
from encoder import get_encoder

# Reproducibility
seed = random.randint(0, 2147483647)
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pretrained_GPT2(model):
    home = expanduser("~")
    state_dict = torch.load(os.path.join(home, 'data/GPT2/gpt2-pytorch_model.bin'), map_location='cpu' if not torch.cuda.is_available() else None)

    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if key.endswith(".g"):
            new_key = key[:-2] + ".weight"
        elif key.endswith(".b"):
            new_key = key[:-2] + ".bias"
        elif key.endswith(".w"):
            new_key = key[:-2] + ".weight"
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    # model.load_state_dict(state_dict)
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    start_model = model
    if hasattr(model, "transformer") and all(not s.startswith('transformer.') for s in state_dict.keys()):
        start_model = model.transformer
    load(start_model, prefix="")

    model.set_tied()
    with torch.no_grad():
        model.transformer.wte.weight[:50257, :] = state_dict['wte.weight']
#     print(missing_keys, unexpected_keys, error_msgs)

def load_checkpoint(config, output_dir, output_prefix, new_train = 0):
    files = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
#         print(f"Creating output directory {output_dir}")
    else:    
        files = [f for f in os.listdir(output_dir) if isfile(join(output_dir, f))]

    # Find training stats
    stats_filename = "stats.json"
    if stats_filename in files:
        with open(os.path.join(output_dir, stats_filename), 'r') as f:
            training_stats = json.load(f)
    else:
        training_stats = []
#     print(f"Length of current stats file: {len(training_stats)}\n")

    # Find checkpoints
    r = re.compile(f"{output_prefix}-[0-9]*.pt")
    model_files = list(filter(r.match, files))
    epochs_list = [int(x[ len(output_prefix)+1 : -3 ]) for x in model_files]
    model = GPT2LMHeadModel(config)
    if epochs_list:
        current_epoch = max(epochs_list)
        min_val_loss = min([x["Valid. Loss"] for x in training_stats])
        model.load_state_dict(
                torch.load(os.path.join(output_dir, f"{output_prefix}-{current_epoch}.pt"))
        )
#         print(f"Checkpoint #{current_epoch} found, with min_val_loss of {min_val_loss:.3f}, resuming.")
    else:
        current_epoch = 0
        min_val_loss = float("inf")
        load_pretrained_GPT2(model)
        print(f"Checkpoint not found, starting from epoch #{current_epoch}, min_val_loss of {min_val_loss:.3f}.")
    return model, current_epoch, min_val_loss, training_stats

def _parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--temperature", type=int, default=0.7, required=False)
    parser.add_argument("--top_p", type=float, default=0.5, required=False)
    parser.add_argument("--n", type=int, default=1, required=False) # ???
    parser.add_argument("--best_of", type=int, default=1, required=False) # ???, required=False
    parser.add_argument("--stream", type=bool, default=False, required=False) # Dunno what this does
    parser.add_argument("--logprobs", default="[0.5,0.5]", required=False)
    parser.add_argument("--echo", default=False, required=False) # Boolean
    parser.add_argument("--stop", default=False, required=False)
    parser.add_argument("--presence_penalty", type=float, default=0.1, required=False) # ???
    parser.add_argument("--frequency_penalty", type=int, default=1, required=False) # ???
    return parser.parse_args()


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

def sample_sequence(model, encoder, max_length=100, start_token=None, batch_size=None, text=None, temperature=1, top_k=0, device='cuda', sample=True):
    # TODO: sample which sequence
    assert text is not None, 'Specify context!'
    context = torch.tensor(encoder.encode(text), device=device, dtype=torch.long).unsqueeze(0)
#     print(f"Context:{text}, encoded:{context}")
    prev = context
    outputs = []
    prevs = []
    pasts = []
    past = None
    with torch.no_grad():
        logits, past = model(prev, past=past)
        logits = logits[:, -1, :] / temperature
        logits = top_k_logits(logits, k=top_k)
        log_probs = F.softmax(logits, dim=-1)
        values, prev = torch.topk(log_probs, k=5, dim=-1)
        p = prev.tolist()
#         for i in range(len(p[0])):
#             text = encoder.decode([p[0][i]])
#             print(f"{text} -> {values[0][i].item()}")
        # Deal with stop reason and early stopping properly, please!
        for j in range(5):
            text = encoder.decode([p[0][j]])
            if "<|endoftext|>" in text or "<|GOAL|>" in text:
                continue
            value = values[0][j].item()
            
            outputs.append({
                "text": text,
                "logprobs": {"token_logprobs": [value]},
            })
            
            prevs.append(torch.clone(prev))
            pasts.append(past)
#             if text == "<|endoftext|>":
#                 return outputs
        for j in range(len(outputs)): # Magic number
            for i in range(max_length-1):
                logits, pasts[j] = model(prevs[j], past=pasts[j])
                logits = logits[:, -1, :] / temperature
                logits = top_k_logits(logits, k=top_k)
                log_probs = F.softmax(logits, dim=-1)
                values, prevs[j] = torch.topk(log_probs, k=1, dim=-1)
                # Also need to trim after <|endoftext|>
                p = prevs[j].tolist()
                text = encoder.decode(p[0])
                if "<|endoftext|>" in text or "<|GOAL|>" in text:
                    break
                value = values[0][0].item()
                outputs[j]["text"] += text
#                 outputs[j]["logprobs"].append(value)
                 
    return outputs


if __name__ == "__main__":
    
    args = _parse()
#     print(args)
    
    output_dir="/home/tung/" 
    output_prefix="proofstep"
    
    config = GPT2Config()
    config.vocab_size += 2 # 2 new tokens
    model, current_epoch, _, _ = load_checkpoint(config, output_dir, output_prefix)
#     print(type(model) == GPT2LMHeadModel)

    device = torch.device("cuda")
    model = model.cuda()
    
    if "<|GOAL|>" not in args.prompt:
        args.prompt = json.loads(args.prompt)
        prompt = args.prompt["prompt"].replace("[LN]", "").replace("GOAL", "<|GOAL|>").replace("PROOFSTEP", "<|PROOFSTEP|>").rstrip(" ")
    else:
        prompt = args.prompt
    
    enc = get_encoder()
    outputs = sample_sequence(model, enc, max_length=200, text=prompt, temperature=args.temperature)
    print(json.dumps({
        "model": "gptf-neo",
        "prompt": prompt,
        "a_unicode": u"♠",
        "choices": outputs,
    }, ensure_ascii=False))
    


# length = 20
# temperature=0.7
# top_k=40
# model.cuda()



    
    






# Output json containing ...?
# {
#     "id": "cmpl-GERzeJQ4lvqPk8SkZu4XMIuR",
#     "object": "text_completion",
#     "created": 1586839808,
#     "model": "davinci:2020-05-03",
#     "choices": [{
#         "text": " of reading speed. You",
#         "index": 0,
#         "logprobs": {token_logprobs: []},
#         "finish_reason": "length" # or "<|endoftext|>"
#     }]
# }


# Want to try changing cmd in openai.lean.