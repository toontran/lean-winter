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

import subprocess
import os
from os.path import isfile, join, expanduser


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
#     print(f"Found checkpoints: {model_files}")
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
    parser.add_argument("--temperature", type=int, default=0.9, required=False)
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

def sample_sequence(model, encoder, max_length=16, start_token=None, batch_size=None, text=None, temperature=1, top_k=0, device='cuda', sample=True):
    # TODO: sample which sequence
    assert text is not None, 'Specify context!'
    context = torch.tensor(encoder.encode(text), device=device, dtype=torch.long).unsqueeze(0)
#     print(f"Context:{text}, encoded:{context}")
    prev = context
    outputs = []
    prevs = []
    pasts = []
    past = None
    num_samples = 10
    with torch.no_grad():
        logits, past = model(prev, past=past)
        logits = logits[:, -1, :] / temperature
        logits = top_k_logits(logits, k=top_k)
        log_probs = F.softmax(logits, dim=-1)
        values, prev = torch.topk(log_probs, k=num_samples, dim=-1)
        p = prev.tolist()
#         for i in range(len(p[0])):
#             text = encoder.decode([p[0][i]])
#             print(f"{text} -> {values[0][i].item()}")
        # Deal with stop reason and early stopping properly, please!
        for j in range(num_samples): # Magic number
            text = encoder.decode([p[0][j]])
#             if "<|endoftext|>" in text or "<|GOAL|>" in text:
#                 continue
            value = values[0][j].item()
            
            outputs.append({
                "text": text,
                "logprobs": {"token_logprobs": [value]},
                "tokens": [p[0][j]],
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
#                 if "<|endoftext|>" in text or "<|GOAL|>" in text:
#                     break
                value = values[0][0].item()
                outputs[j]["text"] += text
                outputs[j]["tokens"].append(p[0][0])
#                 outputs[j]["logprobs"].append(value)
        # Trim all output after <|endoftext|> token
        indices_to_pop = []
        for i, d in enumerate(outputs):
            idx_eot = max(d["text"].find("<|endoftext|>"), 0)
            idx_goal = max(d["text"].find("<|GOAL|>"), 0)
#             idx_comma = d["text"].find(",")
#             idx_comma = max(idx_comma, 0) if d["text"].find("[") < idx_comma else max_length+1
#             idx = min(idx_eot, idx_goal, idx_comma)
            idx = min(idx_eot, idx_goal)
            if idx_eot == idx_goal:# and (idx_comma == idx_eot or idx_comma > max_length):
                indices_to_pop.append(i)
            else:
                d["text"] = d["text"][:idx]
                d["tokens"] = d["tokens"][:idx]
        [outputs[i] for i in range(len(outputs)) if i not in indices_to_pop]
    return outputs

def sample_parseable_sequence(model, encoder, max_length=16, text=None, temperature=1, top_k=1, device='cuda'):
    # terminate if encounter special tokens OR max_depth reached (in which case, auto eliminate)
    # infer -> top_p top_k -> p predictions
    # pass down dfs(), put result into list of predictions
    # if length list > 100 or so, try parsing & eliminate un-parseable candidates
    # if parseable > 30, return first 30 (best candidates)
    # return list
    # TODO: Try top_p top_k first, then try writing terminate condition
    # Then, put the whole infer thing in, try with low p & k
    # Then, try parsing.
    # May use something like: Infer first 3 layers, try, then continue infering next layer w.r.t prob product, trim when width exceed some threshold
    # Update: Top k top p doesn't seem to work well, using torch.multinomial sampling instead
    assert text is not None, 'Specify context!'
    context_tokens = encoder.encode(text)
    context = torch.tensor(context_tokens, device=device, dtype=torch.long).unsqueeze(0)
    
    special_tokens = [
        enc.encode(" <|endoftext|>")[0],
        enc.encode(" <|GOAL|>")[0],
        enc.encode(" <|PROOFSTEP|>")[0],
    ]
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
        values, prev = torch.topk(log_probs, k=top_k, dim=-1)
        p = prev.tolist()
        
        for j in range(top_k):
            text = encoder.decode([p[0][j]])
            value = values[0][j].item()
            outputs.append({
                "text": text,
                "logprobs": {"token_logprobs": [value]},
                "tokens": [p[0][j]],
            })
#             print([p.shape for p in past])
            prevs.append(copy.deepcopy(prev[:, j].unsqueeze(0)))
            pasts.append(copy.deepcopy(past))
#             if text == "<|endoftext|>":
#                 return outputs
        keep_indices = []
        for j in range(len(outputs)): # Magic number
            if prevs[j].tolist()[0][0] in special_tokens:
                keep_indices.append(j)
                continue
            has_end = False
            for i in range(max_length-1):
                logits, pasts[j] = model(prevs[j], past=pasts[j])
                logits = logits[:, -1, :] / temperature
                logits = top_k_logits(logits, k=top_k)
                log_probs = F.softmax(logits, dim=-1)
#                 values, prevs[j] = torch.topk(log_probs, k=1, dim=-1)
                prevs[j] = torch.multinomial(log_probs, num_samples=1)
                # Also need to trim after <|endoftext|>
                p = prevs[j].tolist()
                if p[0][0] in special_tokens:
                    has_end = True
                    break
                text = encoder.decode(p[0])
                value = values[0][0].item()
                outputs[j]["text"] += text
                outputs[j]["tokens"].append(p[0][0])
#             print(outputs)
            if has_end:
                keep_indices.append(j)
#                 outputs[j]["logprobs"].append(value)
#     prev = context
#     output = context
#     past = None
#     has_end = False
#     with torch.no_grad():
#         for i in range(max_length):
#             logits, past = model(prev, past=past)
#             logits = logits[:, -1, :] / temperature
#             logits = top_k_logits(logits, k=top_k)
#             log_probs = F.softmax(logits, dim=-1)
#             if True:
#                 prev = torch.multinomial(log_probs, num_samples=1)
#             else:
#                 _, prev = torch.topk(log_probs, k=1, dim=-1)
#             if prev.tolist()[0][0] in special_tokens:
#                 has_end = True
#                 break
#             output = torch.cat((output, prev), dim=1)
#     output = output[:, len(context_tokens):].tolist()
#     outputs = [{
#         "text": enc.decode(output[0]),
#         "logprobs": {"token_logprobs": [0.5]},
#     }]
    outputs = [outputs[j] for j in keep_indices]
    return outputs
    
    
# def _sample_parsesable_sequence(model, encoder, prev, past, max_length=16, toks=[], temperature=1, top_k=1, device='cuda'):
#     if 
    
#     logits, past = model(prev, past=past)
#     logits = logits[:, -1, :] / temperature
#     logits = top_k_logits(logits, k=top_k)
#     log_probs = F.softmax(logits, dim=-1)
#     values, prev = torch.topk(log_probs, k=top_k, dim=-1)
#     p = prev.tolist()
#     ret = []
#     for k in range(top_k): # Will add top_p
#         ret.extend(
#             _sample_parsesable_sequence(model, encoder, prev, past, 
#                                        max_length=max_length, text=text, 
#                                        temperature=temperature, top_k=top_k, 
#                                        device=device)
#         )
#         if len(ret)

def terminated(tokens, max_length, enc):
    special_tokens = [
        enc.encode(" <|endoftext|>")[0],
        enc.encode(" <|GOAL|>")[0],
        enc.encode(" <|PROOFSTEP|>")[0],
    ]
    if tokens[-1] in special_tokens:
        return True
    elif len(tokens) >= max_length:
        return True
    else:
        return False

# def sample_sequence_budgeted(model, encoder, budget=100, max_length=16, text=None, temperature=1, top_k=1, device='cuda'):
#     # terminate if encounter special tokens OR max_depth reached (in which case, auto eliminate)
#     # infer -> top_p top_k -> p predictions
#     # pass down dfs(), put result into list of predictions
#     # if length list > 100 or so, try parsing & eliminate un-parseable candidates
#     # if parseable > 30, return first 30 (best candidates)
#     # return list
#     # TODO: Try top_p top_k first, then try writing terminate condition
#     # Then, put the whole infer thing in, try with low p & k
#     # Then, try parsing.
#     # May use something like: Infer first 3 layers, try, then continue infering next layer w.r.t prob product, trim when width exceed some threshold
#     assert text is not None, 'Specify context!'
#     context = torch.tensor(encoder.encode(text), device=device, dtype=torch.long).unsqueeze(0)
# #     print(f"Context:{text}, encoded:{context}")
#     prev = context
#     tokens = []
#     past = None
#     with torch.no_grad():
#         logits, past = model(prev, past=past)
#         logits = logits[:, -1, :] / temperature
#         logits = top_k_logits(logits, k=top_k)
#         log_probs = F.softmax(logits, dim=-1)
#         values, prev = torch.topk(log_probs, k=top_k, dim=-1)
#         p = prev.tolist()
        
#         used_budget = 0
#         outputs = []
#         for i in range(top_k):
#             toks = copy.deepcopy(tokens)
#             toks.append(p[0][i]) # ith best token
#             probability = values[0][i].item()
#             current_budget = round(probability * budget)
#             used_budget += current_budget
#             if used_budget >= budget or current_budget == 0:
#                 break

# #             print(f"Spending {current_budget}/{budget} on {encoder.decode([toks[-1]])} with probability {probability:.3f}")
#             outputs.extend(_sample_sequence_budgeted(model, encoder, 
#                                                     copy.deepcopy(prev), copy.deepcopy(past), toks, budget=current_budget, 
#                                                     max_length=max_length, 
#                                                     temperature=temperature, 
#                                                     top_k=top_k, 
#                                                     device=device))
#     return outputs


# def _sample_sequence_budgeted(model, encoder, 
#                               prev, past, tokens, budget=100, 
#                               max_length=16, temperature=1, top_k=1, device='cuda'):
#     if terminated(tokens, max_length, encoder): # Based on tokens
#         # Return empty if tokens not terminated before max_length
#         # Else return without terminating token
#         # TODO: Return a dictionary with text and logprobs instead
#         return [{
#             "text": encoder.decode(tokens[:-1]),    
#         }] # Exclude last token
        
#     logits, past = model(prev, past=past)
#     logits = logits[:, -1, :] / temperature
#     logits = top_k_logits(logits, k=top_k)
#     log_probs = F.softmax(logits, dim=-1)
#     values, prev = torch.topk(log_probs, k=top_k, dim=-1)
#     p = prev.tolist()
#     outputs = []
    
#     # Simply choose most confident token
#     if budget == 1:
#         tokens.append(p[0][0])
#         outputs.extend(_sample_sequence_budgeted(model, encoder, 
#                                                 copy.deepcopy(prev), copy.deepcopy(past), tokens, budget=budget, 
#                                                 max_length=max_length, 
#                                                 temperature=temperature, 
#                                                 top_k=top_k, 
#                                                 device=device))
#         return outputs
    
#     used_budget = 0
#     for i in range(top_k):
#         toks = copy.deepcopy(tokens)
#         toks.append(p[0][i]) # ith best token
#         probability = values[0][i].item()
#         current_budget = round(probability * budget)
#         used_budget += current_budget
#         if used_budget >= budget or current_budget == 0:
#             break
        
# #         print(f"Spending {current_budget}/{budget} on {encoder.decode([toks[-1]])} with probability {probability:.3f}")
#         outputs.extend(_sample_sequence_budgeted(model, encoder, 
#                                                 copy.deepcopy(prev), copy.deepcopy(past), toks, budget=current_budget, 
#                                                 max_length=max_length, 
#                                                 temperature=temperature, 
#                                                 top_k=top_k, 
#                                                 device=device))
#     return outputs

def sample_sequence_budgeted(model, encoder, budget=50, max_length=16, text=None, temperature=1, top_k=1, device='cuda'):
    # terminate if encounter special tokens OR max_depth reached (in which case, auto eliminate)
    # infer -> top_p top_k -> p predictions
    # pass down dfs(), put result into list of predictions
    # if length list > 100 or so, try parsing & eliminate un-parseable candidates
    # if parseable > 30, return first 30 (best candidates)
    # return list
    # TODO: Try top_p top_k first, then try writing terminate condition
    # Then, put the whole infer thing in, try with low p & k
    # Then, try parsing.
    # May use something like: Infer first 3 layers, try, then continue infering next layer w.r.t prob product, trim when width exceed some threshold
    # Update: Top k top p doesn't seem to work well, using torch.multinomial sampling instead
    assert text is not None, 'Specify context!'
    context_tokens = encoder.encode(text)
    context = torch.tensor(context_tokens, device=device, dtype=torch.long).unsqueeze(0)
    
    special_tokens = [
        enc.encode(" <|endoftext|>")[0],
        enc.encode(" <|GOAL|>")[0],
        enc.encode(" <|PROOFSTEP|>")[0],
    ]
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
        values, prev = torch.topk(log_probs, k=top_k, dim=-1)
        p = prev.tolist()
        
        used_budget = 0
        for i in range(top_k):            
            probability = values[0][i].item()
            current_budget = round(probability * budget)
            used_budget += current_budget
            if used_budget >= budget or current_budget == 0:
                break
            
            text = encoder.decode([p[0][i]])
#             print(f"Spending {current_budget:3d}/{budget:3d} on {text:6s} with probability {probability:.3f}")
            for j in range(current_budget):
                outputs.append({
                    "text": text,
                    "logprobs": {"token_logprobs": [probability]},
#                     "tokens": [p[0][i]],
                })
                prevs.append(copy.deepcopy(prev[:, i].unsqueeze(0)))
                pasts.append(copy.deepcopy(past))

        keep_indices = []
        for i in range(len(outputs)): # Magic number
            if prevs[i].tolist()[0][0] in special_tokens:
                keep_indices.append(i)
                continue
            has_end = False
            for j in range(max_length-1):
                logits, pasts[i] = model(prevs[i], past=pasts[i])
                logits = logits[:, -1, :] / temperature
                logits = top_k_logits(logits, k=top_k)
                log_probs = F.softmax(logits, dim=-1)
#                 values, prevs[i] = torch.topk(log_probs, k=1, dim=-1)
                prevs[i] = torch.multinomial(log_probs, num_samples=1)
                # Also need to trim after <|endoftext|>
                p = prevs[i].tolist()
                if p[0][0] in special_tokens:
                    has_end = True
                    break
                text = encoder.decode(p[0])
                value = values[0][0].item()
                outputs[i]["text"] += text
#                 outputs[i]["tokens"].append(p[0][0])
#             print(outputs)
            if has_end:
                keep_indices.append(i)
    outputs = [outputs[i] for i in keep_indices]
    return outputs
    
def check_parseable(l):
    if len(l) == 0:
        return []
    
    home = expanduser("~")
    os.chdir(os.path.join(home, 'pj/lean-tpe-public/'))
    lean_prog = os.path.join(home, 'pj/lean-tpe-public/src/try_parse.lean')
    lean_exec = os.path.join(home, ".elan/bin/lean")
    cmds = [lean_exec, "--run", lean_prog] + l
    # print(cmds)
    test = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = test.communicate()
        # print("STDOUT:", output)
        # print("STDERR:", output)
#     print(output[0])
#     print("================="*10)
    lines = output[1].decode("utf8").strip().split("\n")
#     print(lines)
    ret = []
    for l in lines:
#         print(l[6])
        if l[6] == "F":
            ret.append(False)
        elif l[6] == "S":
            ret.append(True)
        else:
            raise Exception(f"Wrong output format for try_parsing! Got: {l}")
    return ret

import torch
import torch.nn.functional as F
from tqdm import trange


if __name__ == "__main__":
    
    args = _parse()
#     print(args)
    
    output_dir="/home/tst008/pj/lean-winter" # TODO: move to ~/data folder
    output_prefix="proofstep-named"
    
    config = GPT2Config()
    config.vocab_size += 2 # 2 new tokens
    model, current_epoch, _, _ = load_checkpoint(config, output_dir, output_prefix)
#     print(type(model) == GPT2LMHeadModel)

    device = torch.device("cuda")
    model = model.cuda()
    model.eval()
    
    if "<|GOAL|>" not in args.prompt:
        args.prompt = json.loads(args.prompt)
        prompt = args.prompt["prompt"].replace("[LN]", "").replace("GOAL", "<|GOAL|>").replace("PROOFSTEP", "<|PROOFSTEP|>").rstrip(" ")
    else:
        prompt = args.prompt
    
    enc = get_encoder()
    outputs = []
    num_parseable = 0
    count = 0
    while len(outputs) < 20:
        current = []
        for i in range(1):
            o = sample_sequence_budgeted(model, enc, budget=50, max_length=32, text=prompt, temperature=args.temperature, top_k=50)
            current.extend(o)
#         print(f"Found {len(current)}. Checking if parseable..", current)
        l = check_parseable([c["text"] for c in current])
        outputs.extend([current[i] for i in range(len(l)) if l[i] if current[i] not in outputs])
        print(json.dumps({
            "model": "gptf-neo",
#             "prompt": prompt,
#             "a_unicode": u"♠",
            "choices": outputs,
        }, ensure_ascii=False))
#         print(f"Total: {len(outputs)} parseable.")
#         print([o["text"] for o in outputs])
#         print(outputs)
#     args.temperature = 0.9
#     print(f"Prompt:", prompt)
#     prompt = " <|GOAL|>α : Type u,\n_inst_1 : linear_order α,\na b : α\n⊢ linear_order.min a b ≤ b <|PROOFSTEP|>"
#     print(f"Prompt:", prompt)
#     while len(outputs) < 10:
#         current = []
#         while len(current) < 100:
#             current.extend([x for x in sample_sequence_budgeted(model, enc, 
#                                       budget=10, max_length=16, text=prompt, temperature=args.temperature, top_k=10)
#                                 if x not in current])
#             print(current)
#         print(f"Found {len(current)}. Checking if parseable..", current)
#         l = check_parseable([c["text"] for c in current])
#         outputs.extend([current[i] for i in range(len(l)) if l[i] if current[i] not in outputs])
#         print(f"Total: {len(outputs)} parseable.", outputs)

#     context_text = " <|GOAL|>α : Type u,\n_inst_1 : linear_order α,\na b : α\n⊢ linear_order.min a b ≤ b <|PROOFSTEP|>" # Answer: 'min_tac a b'
#     context_tokens = enc.encode(context_text)
#     context_text = prompt
#     context_tokens = enc.encode(context_text)
#     print(context_tokens)
#     for x in range(1):
#         out = sample_sequence(
#             model=model, length=16,
#             context=context_tokens,
#             batch_size=1,
#             temperature=args.temperature, 
#             top_k=10, device=device
#         )
#         print(out.shape)
#         out = out[:, len(context_tokens):].tolist()
#         text = enc.decode(out[0])
#         print(f"Sample:{text}")
#         o = sample_parseable_sequence(model, enc, max_length=16, text=prompt, 
#                                       temperature=args.temperature, top_k=10)
#         print(o.shape)
#         o = o[:, len(context_tokens):].tolist()
#         text = enc.decode(o[0])
#         print(f'Sample parseable:{text}')
#         print(f'Sample parseable:{o[0]["text"]}')
    


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

# Try parse lean string in environment
# Auto fill: parentheses, brackets, curly brackets
