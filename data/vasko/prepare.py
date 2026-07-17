"""
process out.txt file that is exported from nanongram
"""
import random
import os 
import requests 
import tiktoken 
import numpy as np

def _tokenize(enc, text: str, lower: bool = True) -> list[int]: 
        text = text.lower() if lower else text
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)
        return ids 

def _pad(ids: list[str], max_tokens: int = 128, pad: int = 50257) -> list[int] : 
                                                         # ^^^^^ - <|endoftext|> is 50256
    split_ids = []
    for i in range(0, len(ids), max_tokens): 
        ids_ = ids[i:i+max_tokens] 
        n_ids = len(ids_)
        if n_ids < max_tokens:
            ids_ = ids_ + ([pad] * (max_tokens - n_ids)) 
        split_ids.extend(ids_)
    return split_ids 

if __name__ == '__main__': 
    cur_dir = os.path.dirname(__file__)

    input_file_path = os.path.join(cur_dir, 'input.txt')
    if not os.path.exists(input_file_path): 
        raise IOError("manually copy paste in this directory nanongram .txt output")

    with open(input_file_path, 'r', encoding='utf-8') as f: 
        # nanogram preprocessing and cleaning 
        lines = f.readlines() 
        for i, line in enumerate(lines):
            old_start_token, old_end_token = '<s>', '</s>' 
            lines[i] = line.replace(old_start_token, '')\
                           .replace(old_end_token, '')\
                           .strip()
    
    # TODO in future: train and use my own tokenizer 
    enc = tiktoken.get_encoding("gpt2") 
    
    # shuffle, tokenize, split, report 
    split_point = 0.9
    seed = 2004
    random.seed(seed)
    random.shuffle(lines) 
    
    ids = []
    for line in lines: ids.extend(_pad(_tokenize(enc, line)))

    train_ids = ids[:int(len(ids)*split_point)]
    val_ids = ids[int(len(ids)*split_point):]
    
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to numpy 
    # np.uint16 fits cause: 50304 < 2^16 (65536)
    np_train_ids = np.array(train_ids, dtype=np.uint16) 
    np_val_ids = np.array(val_ids, dtype=np.uint16)

    np_train_ids.tofile(os.path.join(cur_dir, 'train.bin'))
    np_val_ids.tofile(os.path.join(cur_dir, 'val.bin'))
