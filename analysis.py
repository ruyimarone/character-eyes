from __future__ import division
import numpy as np
import json
import argparse
import codecs
import os
from collections import namedtuple, defaultdict
from infer import WrappedTagger
from model import Instance

INTEREST_POSS = ['NOUN']
ALL_POSS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'AUX', 'CCONJ', 'DET', 'PART', 'PRON', 'SCONJ']

def bins(activations, nbins, rng):
    return np.histogram(activations, bins=nbins, range=rng)[0]

def pmi(cat_vals, nbins=16, rng=(0,1)):
    """
    :param cat_vals: dictionary from cat name to raw values (need binning)
    :returns: \sum_{T} \sum_{B} P(t, b) (\log P(t, b) - \log P(t) - \log P(b))
    """
    joints = np.array([bins(v, nbins, rng) for c,v in cat_vals.items()]).astype(float)
    joints /= joints.sum()
    j_logs = np.ma.log(joints).filled(0)
    b_margin_logs = np.ma.log(joints.sum(axis=0)).filled(0)
    t_margin_logs = np.ma.log(joints.sum(axis=1)).filled(0)
    logs_in_pmi = ((j_logs - b_margin_logs).transpose() - t_margin_logs).transpose()
    return (joints * logs_in_pmi).sum()

def pmi_macro_abs_avg(activations, words_to_pos, all_poss=ALL_POSS):
    """
    :param words_to_pos: dictionary from word type to tagged POS
    """
    postv_values = defaultdict(list)
    for word, pos in words_to_pos.items():
        if (all_poss is not None and pos not in all_poss) or word not in activations or len(word) < 2:
            continue
        chars = activations[word]
        avg_abs = abs(chars).mean(axis=0)
        postv_values[pos].append(avg_abs)

    postv_np_values = {p:np.array(vals).transpose() for p,vals in postv_values.items()}
    dim = len(postv_np_values['NOUN'])
    unitwise_values = [pmi({p:vals[i] for p,vals in postv_np_values.items()}) for i in range(dim)]
    return np.array(unitwise_values)

def pmi_max_diff(activations, words_to_pos, all_poss=ALL_POSS):
    """
    :param words_to_pos: dictionary from word type to tagged POS
    """
    postv_values = defaultdict(list)
    for word, pos in words_to_pos.items():
        if (all_poss is not None and pos not in all_poss) or word not in activations or len(word) < 2:
            continue
        chars = abs(activations[word][1:]-activations[word][:-1])
        if len(chars) < 1: continue # don't know why this happens in e.g. swedish, but it does
        max_diff = chars.max(axis=0)
        postv_values[pos].append(max_diff)

    postv_np_values = {p:np.array(vals).transpose() for p,vals in postv_values.items()}
    dim = len(postv_np_values['NOUN'])
    unitwise_values = [pmi({p:vals[i] for p,vals in postv_np_values.items()}, nbins=32, rng=(0,2)) for i in range(dim)]
    return np.array(unitwise_values)

### END PMI ###

def kldiv(cat_vals, all_vals, rng=(-1,1)):
    N = len(all_vals)
    M = len(cat_vals)
    all_hist = bins(all_vals, rng)/N
    cat_hist = bins(cat_vals, rng)/M
    return sum([c * np.log2(c/a) for a, c in zip(all_hist, cat_hist) if c > 0])

def kl_final_acts(activations, words_to_pos, postv_poss=INTEREST_POSS, all_poss=ALL_POSS, fwd_dim=64):
    """
    :param words_to_pos: dictionary from word type to tagged POS
    """
    all_values = []
    postv_values = []
    for word, pos in words_to_pos.items():
        if pos not in all_poss or word not in activations:
            continue
        chars = activations[word]
        acts = np.concatenate([chars[-1][:fwd_dim], chars[0][fwd_dim:]])
        all_values.append(acts)
        if pos in postv_poss:
            postv_values.append(acts)

    all_values = np.array(all_values).transpose()
    postv_values = np.array(postv_values).transpose()
    return np.array([kldiv(postv_values[i], all_values[i]) for i in range(len(all_values))])

def kl_max_diff(activations, words_to_pos, postv_poss=INTEREST_POSS, all_poss=ALL_POSS):
    """
    :param words_to_pos: dictionary from word type to tagged POS
    """
    all_values = []
    postv_values = []
    for word, pos in words_to_pos.items():
        if pos not in all_poss or word not in activations or len(word) < 2:
            continue
        chars = abs(activations[word][1:]-activations[word][:-1])
        if len(chars) < 1: continue # don't know why this happens in e.g. swedish, but it does
        max_diff = chars.max(axis=0)
        all_values.append(max_diff)
        if pos in postv_poss:
            postv_values.append(max_diff)

    all_values = np.array(all_values).transpose()
    postv_values = np.array(postv_values).transpose()
    return np.array([kldiv(postv_values[i], all_values[i], rng=(0,2)) for i in range(len(all_values))])

### END KL ###

def average_activation(activations, words):
    # micro-averaged over all characters
    values = []
    n_chars = sum(len(w) for w in words)
    for word in words:
        chars = activations[word]
        for char in chars:
            values.append(char)

    values = np.array(values)
    return values.mean(axis=0)

def average_abs_activation(activations, words):
    # micro-averaged over all characters
    values = []
    n_chars = sum(len(w) for w in words)
    for word in words:
        chars = activations[word]
        for char in chars:
            values.append(abs(char))

    values = np.array(values)
    return values.mean(axis=0)

def mac_avg_activation(activations, words):
    # macro-averaged across words
    values = []
    for word in words:
        chars = activations[word]
        values.append(chars.mean(axis=0))

    values = np.array(values)
    return values.mean(axis=0)

def mac_avg_abs_activation(activations, words):
    # macro-averaged across words
    values = []
    for word in words:
        chars = activations[word]
        values.append(abs(chars).mean(axis=0))

    values = np.array(values)
    return values.mean(axis=0)

def max_abs_activation(activations, words):
    values = []
    for word in words:
        chars = abs(activations[word])
        values.append(chars.max(axis=0))

    values = np.array(values)
    return values.mean(axis=0)

def max_jump_activation(activations, words):
    values = []
    n_chars = sum(len(w) for w in words)
    for word in words:
        if len(word) < 2: continue
        chars = abs(activations[word][1:]-activations[word][:-1])
        values.append(chars.max(axis=0))

    values = np.array(values)
    return values.mean(axis=0)

### END BASE_FN ###

def avg_final_activation(activations, words, fwd_dim=64):
    values = []
    for word in words:
        chars = activations[word]
        acts = np.concatenate([chars[-1][:fwd_dim], chars[0][fwd_dim:]])
        values.append(acts)

    values = np.array(values)
    return values.mean(axis=0)

def average_diff(activations, words):
    values = []
    n_chars = sum(len(w) for w in words)
    for word in words:
        chars = activations[word]
        for first, second in zip(chars, chars[1:]):
            values.append(abs(first - second))

    values = np.array(values)
    return values.mean(axis=0)

def equal_resample(*samples):
    min_words = min(len(x) for x in samples)
    outs = []
    for sample in samples:
        outs.append(list(np.random.choice(sample, min_words, replace=False)))
    return outs

def get_single_activation(model, word):
    _, embeddings = model.forward_text(word)
    activations = embeddings[0][1:-1]
    activations = np.stack([a.npvalue() for a in activations], axis=1)
    return activations.T

def get_head(sorted_values, p_mass = 0.5):
    """Finds the N values needed to account for a certain amount of "mass"
    :param sorted_values: inverse sorted (largest to smallest) values
    :param p_mass: the mass of the head
    """
    total = sum(abs(sorted_values))
    head = p_mass * total
    return np.argmax(np.cumsum(sorted_values)>=head)
