import os
from collections import defaultdict, Counter

def get_all_pos(dataset, instances, freq_threshold, unambig_threshold):
    """get_all_pos: returns a map from a POS to a list of words, if those
       words are unambiguously that part-of-speech and occur frequently.

    :param dataset: a ProcessedDataset for the language
    :param instances: instances (e.g. dataset.{training, test, dev}_instances)
    :param freq_threshold: only process words that occur this many times
    :param unambig_threshold: only return words that are a given POS x% of the time
    """
    words_to_tags =  defaultdict(Counter)
    i2pos = dataset.i2ts['POS']
    for instance in instances:
        for word, tag in zip(instance.sentence, instance.tags['POS']):
            word = dataset.i2w[word]
            tag = i2pos[tag]
            words_to_tags[word][tag] += 1

    words = defaultdict(list)
    for word, tags in words_to_tags.items():
        count = sum(tags.values())

        if count < freq_threshold:
            continue

        for tag, tag_count in tags.items():
            if tag_count / count >= unambig_threshold:
                words[tag].append(word)

    return words

def get_best_model_name(directory):
    """
    :param directory: search this directory for a best model filename
    """
    print(directory)
    filenames = [f for f in os.listdir(directory) if 'best' in f and f.endswith('.bin')]
    if len(filenames) != 1:
        raise Exception("Found multiple matches: {}".format(filenames))
    return os.path.join(directory, filenames[0])
