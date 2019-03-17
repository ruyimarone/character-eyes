def get_all_pos(dataset, instances, freq_threshold, unambig_threshold):
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
