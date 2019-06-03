"""
Test models outside the training loop.
"""
from __future__ import division
import logging
import infer
import os
import progressbar
import argparse
from infer import WrappedTagger
from collections import Counter
from evaluate_morphotags import Evaluator
from model import get_word_chars, NONE_TAG, POS_KEY, Instance
import utils

logging.basicConfig(level=logging.INFO)

def evaluate(wrapped_model, instances, log_dir_name = None, use_bar=False):
    """Evaluate the given model and output statistics about the results

    :param wrapped_model: a WrappedModel to test
    :param instances: the instances to evaluate on
    :param log_dir_name: optional, a directory to log results to
    :param use_bar: optional, show a progress bar
    """
    w2i, t2is, c2i, i2w, i2ts, i2c, \
    training_instances, training_vocab, \
    dev_instances, dev_vocab, test_instances, tag_set_sizes = wrapped_model.dataset.get_all_params()


    logging.info("\n")
    logging.info("Number instances: {}".format(len(instances)))
    wrapped_model.model.disable_dropout()
    test_correct = Counter()
    test_total = Counter()
    test_oov_total = Counter()
    bar = progressbar.ProgressBar()
    total_wrong = Counter()
    total_wrong_oov = Counter()
    f1_eval = Evaluator(m = 'att')


    test_outputs = []


    for instance in (bar(instances) if use_bar else instances):
        if len(instance.sentence) == 0: continue
        gold_tags = instance.tags
        for att in wrapped_model.model.attributes:
            if att not in instance.tags:
                gold_tags[att] = [t2is[att][NONE_TAG]] * len(instance.sentence)
        word_chars = get_word_chars(instance.sentence, i2w, c2i)
        out_tags_set, _ = wrapped_model.model.tag_sentence(word_chars)

        gold_strings = utils.morphotag_strings(i2ts, gold_tags)
        obs_strings = utils.morphotag_strings(i2ts, out_tags_set)
        for g, o in zip(gold_strings, obs_strings):
            f1_eval.add_instance(utils.split_tagstring(g, has_pos=True), utils.split_tagstring(o, has_pos=True))
        for att, tags in gold_tags.items():
            out_tags = out_tags_set[att]

            oov_strings = []
            for word, gold, out in zip(instance.sentence, tags, out_tags):
                if gold == out:
                    test_correct[att] += 1
                else:
                    # Got the wrong tag
                    total_wrong[att] += 1
                    if i2w[word] not in training_vocab:
                        total_wrong_oov[att] += 1

                if i2w[word] not in training_vocab:
                    test_oov_total[att] += 1
                    oov_strings.append("OOV")
                else:
                    oov_strings.append("")

            test_total[att] += len(tags)
        test_outputs.append(("\n"
                         + "\n".join(["\t".join(z) for z in zip([i2w[w] for w in instance.sentence],
                                                                     gold_strings, obs_strings, oov_strings)])
                         + "\n").encode('utf8'))

    if log_dir_name:
        with open("{}/testout.txt".format(log_dir_name), 'w') as test_writer:
            for output in test_outputs:
                test_writer.write(output)


    # log test results
    logging.info("POS Test Accuracy: {}".format(test_correct[POS_KEY] / test_total[POS_KEY]))
    logging.info("POS % Test OOV accuracy: {}".format((test_oov_total[POS_KEY] - total_wrong_oov[POS_KEY]) / test_oov_total[POS_KEY]))
    if total_wrong[POS_KEY] > 0:
        logging.info("POS % Test Wrong that are OOV: {}".format(total_wrong_oov[POS_KEY] / total_wrong[POS_KEY]))
    for attr in t2is.keys():
        if attr != POS_KEY:
            logging.info("{} F1: {}".format(attr, f1_eval.mic_f1(att = attr)))
    logging.info("Total attribute F1s: {} micro, {} macro, POS included = {}".format(f1_eval.mic_f1(), f1_eval.mac_f1(), False))

    logging.info("Total test tokens: {}, Total test OOV: {}, % OOV: {}".format(test_total[POS_KEY], test_oov_total[POS_KEY], test_oov_total[POS_KEY] / test_total[POS_KEY]))

    results = {
                'pos_acc' : (test_correct[POS_KEY] / test_total[POS_KEY]),
                'pos_oov_accuracy' : (test_oov_total[POS_KEY] - total_wrong_oov[POS_KEY]) / test_oov_total[POS_KEY],
                'pos_wrong_oov' : (total_wrong_oov[POS_KEY] / total_wrong[POS_KEY])
            }

    return results

if __name__ == '__main__':


    # ===-----------------------------------------------------------------------===
    # Argument parsing
    # ===-----------------------------------------------------------------------===
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
    parser.add_argument("--model",  required=True, dest="model", help="saved model to analyze")
    parser.add_argument("--dev", action="store_true", dest="dev", help="eval on dev data")
    parser.add_argument("--train", action="store_true", dest="train", help="eval on train data")
    parser.add_argument("--test", action="store_true", dest="test", help="eval on test data")

    parser.add_argument("--hidden-dim", default=128, dest="hidden_dim", type=int, help="Size of LSTM hidden layers (default - 128)")
    parser.add_argument("--word-level-dim", default=None, dest="word_level_dim", type=int, help="Size of the word level LSTM hidden layers (defaults to --hidden-dim)")
    parser.add_argument("--forward-dim", default = None, dest="forward_dim", type=int, help="Number of forward units in character rnn")
    parser.add_argument("--backward-dim", default = None, dest="backward_dim", type=int, help="Number of backward units in character rnn")
    options = parser.parse_args()

    print(options.forward_dim, options.backward_dim)
    if options.forward_dim is not None and options.backward_dim is not None:
        print("Using asym")
        options.hidden_dim = (options.forward_dim, options.backward_dim)

    if not options.word_level_dim:
        if type(options.hidden_dim) == int:
            options.word_level_dim = options.hidden_dim
        else:
            #if we are using an asym bilstm
            options.word_level_dim = sum(options.hidden_dim)

    model = WrappedTagger(options.dataset,
                num_lstm_layers = 2,
                hidden_dim = options.hidden_dim,
                char_embedding_dim = 256,
                word_level_dim = options.word_level_dim)

    model.load_weights(options.model, verbose=True)


    if options.train:
        instances = model.dataset.training_instances
    elif options.dev:
        instances = model.dataset.dev_instances
    elif options.test:
        instances = model.dataset.test_instances
    else:
        raise Exception("Must provide one of --train,dev,test")

    evaluate(model, instances, use_bar=True)

