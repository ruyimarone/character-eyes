'''
Main application script for tagging parts-of-speech and morphosyntactic tags. Run with --help for command line arguments.
'''

from collections import Counter
from _collections import defaultdict
from evaluate_morphotags import Evaluator

import collections
import argparse
import random
import pickle
import logging
import progressbar
import os
import sys
import dynet_config
import numpy as np

import utils

__author__ = "Yuval Pinter and Robert Guthrie, 2017. Modified Marc Marone, 2018"

Instance = collections.namedtuple("Instance", ["sentence", "tags"])

NONE_TAG = "<NONE>"
POS_KEY = "POS"
PADDING_CHAR = "<*>"

DEFAULT_WORD_EMBEDDING_SIZE = 64
DEFAULT_CHAR_EMBEDDING_SIZE = 256

normalized_words = collections.Counter()

class LSTMTagger:
    '''
    Joint POS/morphosyntactic attribute tagger based on LSTM.
    Embeddings are fed into Bi-LSTM layers, then hidden phases are fed into an MLP for each attribute type (including POS tags).
    Class "inspired" by Dynet's BiLSTM tagger tutorial script available at:
    https://github.com/clab/dynet_tutorial_examples/blob/master/tutorial_bilstm_tagger.py
    '''

    def __init__(self, tagset_sizes, num_lstm_layers, hidden_dim, word_level_dim, charset_size, char_embedding_dim, vocab_size=None, word_embedding_dim=None):
        '''
        :param tagset_sizes: dictionary of attribute_name:number_of_possible_tags
        :param num_lstm_layers: number of desired LSTM layers
        :param hidden_dim: size of hidden dimension (same for all LSTM layers, including character-level). If tuple, the char level birnn will be asymmetric with (forward, backward)
        :param word_embeddings: pre-trained list of embeddings, assumes order by word ID (optional)
        :param charset_size: number of characters expected in dataset (needed for character embedding initialization)
        :param char_embedding_dim: desired character embedding dimension
        :param vocab_size: number of words in model (ignored if pre-trained embeddings are given)
        :param word_embedding_dim: desired word embedding dimension (ignored if pre-trained embeddings are given)
        '''
        self.model = dy.Model()
        self.tagset_sizes = tagset_sizes
        self.attributes = tagset_sizes.keys()

        # Char LSTM Parameters
        self.char_lookup = self.model.add_lookup_parameters((charset_size, char_embedding_dim), name="ce")
        logging.info("char bilstm: char_embedding_dim {} hidden {}".format(char_embedding_dim, hidden_dim))
        self.char_bi_lstm = AsymBiRNNBuilder(1, char_embedding_dim, hidden_dim, self.model, dy.LSTMBuilder)

        if  type(hidden_dim) == int:
            input_dim = hidden_dim
        else:
            input_dim = sum(hidden_dim)



        logging.info("word bilstm: input_dim {} word_level_dim {}".format(input_dim, word_level_dim))
        self.word_bi_lstm = dy.BiRNNBuilder(num_lstm_layers, input_dim, word_level_dim, self.model, dy.LSTMBuilder)

        # Matrix that maps from Bi-LSTM output to num tags
        self.lstm_to_tags_params = {}
        self.lstm_to_tags_bias = {}
        self.mlp_out = {}
        self.mlp_out_bias = {}
        for att, set_size in tagset_sizes.items():
            self.lstm_to_tags_params[att] = self.model.add_parameters((set_size, word_level_dim), name=att+"H")
            self.lstm_to_tags_bias[att] = self.model.add_parameters(set_size, name=att+"Hb")
            self.mlp_out[att] = self.model.add_parameters((set_size, set_size), name=att+"O")
            self.mlp_out_bias[att] = self.model.add_parameters(set_size, name=att+"Ob")

    def word_rep(self, char_ids):
        '''
        :param word: index of word in lookup table
        '''

        # only char representation, no word embeddings
        char_embs = [self.char_lookup[cid] for cid in char_ids]
        return self.char_bi_lstm.final_hiddens(char_embs)

    def build_tagging_graph(self, word_chars):
        dy.renew_cg()

        all_embeddings = [self.word_rep(chars) for chars in word_chars]
        embeddings, char_embeddings = list(zip(*all_embeddings))
        lstm_out = self.word_bi_lstm.transduce(embeddings)

        scores = {}
        for att in self.attributes:
            H  = self.lstm_to_tags_params[att]
            Hb = self.lstm_to_tags_bias[att]
            O = self.mlp_out[att]
            Ob = self.mlp_out_bias[att]
            scores[att] = []
            for rep in lstm_out:
                score_t = O * dy.tanh(H * rep + Hb) + Ob
                scores[att].append(score_t)

        return scores, char_embeddings

    def loss(self, word_chars, tags_set):
        '''
        For use in training phase.
        Tag sentence (all attributes) and compute loss based on probability of expected tags.
        '''
        observations_set, _ = self.build_tagging_graph(word_chars)
        errors = {}
        for att, tags in tags_set.items():
            err = []
            for obs, tag in zip(observations_set[att], tags):
                err_t = dy.pickneglogsoftmax(obs, tag)
                err.append(err_t)
            errors[att] = dy.esum(err)
        return errors

    def tag_sentence(self, word_chars):
        '''
        For use in testing phase.
        Tag sentence and return tags for each attribute, without caluclating loss.
        '''
        observations_set, char_embeddings = self.build_tagging_graph(word_chars)
        tag_seqs = {}
        for att, observations in observations_set.items():
            observations = [ dy.softmax(obs) for obs in observations ]
            probs = [ obs.npvalue() for obs in observations ]
            tag_seq = []
            for prob in probs:
                tag_t = np.argmax(prob)
                tag_seq.append(tag_t)
            tag_seqs[att] = tag_seq
        return tag_seqs, char_embeddings

    def set_dropout(self, p):
        self.word_bi_lstm.set_dropout(p)

    def disable_dropout(self):
        self.word_bi_lstm.disable_dropout()

    def save(self, file_name):
        '''
        Serialize model parameters for future loading and use.
        TODO change reading in scripts/test_model.py
        '''
        self.model.save(file_name)

        with open(file_name + "-atts", 'w') as attdict:
            attdict.write("\t".join(sorted(self.attributes)))


class ProcessedDataset:
    def __init__(self, dataset_path, training_sentence_size=None, token_size=None):
        dataset = pickle.load(open(dataset_path, "rb"))
        self.dataset = dataset

        self.w2i = dataset["w2i"]
        self.t2is = dataset["t2is"]
        self.c2i = dataset["c2i"]
        self.i2w = { i: w for w, i in self.w2i.items()} # Inverse mapping
        self.i2ts = { att: {i: t for t, i in t2i.items()} for att, t2i in self.t2is.items()}
        self.i2c = { i: c for c, i in self.c2i.items()}

        self.training_instances = dataset["training_instances"]
        self.training_vocab = dataset["training_vocab"]
        self.dev_instances = dataset["dev_instances"]
        self.dev_vocab = dataset["dev_vocab"]
        self.test_instances = dataset["test_instances"]

        # trim training set for size evaluation (sentence based)
        if training_sentence_size is not None and len(self.training_instances) > training_sentence_size:
            random.shuffle(self.training_instances)
            self.training_instances = self.training_instances[:training_sentence_size]

        # trim training set for size evaluation (token based)
        training_corpus_size = sum(self.training_vocab.values())
        if token_size is not None and training_corpus_size > token_size:
            random.shuffle(self.training_instances)
            cumulative_tokens = 0
            cutoff_index = -1
            for i,inst in enumerate(self.training_instances):
                cumulative_tokens += len(inst.sentence)
                if cumulative_tokens >= token_size:
                    self.training_instances = self.training_instances[:i+1]
                    break

        self.tag_set_sizes = {att: len(t2i) for att, t2i in list(self.t2is.items())}

    def get_all_params(self):
        """
        Simplify model wrapping
        """
        return self.w2i, self.t2is, self.c2i, \
                self.i2w, self.i2ts, self.i2c, \
                self.training_instances, self.training_vocab, \
                self.dev_instances, self.dev_vocab, \
                self.test_instances, self.tag_set_sizes

    def instance_to_sentence(self, instance):
        return ' '.join([self.i2w[i] for i in instance.sentence])



### END OF CLASSES ###

def normalize(word):
    if 'http' in word:
        normalized_words[word] += 1
        return 'URL'
    elif '@' in word:
        normalized_words[word] += 1
        return 'EMAIL'
    return word

def get_word_chars(sentence, i2w, c2i):
    """get_word_chars: gets the character index level representation of a sentence,
    prior to processing with the character level RNN

    :param sentence: a list of word indices
    :param i2w: index to word mappings
    :param c2i: character to index mappings
    """
    pad_char = c2i[PADDING_CHAR]
    return [[pad_char] + [c2i[c] for c in normalize(i2w[word])] + [pad_char] for word in sentence]

if __name__ == "__main__":
    print("setting python internal rand seed=1")
    random.seed(1)

    # ===-----------------------------------------------------------------------===
    # Argument parsing
    # ===-----------------------------------------------------------------------===
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
    parser.add_argument("--num-epochs", default=20, dest="num_epochs", type=int, help="Number of full passes through training set (default - 20)")
    parser.add_argument("--num-lstm-layers", default=2, dest="lstm_layers", type=int, help="Number of LSTM layers (default - 2)")
    parser.add_argument("--hidden-dim", default=128, dest="hidden_dim", type=int, help="Size of LSTM hidden layers (default - 128)")
    parser.add_argument("--word-level-dim", default=None, dest="word_level_dim", type=int, help="Size of the word level LSTM hidden layers (defaults to --hidden-dim)")
    parser.add_argument("--forward-dim", default = None, dest="forward_dim", type=int, help="Number of forward units in character rnn")
    parser.add_argument("--backward-dim", default = None, dest="backward_dim", type=int, help="Number of backward units in character rnn")
    parser.add_argument("--training-sentence-size", default=None, dest="training_sentence_size", type=int, help="Instance count of training set (default - unlimited)")
    parser.add_argument("--token-size", default=None, dest="token_size", type=int, help="Token count of training set (default - unlimited)")
    parser.add_argument("--learning-rate", default=0.01, dest="learning_rate", type=float, help="Initial learning rate (default - 0.01)")
    parser.add_argument("--dropout", default=-1, dest="dropout", type=float, help="Amount of dropout to apply to LSTM part of graph (default - off)")
    parser.add_argument("--log-dir", default="log", dest="log_dir", help="Directory where to write logs / serialized models")
    parser.add_argument("--no-model", dest="no_model", action="store_true", help="Don't serialize models")
    parser.add_argument("--dynet-mem", help="Ignore this external argument")
    parser.add_argument("--debug", dest="debug", action="store_true", help="Debug mode")
    parser.add_argument("--log-to-stdout", dest="log_to_stdout", action="store_true", help="Log to STDOUT")
    parser.add_argument("--test", dest="test", action="store_true", help="enable test mode")
    parser.add_argument("--seed", default=1, type=int, help="dynet random seed")
    options = parser.parse_args()


    #validate params
    if options.forward_dim is not None and options.backward_dim is not None:
        print("Asym parameters set")
        options.hidden_dim = (options.forward_dim, options.backward_dim)

    if not options.word_level_dim:
        if type(options.hidden_dim) == int:
            options.word_level_dim = options.hidden_dim
        else:
            #if we are using an asym bilstm
            options.word_level_dim = sum(options.hidden_dim)


    # ===-----------------------------------------------------------------------===
    # Set up logging
    # ===-----------------------------------------------------------------------===

    if not os.path.exists(options.log_dir):
        os.mkdir(options.log_dir)
    if options.log_to_stdout:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(filename=options.log_dir + "/log.txt", filemode="w", format="%(message)s", level=logging.INFO)
    train_dev_cost = utils.CSVLogger(options.log_dir + "/train_dev.log", ["Train.cost", "Dev.cost"])


    # ===-----------------------------------------------------------------------===
    # Log run parameters
    # ===-----------------------------------------------------------------------===
    logging.info(options)
    logging.info(
    """
    Dataset: {dataset}
    Num Epochs: {epochs}
    LSTM: {layers} layers, {hidden} hidden dim, {word} word level dim
    Training set size limit: {sent} sentences or {tokes} tokens
    Initial Learning Rate: {lr}
    Dropout: {dropout}

    """.format(
            dataset=options.dataset,
            epochs=options.num_epochs,
            layers=options.lstm_layers,
            hidden=options.hidden_dim,
            word=options.word_level_dim,
            sent=options.training_sentence_size,
            tokes=options.token_size,
            lr=options.learning_rate,
            dropout=options.dropout)
    )

    # after reading the seed from options, import dynet and other dynet dependent modules
    dynet_config.set(random_seed=options.seed)
    import dynet as dy
    from AsymBiLSTM import AsymBiRNNBuilder

    if options.debug:
        print("DEBUG MODE")

    # ===-----------------------------------------------------------------------===
    # Read in dataset
    # ===-----------------------------------------------------------------------===
    processed_dataset = ProcessedDataset(options.dataset,
                        options.training_sentence_size,
                        options.token_size)

    w2i, t2is, c2i, i2w, i2ts, i2c, \
    training_instances, training_vocab, \
    dev_instances, dev_vocab, test_instances, tag_set_sizes = processed_dataset.get_all_params()

    # ===-----------------------------------------------------------------------===
    # Build model and trainer
    # ===-----------------------------------------------------------------------===


    model = LSTMTagger(tagset_sizes=tag_set_sizes,
                       num_lstm_layers=options.lstm_layers,
                       hidden_dim=options.hidden_dim,
                       word_level_dim=options.word_level_dim,
                       charset_size=len(c2i),
                       char_embedding_dim=DEFAULT_CHAR_EMBEDDING_SIZE,
                       vocab_size=len(w2i),
                       word_embedding_dim=DEFAULT_WORD_EMBEDDING_SIZE)

    trainer = dy.MomentumSGDTrainer(model.model, options.learning_rate, 0.9)
    logging.info("Training Algorithm: {}".format(type(trainer)))

    logging.info("Number training instances: {}".format(len(training_instances)))
    logging.info("Number dev instances: {}".format(len(dev_instances)))

    best_dev_pos = 0.0
    old_best_name = None

    for epoch in range(options.num_epochs):
        bar = progressbar.ProgressBar()

        # set up epoch
        random.shuffle(training_instances)
        train_loss = 0.0

        if options.dropout > 0:
            model.set_dropout(options.dropout)

        # debug samples small set for faster full loop
        if options.debug:
            train_instances = training_instances[0:int(len(training_instances)/20)]
        else:
            train_instances = training_instances

        # main training loop
        for idx,instance in enumerate(bar(train_instances)):
            if len(instance.sentence) == 0: continue

            gold_tags = instance.tags
            for att in model.attributes:
                if att not in instance.tags:
                    # 'pad' entire sentence with none tags
                    gold_tags[att] = [t2is[att][NONE_TAG]] * len(instance.sentence)

            word_chars = get_word_chars(instance.sentence, i2w, c2i)

            # calculate all losses for sentence
            loss_exprs = model.loss(word_chars, gold_tags)
            loss_expr = dy.esum(list(loss_exprs.values()))
            loss = loss_expr.scalar_value()

            # bail if loss is NaN
            if np.isnan(loss):
                assert False, "NaN occurred"

            train_loss += (loss / len(instance.sentence))

            # backward pass and parameter update
            loss_expr.backward()
            trainer.update()

        # log epoch's train phase
        logging.info("\n")
        logging.info("Epoch {} complete".format(epoch + 1))
        # here used to be a learning rate update, no longer supported in dynet 2.0
        print(trainer.status())

        train_loss = train_loss / len(train_instances)

        # evaluate dev data
        model.disable_dropout()
        dev_loss = 0.0
        dev_correct = Counter()
        dev_total = Counter()
        dev_oov_total = Counter()
        bar = progressbar.ProgressBar()
        total_wrong = Counter()
        total_wrong_oov = Counter()
        f1_eval = Evaluator(m = 'att')
        if options.debug:
            d_instances = dev_instances[0:int(len(dev_instances)/10)]
        else:
            d_instances = dev_instances
        with open("{}/devout_epoch-{:02d}.txt".format(options.log_dir, epoch + 1), 'w') as dev_writer:
            for instance in bar(d_instances):
                if len(instance.sentence) == 0: continue
                gold_tags = instance.tags
                word_chars = get_word_chars(instance.sentence, i2w, c2i)
                for att in model.attributes:
                    if att not in instance.tags:
                        gold_tags[att] = [t2is[att][NONE_TAG]] * len(instance.sentence)
                losses = model.loss(word_chars, gold_tags)
                total_loss = sum([l.scalar_value() for l in list(losses.values())])
                out_tags_set, _ = model.tag_sentence(word_chars)

                gold_strings = utils.morphotag_strings(i2ts, gold_tags)
                obs_strings = utils.morphotag_strings(i2ts, out_tags_set)
                for g, o in zip(gold_strings, obs_strings):
                    f1_eval.add_instance(utils.split_tagstring(g, has_pos=True), utils.split_tagstring(o, has_pos=True))
                for att, tags in list(gold_tags.items()):
                    out_tags = out_tags_set[att]
                    correct_sent = True

                    oov_strings = []
                    for word, gold, out in zip(instance.sentence, tags, out_tags):
                        if gold == out:
                            dev_correct[att] += 1
                        else:
                            # Got the wrong tag
                            total_wrong[att] += 1
                            correct_sent = False
                            if i2w[word] not in training_vocab:
                                total_wrong_oov[att] += 1

                        if i2w[word] not in training_vocab:
                            dev_oov_total[att] += 1
                            oov_strings.append("OOV")
                        else:
                            oov_strings.append("")

                    dev_total[att] += len(tags)

                dev_loss += (total_loss / len(instance.sentence))

                dev_writer.write(("\n"
                                 + "\n".join(["\t".join(z) for z in zip([i2w[w] for w in instance.sentence],
                                                                             gold_strings, obs_strings, oov_strings)])
                                 + "\n"))


        dev_loss = dev_loss / len(d_instances)

        dev_pos_accuracy = (dev_correct[POS_KEY] / dev_total[POS_KEY])
        # log epoch results
        logging.info("POS Dev Accuracy: {}".format(dev_correct[POS_KEY] / dev_total[POS_KEY]))
        logging.info("POS % OOV accuracy: {}".format((dev_oov_total[POS_KEY] - total_wrong_oov[POS_KEY]) / dev_oov_total[POS_KEY]))
        if total_wrong[POS_KEY] > 0:
            logging.info("POS % Wrong that are OOV: {}".format(total_wrong_oov[POS_KEY] / total_wrong[POS_KEY]))
        for attr in t2is.keys():
            if attr != POS_KEY:
                logging.info("{} F1: {}".format(attr, f1_eval.mic_f1(att = attr)))
        logging.info("Total attribute F1s: {} micro, {} macro, POS included = {}".format(f1_eval.mic_f1(), f1_eval.mac_f1(), False))

        logging.info("Total dev tokens: {}, Total dev OOV: {}, % OOV: {}".format(dev_total[POS_KEY], dev_oov_total[POS_KEY], dev_oov_total[POS_KEY] / dev_total[POS_KEY]))

        logging.info("Train Loss: {}".format(train_loss))
        logging.info("Dev Loss: {}".format(dev_loss))
        train_dev_cost.add_column([train_loss, dev_loss])

        #after the first epoch, log out the normalized words
        if epoch == 0:
            logging.info("Writing {} normalized words".format(len(normalized_words.keys())))
            with open("{}/normalized_words.txt".format(options.log_dir), 'w') as f:
                for word, count in normalized_words.most_common():
                    f.write("{} {}\n".format(word, count))



        if epoch > 1 and epoch % 10 != 0: # leave outputs from epochs 1,10,20, etc.
            old_devout_file_name = "{}/devout_epoch-{:02d}.txt".format(options.log_dir, epoch)
            os.remove(old_devout_file_name)

        # write best model by dev pos accuracy in addition to periodic writeouts
        if dev_pos_accuracy > best_dev_pos:
            print("{:.4f} > {:.4f}, writing new best dev model".format(dev_pos_accuracy * 100, best_dev_pos * 100))
            best_dev_pos = dev_pos_accuracy
            #remove old best
            if old_best_name:
                os.remove(old_best_name)
                os.remove(old_best_name + "-atts")

            new_model_file_name = "{}/best_model_epoch-{:02d}-{:.4f}.bin".format(options.log_dir, epoch + 1, dev_pos_accuracy)
            model.save(new_model_file_name)
            old_best_name = new_model_file_name

        # serialize model
        if not options.no_model:
            new_model_file_name = "{}/model_epoch-{:02d}.bin".format(options.log_dir, epoch + 1)
            logging.info("Saving model to {}".format(new_model_file_name))
            model.save(new_model_file_name)
            if epoch > 1 and epoch % 10 != 0: # leave models from epochs 1,10,20, etc.
                logging.info("Removing files from previous epoch.")
                old_model_file_name = "{}/model_epoch-{:02d}.bin".format(options.log_dir, epoch)
                os.remove(old_model_file_name)
                os.remove(old_model_file_name + "-atts")

        # epoch loop ends

    # evaluate test data (once)
    logging.info("\n")
    logging.info("Number test instances: {}".format(len(test_instances)))
    model.disable_dropout()
    test_correct = Counter()
    test_total = Counter()
    test_oov_total = Counter()
    bar = progressbar.ProgressBar()
    total_wrong = Counter()
    total_wrong_oov = Counter()
    f1_eval = Evaluator(m = 'att')
    if options.debug:
        t_instances = test_instances[0:int(len(test_instances)/10)]
    else:
        t_instances = test_instances
    with open("{}/testout.txt".format(options.log_dir), 'w') as test_writer:
        for instance in bar(t_instances):
            if len(instance.sentence) == 0: continue
            gold_tags = instance.tags
            for att in model.attributes:
                if att not in instance.tags:
                    gold_tags[att] = [t2is[att][NONE_TAG]] * len(instance.sentence)
            word_chars = get_word_chars(instance.sentence, i2w, c2i)
            out_tags_set, _ = model.tag_sentence(word_chars)

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
            test_writer.write(("\n"
                             + "\n".join(["\t".join(z) for z in zip([i2w[w] for w in instance.sentence],
                                                                         gold_strings, obs_strings, oov_strings)])
                             + "\n"))


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

# imported as a module, not a script
# so initialize dynet without a seed
else:
    import dynet as dy
    from AsymBiLSTM import AsymBiRNNBuilder

