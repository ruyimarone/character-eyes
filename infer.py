import model
from utils import PADDING_CHAR, get_word_chars

class WrappedTagger:
    """WrappedTagger
       Wraps a model and dataset to make post training analysis easier
    """
    def __init__(self, dataset_path, num_lstm_layers, hidden_dim, char_embedding_dim, \
            word_level_dim):
        self.dataset = model.ProcessedDataset(dataset_path)


        self.model = model.LSTMTagger(tagset_sizes = self.dataset.tag_set_sizes,
                                num_lstm_layers = num_lstm_layers,
                                hidden_dim = hidden_dim,
                                word_level_dim = word_level_dim,
                                charset_size = len(self.dataset.c2i),
                                char_embedding_dim = char_embedding_dim,
                                vocab_size = None,
                                word_embedding_dim=None)

        self.desc = "WrappedTagger<{}> char:{} word:{}".format(dataset_path, hidden_dim, word_level_dim)
        self.hidden_dim = hidden_dim
        self.word_level_dim = word_level_dim
        self.dataset_path = dataset_path

    def forward_text(self, raw_text):
        tokens = raw_text.split(' ')
        pad_char = self.dataset.c2i[PADDING_CHAR]
        chars = [[pad_char] + [self.dataset.c2i[c] for c in word] + [pad_char] for word in tokens]
        out_tags_set, char_embeddings = self.model.tag_sentence(chars)
        return out_tags_set, char_embeddings

    def get_pos(self, out_tags):
        return (' '.join([self.dataset.i2ts['POS'][i] for i in out_tags['POS']]))

    def load_weights(self, weights_path, verbose=True):
        """Loads weights, accounting for differing model versions.
           Will load weights from a normal dynet lstm into an asym version.

        :param weights_path: path to the weights file
        :param verbose: print messages as special cases arise
        """

        legacy_to_new = {
         '/birnn/vanilla-lstm-builder/_0': '/asymbirnn/vanilla-lstm-builder/_0',
         '/birnn/vanilla-lstm-builder/_1': '/asymbirnn/vanilla-lstm-builder/_1',
         '/birnn/vanilla-lstm-builder/_2': '/asymbirnn/vanilla-lstm-builder/_2',
         '/birnn/vanilla-lstm-builder_1/_0': '/asymbirnn/vanilla-lstm-builder_1/_0',
         '/birnn/vanilla-lstm-builder_1/_1': '/asymbirnn/vanilla-lstm-builder_1/_1',
         '/birnn/vanilla-lstm-builder_1/_2': '/asymbirnn/vanilla-lstm-builder_1/_2',
         '/birnn_1/vanilla-lstm-builder/_0': '/birnn/vanilla-lstm-builder/_0',
         '/birnn_1/vanilla-lstm-builder/_1': '/birnn/vanilla-lstm-builder/_1',
         '/birnn_1/vanilla-lstm-builder/_2': '/birnn/vanilla-lstm-builder/_2',
         '/birnn_1/vanilla-lstm-builder_1/_0': '/birnn/vanilla-lstm-builder_1/_0',
         '/birnn_1/vanilla-lstm-builder_1/_1': '/birnn/vanilla-lstm-builder_1/_1',
         '/birnn_1/vanilla-lstm-builder_1/_2': '/birnn/vanilla-lstm-builder_1/_2',
         '/birnn_1/vanilla-lstm-builder_2/_0': '/birnn/vanilla-lstm-builder_2/_0',
         '/birnn_1/vanilla-lstm-builder_2/_1': '/birnn/vanilla-lstm-builder_2/_1',
         '/birnn_1/vanilla-lstm-builder_2/_2': '/birnn/vanilla-lstm-builder_2/_2',
         '/birnn_1/vanilla-lstm-builder_3/_0': '/birnn/vanilla-lstm-builder_3/_0',
         '/birnn_1/vanilla-lstm-builder_3/_1': '/birnn/vanilla-lstm-builder_3/_1',
         '/birnn_1/vanilla-lstm-builder_3/_2': '/birnn/vanilla-lstm-builder_3/_2'}
        new_to_legacy = {n : l for l, n in legacy_to_new.items()}

        source_param_names = []
        with open(weights_path, 'r') as f:
            for line in f:
                if line.startswith("#Parameter#"):
                    source_param_names.append(line.split()[1])

        legacy = all(name in source_param_names for name in legacy_to_new)
        if verbose:
            if legacy:
                print("Using legacy loading")
            else:
                print("Using standard loading")

        dy_model = self.model.model # yikes
        for param in (dy_model.parameters_list() + dy_model.lookup_parameters_list()):
            source_name = param.name()
            if legacy and source_name in new_to_legacy:
                source_name = new_to_legacy[source_name]

            #populate target param with source param weights
            param.populate(weights_path, source_name)

        self.model.disable_dropout()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.desc

