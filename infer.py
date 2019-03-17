from model import LSTMTagger, ProcessedDataset, PADDING_CHAR, get_word_chars


# processed_dataset = ProcessedDataset(
class WrappedTagger:
    def __init__(self, dataset_path, num_lstm_layers, hidden_dim, char_embedding_dim, word_level_dim = None, use_elman=False):

        self.dataset = ProcessedDataset(dataset_path)

        #TODO fixme
        if word_level_dim == None:
            word_level_dim = 128

        self.model = LSTMTagger(tagset_sizes = self.dataset.tag_set_sizes,
                                num_lstm_layers = num_lstm_layers,
                                hidden_dim = hidden_dim,
                                word_level_dim = word_level_dim,
                                no_we_update = True, #doesnt matter?
                                use_char_rnn = True,
                                charset_size = len(self.dataset.c2i),
                                char_embedding_dim = char_embedding_dim,
                                use_elman_rnn = use_elman,
                                att_props = None,
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
        self.model.model.populate(weights_path)
        if not verbose:
            print("loaded model")
        #make sure dropuut is disabled at test time
        self.model.disable_dropout()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.desc

