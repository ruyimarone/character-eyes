import unittest
import dynet as dy
import string

from model import Instance
class ModelRegression(unittest.TestCase):
    def setUp(self):
        dy.reset_random_seed(1)
        self.i2w = {0 : 'this',
                    1 : 'is',
                    2 : 'a',
                    3 : 'test',
                    4 : 'thisisanoov',
                }
        self.w2i = {v : k for k, v in self.i2w.items()}
        self.c2i = {c : i for i, c in enumerate(list(string.printable) + ["<*>"])}
        self.i2c = {v : k for k, v in self.c2i.items()}

        self.test_sentance = Instance([0, 1, 2, 3], {})

    def test_seed_ok(self):
        self.assertAlmostEqual(dy.random_normal(1).npvalue()[0], 0.3063996, places=4)

    def test_base_model(self):
        from model import LSTMTagger, get_word_chars
        model = LSTMTagger(tagset_sizes={'POS' : 10},
                           num_lstm_layers=2,
                           hidden_dim=128,
                           word_level_dim=128,
                           charset_size=len(self.c2i),
                           char_embedding_dim=128,
                           vocab_size=1000,
                           word_embedding_dim=128)
        # model.
        word_chars = get_word_chars([0, 1, 2, 3], self.i2w, self.c2i)
        tags, char_embeddings = model.tag_sentence(word_chars)

    def test_dropout(self):
        from model import LSTMTagger, get_word_chars
        model = LSTMTagger(tagset_sizes={'POS' : 10},
                           num_lstm_layers=2,
                           hidden_dim=128,
                           word_level_dim=128,
                           charset_size=len(self.c2i),
                           char_embedding_dim=128,
                           vocab_size=1000,
                           word_embedding_dim=128)
        # model.
        word_chars = get_word_chars([0, 1, 2, 3], self.i2w, self.c2i)
        tags, char_embeddings = model.tag_sentence(word_chars)
        print(char_embeddings)

if __name__ == '__main__':
    unittest.main()
