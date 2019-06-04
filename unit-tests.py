import unittest
import dynet as dy
import string

from utils import Instance, get_word_chars
from model import LSTMTagger

def numpy_values(data):
    results = []
    if type(data[0]) == dy.Expression:
        for d in data:
            results.append(d.npvalue())
    else:
        for sub in data:
            results.append(numpy_values(sub))
    return results


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
        self.assertEqual(len(char_embeddings), 4)
        for word in char_embeddings:
            for char in word:
                char = char.npvalue()
                self.assertTrue(len(char) == 128)

    def test_dropout(self):
        model = LSTMTagger(tagset_sizes={'POS' : 10, 'ABC' : 20},
                           num_lstm_layers=2,
                           hidden_dim=128,
                           word_level_dim=128,
                           charset_size=len(self.c2i),
                           char_embedding_dim=128,
                           vocab_size=1000,
                           word_embedding_dim=128)

        model.disable_dropout()
        word_chars = get_word_chars([0, 1, 2, 3], self.i2w, self.c2i)
        a, char_embeddings_no_drop = model.tag_sentence(word_chars)
        char_embeddings_no_drop = numpy_values(char_embeddings_no_drop)

        dy.renew_cg()
        model.set_dropout(0.50)
        word_chars = get_word_chars([0, 1, 2, 3], self.i2w, self.c2i)
        b, char_embeddings_with_drop = model.tag_sentence(word_chars)
        char_embeddings_with_drop = numpy_values(char_embeddings_with_drop)

        print(char_embeddings_no_drop[0][0])
        print(char_embeddings_with_drop[0][0])
        print(a, b)

if __name__ == '__main__':
    unittest.main()
