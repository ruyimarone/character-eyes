import dynet as dy

class AsymBiRNNBuilder(object):
    """

    (MODIFIED/STOLEN from the dynet source (python/_dynet.pyx))

    Modified to wire differently dimensioned (asymmetric) RNNs
    Can also handle degenerate asymmetric RNNs, where either the forward or backward dimension is 0.
    Provides a convenience method to get the final hidden states of sub rnn.
    """
    def __init__(self, num_layers, input_dim, hidden_dim, model, rnn_builder_factory, builder_layers=None, logger = None):
        """Args:
            num_layers: depth of the BiRNN
            input_dim: size of the inputs
            hidden_dim: size of the outputs (and intermediate layer representations.) This can be a tuple, specifiying different sizes for the fwd and bwd RNNs.
            model
            rnn_builder_factory: RNNBuilder subclass, e.g. LSTMBuilder
            builder_layers: list of (forward, backward) pairs of RNNBuilder instances to directly initialize layers
        """
        self.spec = num_layers, input_dim, hidden_dim, rnn_builder_factory, builder_layers
        model = self.model = model.add_subcollection("asymbirnn")

        self.logger = logger

        #
        if type(hidden_dim) == int:
            assert hidden_dim % 2 == 0, "AsymBiRNN hidden dimension must be even when given as int: %d" % hidden_dim
            forward_dim = hidden_dim / 2
            backward_dim = hidden_dim / 2
        else:
            print("using asym rnn, f = {}, b = {}".format(hidden_dim[0], hidden_dim[1]))
            forward_dim, backward_dim = hidden_dim

        self.forward_dim = forward_dim
        self.backward_dim = backward_dim

        if logger:
            logger.info("ASYM BiRNN input {}".format(input_dim))
            logger.info("ASYM BiRNN forward {}".format(self.forward_dim))
            logger.info("ASYM BiRNN backward {}".format(self.backward_dim))

        if builder_layers is None:
            assert num_layers > 0, "BiRNN number of layers must be positive: %d" % num_layers
            self.builder_layers = []

            f = rnn_builder_factory(1, input_dim, forward_dim, model) if forward_dim != 0 else None
            b = rnn_builder_factory(1, input_dim, backward_dim, model) if backward_dim != 0 else None

            self.builder_layers.append((f,b))
            for _ in range(num_layers-1):
                f = rnn_builder_factory(1, forward_dim + backward_dim, forward_dim, model) if forward_dim != 0 else None
                b = rnn_builder_factory(1, forward_dim + backward_dim, backward_dim, model) if backward_dim != 0 else None
                self.builder_layers.append((f,b))
        else:
            self.builder_layers = builder_layers

    @classmethod
    def from_spec(cls, spec, model):
        num_layers, input_dim, hidden_dim, rnn_builder_factory, builder_layers = spec
        return cls(num_layers, input_dim, hidden_dim, model, rnn_builder_factory, builder_layers)

    def param_collection(self): return self.model

    def whoami(self): return "AsymBiRNNBuilder"

    def set_dropout(self, p):
      for (fb,bb) in self.builder_layers:
        fb.set_dropout(p)
        bb.set_dropout(p)

    def disable_dropout(self):
      for (fb,bb) in self.builder_layers:
        fb.disable_dropout()
        bb.disable_dropout()

    def add_inputs(self, es):
        """
        Not implemented for asymmetric RNNs
        """

        raise NotImplementedError("Not implemented for asymmetric RNNs")

    def transduce(self, es):
        """
        returns the list of output Expressions obtained by adding the given inputs
        to the current state, one by one, to both the forward and backward RNNs,
        and concatenating.

        @param es: a list of Expression
        see also add_inputs(xs)
        .transduce(xs) is different from .add_inputs(xs) in the following way:
            .add_inputs(xs) returns a list of RNNState pairs. RNNState objects can be
             queried in various ways. In particular, they allow access to the previous
             state, as well as to the state-vectors (h() and s() )
            .transduce(xs) returns a list of Expression. These are just the output
             expressions. For many cases, this suffices.
             transduce is much more memory efficient than add_inputs.
        """
        # for e in es:
            # ensure_freshness(e)
        for (fb,bb) in self.builder_layers:
            fs = fb.initial_state().transduce(es) if self.forward_dim != 0 else None
            bs = bb.initial_state().transduce(reversed(es)) if self.backward_dim != 0 else None
            if fs is None or bs is None:
                es = fs if fs is not None else bs
            else:
                es = [dy.concatenate([f,b]) for f,b in zip(fs, reversed(bs))]
        return es

    def final_hiddens(self, es):
        """
        Like transduce, but it returns the final hidden state of each sub RNN.
        This is NOT the hidden state of the last character.
        """
        for (fb,bb) in self.builder_layers:
            fs = fb.initial_state().transduce(es) if self.forward_dim != 0 else None
            bs = bb.initial_state().transduce(reversed(es)) if self.backward_dim != 0 else None
            if fs is None or bs is None:
                es = fs if fs is not None else list(reversed(bs))
            else:
                es = [dy.concatenate([f,b]) for f,b in zip(fs, reversed(bs))]

        if self.forward_dim == 0:
            #return the last backward state
            return es[0], es
        elif self.backward_dim == 0:
            #return the last forward state
            return es[-1], es

        last_forward = es[-1][:self.forward_dim]
        last_backward = es[0][self.forward_dim:]
        final_hidden = dy.concatenate([last_forward, last_backward])
        assert (final_hidden.dim()[0][0] == self.backward_dim + self.forward_dim)
        return final_hidden, es

