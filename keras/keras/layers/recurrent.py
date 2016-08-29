# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from .. import activations, initializations
from ..utils.theano_utils import shared_zeros, alloc_zeros_matrix
from ..layers.core import Layer
from .. import regularizers

from six.moves import range

class SimpleRNN(Layer):
    '''
        Fully connected RNN where output is to fed back to input.
        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, input_dim, output_dim,
        init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
        truncate_gradient=-1, return_sequences=False):
        super(SimpleRNN,self).__init__()
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.input = T.tensor3()

        self.W = self.init((self.input_dim, self.output_dim))
        self.U = self.init((self.output_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))
        self.params = [self.W, self.U, self.b]

        if weights is not None:
            self.set_weights(weights)

    def _step(self, x_t, h_tm1, u):
        '''
            Variable names follow the conventions from:
            http://deeplearning.net/software/theano/library/scan.html
        '''
        return self.activation(x_t + T.dot(h_tm1, u))

    def output(self, train):
        X = self.get_input(train) # shape: (nb_samples, time (padded with zeros at the end), input_dim)
        # new shape: (time, nb_samples, input_dim) -> because theano.scan iterates over main dimension
        X = X.dimshuffle((1,0,2))

        x = T.dot(X, self.W) + self.b
        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        outputs, updates = theano.scan(
            self._step, # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=x, # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=alloc_zeros_matrix(X.shape[1], self.output_dim),
            non_sequences=self.U, # static inputs to _step
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}
class SimpleBRNN_Entity_Barzilay(Layer):
    '''
        Fully connected Bi-directional RNN where:
            Output at time=t is fed back to input for time=t+1 in a forward pass
            Output at time=t is fed back to input for time=t-1 in a backward pass
    '''
    def __init__(self, input_dim, output_dim,
        init='uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
        truncate_gradient=-1,  return_sequences=False):
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.input = T.tensor3()
        self.W_o  =  self.init((self.input_dim, self.output_dim))
        self.W_if = self.init((self.input_dim, self.output_dim))    # Input -> Forward
        self.W_ib = self.init((self.input_dim, self.output_dim))    # Input -> Backward
        self.W_ff = self.init((self.output_dim, self.output_dim))   # Forward tm1 -> Forward t
        self.W_bb = self.init((self.output_dim, self.output_dim))   # Backward t -> Backward tm1
        self.b_if = shared_zeros((self.output_dim))
        self.b_ib = shared_zeros((self.output_dim))
        self.b_f = shared_zeros((self.output_dim))
        self.b_b = shared_zeros((self.output_dim))
        self.b_o =  shared_zeros((self.output_dim))
        self.params = [self.W_o,self.W_if,self.W_ib, self.W_ff, self.W_bb,self.b_if,self.b_ib, self.b_f, self.b_b, self.b_o]


        if weights is not None:
            self.set_weights(weights)

    def _step(self, x_t, h_tm1, u,b):
        return self.activation(x_t + T.dot(h_tm1, u)+b)

    def output(self, train):
        X = self.get_input(train) # shape: (nb_samples, time (padded with zeros at the end), input_dim)
        # new shape: (time, nb_samples, input_dim) -> because theano.scan iterates over main dimension
        X = X.dimshuffle((1, 0, 2))
        lenX=X.shape[0]
        Entity=X[lenX-2:].dimshuffle(1,0,2)

        X=X[:lenX-2]
        b_o=self.b_o
        b_on= T.repeat(T.repeat(b_o.reshape((1,self.output_dim)),X.shape[0],axis=0).reshape((1,X.shape[0],self.output_dim)),X.shape[1],axis=0)
        xf = self.activation(T.dot(X, self.W_if) + self.b_if)
        xb = self.activation(T.dot(X, self.W_ib) + self.b_ib)

        # Iterate forward over the first dimension of the x array (=time).
        outputs_f, updates_f = theano.scan(
            self._step,  # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=xf,  # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=alloc_zeros_matrix(X.shape[1], self.output_dim),
            non_sequences=[self.W_ff,self.b_f],  # static inputs to _step
            truncate_gradient=self.truncate_gradient
        )
        # Iterate backward over the first dimension of the x array (=time).
        outputs_b, updates_b = theano.scan(
            self._step,  # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=xb,  # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=alloc_zeros_matrix(X.shape[1], self.output_dim),
            non_sequences=[self.W_bb,self.b_b],  # static inputs to _step
            truncate_gradient=self.truncate_gradient,
            go_backwards=True  # Iterate backwards through time
        )
        #return outputs_f.dimshuffle((1, 0, 2))
        if self.return_sequences:
            return T.concatenate([T.add(T.tensordot(T.add(outputs_f.dimshuffle((1, 0, 2)), outputs_b[::-1].dimshuffle((1,0,2))),self.W_o,[[2],[0]]),b_on),Entity],axis=1)
        return T.concatenate((outputs_f[-1], outputs_b[0]))

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}

class BLSTM(Layer):
    def __init__(self, input_dim, output_dim,
        init='glorot_uniform', inner_init='orthogonal',
        activation='tanh', inner_activation='hard_sigmoid',
        weights=None, truncate_gradient=-1, return_sequences=False,
        is_entity=False, regularize=False):

        self.is_entity = is_entity
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_if = self.init((self.input_dim, self.output_dim))
        self.W_ib = self.init((self.input_dim, self.output_dim))
        self.U_if = self.inner_init((self.output_dim, self.output_dim))
        self.U_ib = self.inner_init((self.output_dim, self.output_dim))
        self.b_if = shared_zeros((self.output_dim))
        self.b_ib = shared_zeros((self.output_dim))

        self.W_ff = self.init((self.input_dim, self.output_dim))
        self.W_fb = self.init((self.input_dim, self.output_dim))
        self.U_ff = self.inner_init((self.output_dim, self.output_dim))
        self.U_fb = self.inner_init((self.output_dim, self.output_dim))
        self.b_ff = shared_zeros((self.output_dim))
        self.b_fb = shared_zeros((self.output_dim))

        self.W_cf = self.init((self.input_dim, self.output_dim))
        self.W_cb = self.init((self.input_dim, self.output_dim))
        self.U_cf = self.inner_init((self.output_dim, self.output_dim))
        self.U_cb = self.inner_init((self.output_dim, self.output_dim))
        self.b_cf = shared_zeros((self.output_dim))
        self.b_cb = shared_zeros((self.output_dim))

        self.W_of = self.init((self.input_dim, self.output_dim))
        self.W_ob = self.init((self.input_dim, self.output_dim))
        self.U_of = self.inner_init((self.output_dim, self.output_dim))
        self.U_ob = self.inner_init((self.output_dim, self.output_dim))
        self.b_of = shared_zeros((self.output_dim))
        self.b_ob = shared_zeros((self.output_dim))

        self.W_yf = self.init((self.output_dim, self.output_dim))
        self.W_yb = self.init((self.output_dim, self.output_dim))
        #self.W_y = self.init((self.output_dim, self.output_dim))
        self.b_y = shared_zeros((self.output_dim))

        self.params = [
            self.W_if, self.U_if, self.b_if,
            self.W_ib, self.U_ib, self.b_ib,

            self.W_cf, self.U_cf, self.b_cf,
            self.W_cb, self.U_cb, self.b_cb,

            self.W_ff, self.U_ff, self.b_ff,
            self.W_fb, self.U_fb, self.b_fb,

            self.W_of, self.U_of, self.b_of,
            self.W_ob, self.U_ob, self.b_ob,

            self.W_yf, self.W_yb, self.b_y
            #self.W_y, self.b_y
        ]
        if regularize:
            self.regularizers = []
            for i in self.params:
                self.regularizers.append(regularizers.my_l2)

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
        xi_t, xf_t, xo_t, xc_t,
        h_tm1, c_tm1,
        u_i, u_f, u_o, u_c):
        i_t = self.inner_activation(xi_t + T.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))


        if self.is_entity:
            Entity = X[-1:].dimshuffle(1,0,2)
            X = X[:-1]

        b_y = self.b_y
        b_yn = T.repeat(T.repeat(b_y.reshape((1,self.output_dim)),X.shape[0],axis=0).reshape((1,X.shape[0],self.output_dim)), X.shape[1], axis=0)

        xif = T.dot(X, self.W_if) + self.b_if
        xib = T.dot(X, self.W_ib) + self.b_ib

        xff = T.dot(X, self.W_ff) + self.b_ff
        xfb = T.dot(X, self.W_fb) + self.b_fb

        xcf = T.dot(X, self.W_cf) + self.b_cf
        xcb = T.dot(X, self.W_cb) + self.b_cb

        xof = T.dot(X, self.W_of) + self.b_of
        xob = T.dot(X, self.W_ob) + self.b_ob

        [outputs_f, memories_f], updates_f = theano.scan(
            self._step,
            sequences=[xif, xff, xof, xcf],
            outputs_info=[
                alloc_zeros_matrix(X.shape[1], self.output_dim),
                alloc_zeros_matrix(X.shape[1], self.output_dim)
            ],
            non_sequences=[self.U_if, self.U_ff, self.U_of, self.U_cf],
            truncate_gradient=self.truncate_gradient
        )
        [outputs_b, memories_b], updates_b = theano.scan(
            self._step,
            sequences=[xib, xfb, xob, xcb],
            outputs_info=[
                alloc_zeros_matrix(X.shape[1], self.output_dim),
                alloc_zeros_matrix(X.shape[1], self.output_dim)
            ],
            non_sequences=[self.U_ib, self.U_fb, self.U_ob, self.U_cb],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            y = T.add(T.add(
                    T.tensordot(outputs_f.dimshuffle((1,0,2)), self.W_yf, [[2],[0]]),
                    T.tensordot(outputs_b[::-1].dimshuffle((1,0,2)), self.W_yb, [[2],[0]])),
                b_yn)
            # y = T.add(T.tensordot(
            #     T.add(outputs_f.dimshuffle((1, 0, 2)),
            #           outputs_b[::-1].dimshuffle((1,0,2))),
            #     self.W_y,[[2],[0]]),b_yn)
            if self.is_entity:
                return T.concatenate([y, Entity], axis=1)
            else:
                return y
        return T.concatenate((outputs_f[-1], outputs_b[0]))

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}

class BRNN(Layer):
    '''
        Fully connected Bi-directional RNN where:
            Output at time=t is fed back to input for time=t+1 in a forward pass
            Output at time=t is fed back to input for time=t-1 in a backward pass
    '''
    def __init__(self, input_dim, output_dim,
        init='uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
        truncate_gradient=-1,  return_sequences=False, is_entity=False, regularize=False):
        #whyjay
        self.is_entity = is_entity

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.input = T.tensor3()
        self.W_o  =  self.init((self.input_dim, self.output_dim))
        self.W_if = self.init((self.input_dim, self.output_dim))    # Input -> Forward
        self.W_ib = self.init((self.input_dim, self.output_dim))    # Input -> Backward
        self.W_ff = self.init((self.output_dim, self.output_dim))   # Forward tm1 -> Forward t
        self.W_bb = self.init((self.output_dim, self.output_dim))   # Backward t -> Backward tm1
        self.b_if = shared_zeros((self.output_dim))
        self.b_ib = shared_zeros((self.output_dim))
        self.b_f = shared_zeros((self.output_dim))
        self.b_b = shared_zeros((self.output_dim))
        self.b_o =  shared_zeros((self.output_dim))
        self.params = [self.W_o,self.W_if,self.W_ib, self.W_ff, self.W_bb,self.b_if,self.b_ib, self.b_f, self.b_b, self.b_o]

        if regularize:
            self.regularizers = []
            for i in self.params:
                self.regularizers.append(regularizers.my_l2)

        if weights is not None:
            self.set_weights(weights)

    def _step(self, x_t, h_tm1, u,b):
        return self.activation(x_t + T.dot(h_tm1, u)+b)

    def output(self, train):
        X = self.get_input(train) # shape: (nb_samples, time (padded with zeros at the end), input_dim)
        # new shape: (time, nb_samples, input_dim) -> because theano.scan iterates over main dimension
        X = X.dimshuffle((1, 0, 2))

        if self.is_entity:
            lenX=X.shape[0]
            Entity=X[lenX-1:].dimshuffle(1,0,2)
            X=X[:lenX-1]

        xf = self.activation(T.dot(X, self.W_if) + self.b_if)
        xb = self.activation(T.dot(X, self.W_ib) + self.b_ib)
        b_o=self.b_o
        b_on= T.repeat(T.repeat(b_o.reshape((1,self.output_dim)),X.shape[0],axis=0).reshape((1,X.shape[0],self.output_dim)),X.shape[1],axis=0)

        # Iterate forward over the first dimension of the x array (=time).
        outputs_f, updates_f = theano.scan(
            self._step,  # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=xf,  # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=alloc_zeros_matrix(X.shape[1], self.output_dim),
            non_sequences=[self.W_ff,self.b_f],  # static inputs to _step
            truncate_gradient=self.truncate_gradient
        )
        # Iterate backward over the first dimension of the x array (=time).
        outputs_b, updates_b = theano.scan(
            self._step,  # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=xb,  # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=alloc_zeros_matrix(X.shape[1], self.output_dim),
            non_sequences=[self.W_bb,self.b_b],  # static inputs to _step
            truncate_gradient=self.truncate_gradient,
            go_backwards=True  # Iterate backwards through time
        )
        #return outputs_f.dimshuffle((1, 0, 2))
        if self.return_sequences:
            if self.is_entity:
                return T.concatenate([T.add(T.tensordot(T.add(outputs_f.dimshuffle((1, 0, 2)), outputs_b[::-1].dimshuffle((1,0,2))),self.W_o,[[2],[0]]),b_on),Entity],axis=1)
            else:
                return T.add(T.tensordot(T.add(outputs_f.dimshuffle((1, 0, 2)), outputs_b[::-1].dimshuffle((1,0,2))),self.W_o,[[2],[0]]),b_on)

        return T.concatenate((outputs_f[-1], outputs_b[0]))

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}

class SimpleDeepRNN(Layer):
    '''
        Fully connected RNN where the output of multiple timesteps
        (up to "depth" steps in the past) is fed back to the input:

        output = activation( W.x_t + b + inner_activation(U_1.h_tm1) + inner_activation(U_2.h_tm2) + ... )

        This demonstrates how to build RNNs with arbitrary lookback.
        Also (probably) not a super useful model.
    '''
    def __init__(self, input_dim, output_dim, depth=3,
        init='glorot_uniform', inner_init='orthogonal',
        activation='sigmoid', inner_activation='hard_sigmoid',
        weights=None, truncate_gradient=-1, return_sequences=False):
        super(SimpleDeepRNN,self).__init__()
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.depth = depth
        self.return_sequences = return_sequences
        self.input = T.tensor3()

        self.W = self.init((self.input_dim, self.output_dim))
        self.Us = [self.init((self.output_dim, self.output_dim)) for _ in range(self.depth)]
        self.b = shared_zeros((self.output_dim))
        self.params = [self.W] + self.Us + [self.b]

        if weights is not None:
            self.set_weights(weights)

    def _step(self, *args):
        o = args[0]
        for i in range(1, self.depth+1):
            o += self.inner_activation(T.dot(args[i], args[i+self.depth]))
        return self.activation(o)

    def output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))

        x = T.dot(X, self.W) + self.b

        outputs, updates = theano.scan(
            self._step,
            sequences=x,
            outputs_info=[dict(
                initial=T.alloc(np.cast[theano.config.floatX](0.), self.depth, X.shape[1], self.output_dim),
                taps = [(-i-1) for i in range(self.depth)]
            )],
            non_sequences=self.Us,
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "depth":self.depth,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}



class GRU(Layer):
    '''
        Gated Recurrent Unit - Cho et al. 2014

        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            On the Properties of Neural Machine Translation: Encoder–Decoder Approaches
                http://www.aclweb.org/anthology/W14-4012
            Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
                http://arxiv.org/pdf/1412.3555v1.pdf
    '''
    def __init__(self, input_dim, output_dim=128,
        init='glorot_uniform', inner_init='orthogonal',
        activation='sigmoid', inner_activation='hard_sigmoid',
        weights=None, truncate_gradient=-1, return_sequences=False):

        super(GRU,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_z = self.init((self.input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((self.input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((self.input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
        xz_t, xr_t, xh_t,
        h_tm1,
        u_z, u_r, u_h):
        z = self.inner_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = z * h_tm1 + (1 - z) * hh_t
        return h_t

    def output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h],
            outputs_info=alloc_zeros_matrix(X.shape[1], self.output_dim),
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}



class LSTM(Layer):
    '''
        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html

        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    '''
    def __init__(self, input_dim, output_dim=128,
        init='glorot_uniform', inner_init='orthogonal',
        activation='tanh', inner_activation='hard_sigmoid',
        weights=None, truncate_gradient=-1, return_sequences=False):

        super(LSTM,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_i = self.init((self.input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim))

        self.W_f = self.init((self.input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = shared_zeros((self.output_dim))

        self.W_c = self.init((self.input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((self.input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
        xi_t, xf_t, xo_t, xc_t,
        h_tm1, c_tm1,
        u_i, u_f, u_o, u_c):
        i_t = self.inner_activation(xi_t + T.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))

        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o

        [outputs, memories], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc],
            outputs_info=[
                alloc_zeros_matrix(X.shape[1], self.output_dim),
                alloc_zeros_matrix(X.shape[1], self.output_dim)
            ],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}


