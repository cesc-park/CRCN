# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from .. import activations, initializations
from ..utils.theano_utils import shared_zeros, alloc_zeros_matrix
from ..layers.core import Layer

from six.moves import range


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


