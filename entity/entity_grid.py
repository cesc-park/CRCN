#https://github.com/kedz/discourse
import numpy as np
from collections import defaultdict
from os import listdir
from os.path import join, isfile, splitext
import math
import pandas as pd
import itertools
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def generate_transitions(role_set, path_size):
    """Create an ordered list of transitions, from a set role labels.
        The number of roles is exponential in the path size."""

    # Each loop iteration i adds all transitions
    # of size (i+1) to the list trans.
    trans = ['']

    for p in range(path_size):
        new_trans = []
        for r in role_set:
            for t in trans:
                new_trans.append(t+r)
        trans = new_trans

    # trans is now a sorted list of all possible role transitions.
    return sorted(trans)


def new_entity_grid(grid_dataframe_or_file_handle,
                    syntax=False, max_salience=2,
                    history=2, sentences=None):
        """Construct a new EntityGrid from a pandas DataFrame of
            entity role transitions or an entity grid file."""

        # grid_dataframe_or_file_handle is a matrix representing
        # an entity's role in a sentence, or a file handle to a
        # text file containg one.
        # Columns correspond to entities and rows correspond
        # to a document's sentences.
        # e.g. an entity grid with 3 entities and 2 sentences:
        #     [['X','-'],
        #      ['O','S'],
        #      ['S','O']]
        #
        # When syntax is true, use the role set (S,O,X,-), else use (X,-).
        #
        # When max_salience is > 1, split entities into categories
        # by frequency up to the threshold set by max_salience.
        # Transition set then becomes the cartesian product
        # of {salient,nonsalient} x {s,o,x,-}^length.
        #
        # History is the transition length.
        # When history is 3 and syntax=False,
        # the transitions are {---,--X,-X-,X--,-XX,XX-,X-X,XXX}.

        # If we are constructing from a file, read in and parse, otherwise
        # just get the dataframe.
        entities=[]
        if isinstance(grid_dataframe_or_file_handle, file):
            grid_dataframe = parse_grid_file(grid_dataframe_or_file_handle)
        elif type(grid_dataframe_or_file_handle) is str:
            grid_dataframe, entities =parse_grid_string(grid_dataframe_or_file_handle)
        else:
            grid_dataframe = grid_dataframe_or_file_handle
        #print_full(grid_dataframe)
        # Initialize the transition set
        if syntax is True:
            trans = generate_transitions(('-', 'X', 'S', 'O'), history)
        else:
            trans = generate_transitions(('-', 'X'), history)

        # Split entity role grid into multiple salience levels
        grids = _split_grid_by_salience(grid_dataframe, max_salience)

        # If no document sentences are supplied, generate fake sentences:
        # 'Sentence 1', 'Sentence 2', ... 'Sentence n'.
        if sentences is None:
            sentences = ["Sentence {}".format(s) for s in grids[0].columns]
        model=EntityGrid(grids, trans, sentences, history)
        model.entities=entities
        return model


def _split_grid_by_salience(grid_dataframe, max_salience):
    """Split a dataframe represnting the whole document into
        several dataframes for each salience level."""

    # List of dataframes by salience.
    data_frames_by_salience = []

    # smap maps salience lvl to a list of rows.
    smap = defaultdict(list)
    for s in range(1, max_salience+1):
        smap[s] = []

    # Make sure this is a DataFrame -- previous versions used char matrices.
    if isinstance(grid_dataframe, pd.DataFrame):
        # Add each row to the appropriate salience list.
        for ent, roles in grid_dataframe.iterrows():
            salience = len(filter(lambda r: r != '-', roles))
            # Salience levels above max salience are thresholded.
            if salience > max_salience:
                salience = max_salience
            smap[salience].append((ent, roles))
    else:
        raise TypeError('Argument grid_dataframe must be of type DataFrame.')

    # For each salience level in smap, make a new DataFrame
    # and add to data_frames_by_salience.
    for s in smap:
        if len(smap[s]) > 0:
            entities = [ent for ent, _ in smap[s]]
            roles = [r for _, r in smap[s]]
            grid = pd.DataFrame(roles, index=entities,
                                columns=grid_dataframe.columns)
            data_frames_by_salience.append(grid)
        # If no entities in this salience level return an empty dataframe.
        else:
            grid = pd.DataFrame([])
            data_frames_by_salience.append(grid)

    return data_frames_by_salience


def parse_grid_file(f):
    """Read from an open file handle and parse an entity grid."""

    lines = ''
    for line in f:
        lines += line
    return parse_grid_string(lines)


def parse_grid_string(grid_str):
    """Parses output of TestGrid from the Brown Coherence Toolkit or
        other text encoded Entity Grid into a pandas DataFrame."""

    lines = grid_str.strip().split("\n")

    # A list of lists holding entity transitions.
    # This will become the data in a pandas
    # DataFrame representing the entity grid.
    grid = []

    # A list of entities in this document.
    # This will become the index of a pandas
    # DataFrame representing the entity grid.
    entities = []

    for line in lines:

        if (line.strip() is not ""):

            # Each entry is separated by a space.
            roles = line.strip().split(" ")

            # The first entry is the entity name. Add this to the index.
            entities.append(roles[0].decode("ascii", "ignore"))

            # Add the transitions (these start at index 1)
            # for this entity to egrid.
            roles = [roles[t].upper() for t in range(1, len(roles))]
            grid.append(roles)

    egrid_df = pd.DataFrame(grid, index=entities)

    return egrid_df, entities


class EntityGrid:

    def __init__(self, grids, trans, sentences,
                 history=2):
        """An EntityGrid computes the transition count
            vector representation of a document."""

        if isinstance(grids, list) or isinstance(grids, tuple):
            self.grids = grids
        elif isinstance(grids, pd.DataFrame):
            self.grids = (grids)
        else:
            raise TypeError('Argument \'grids\' must be a \
                            list/tuple of pandas DataFrames or \
                            a single DataFrame')

        # Store ordered pretty sentences for pretty printing.
        self.sentences = sentences

        # Transition counts for each salience level.
        self._tcounts = [self._count_transitions(grid, trans, history)
                         for grid in self.grids]
        self.entities=[]
        # Store set of possible transitions.
        self.trans = trans

        # Length of history for this model.
        self.history = history

        # Retrieve the number of salience levels for this model.
        self.salience_levels = len(self.grids)

        # Total number of transitions by salience level.
        self._tot_trans = [sum(tcounts.values()) for tcounts in self._tcounts]

        # Transition Probability Vector is None until getter is called.
        self._tpv = None

        # Transition Probability Dataframe is None until getter is called.
        self._tpv_df = None

        # Transition Count Vector is None until getter is called.
        self._tcv = None

        # Transition Count Dataframe is None until getter is called.
        self._tcv_df = None

        # Generate feature labels (the cartesian product
        # of salience and transition sets.
        sal_lvls = ["S{}:".format(s) for s in range(1, self.salience_levels+1)]
        labels = ["{}{}".format(sal, t)
                  for sal, t in itertools.product(sal_lvls, trans)]
        self.labels = labels

    def _count_transitions(self, grid, trans, history):
        """Counts the transitions that occur in this grid."""

        tcounts = defaultdict(int)

        # For each column and for each row, count the entity transitions
        for ent, roles in grid.iterrows():

            for r, value in enumerate(roles.values):
                t = value
                i = r+1
                while len(t) < history and i < len(roles.values):
                    t += roles.values[i]

                    if t in trans:
                        tcounts[t] += 1
                    i += 1
        return tcounts

    def _build_vector_rep(self, trans, normalized):
        """Construct the transition count vector representation.
            If normalized, these can be interpreted as a generative model
            for entity transitions."""

        # For each feature for each salience level, add to feature vector v.
        v = []
        for salience, counts in enumerate(self._tcounts):
            for t in trans:
                if normalized:
                    if self._tot_trans[salience] > 0:
                        v.append(counts[t]/float(self._tot_trans[salience]))
                    else:
                        v.append(0)
                else:
                    v.append(counts[t])

        # Keep DataFrame for pretty printing.
        v_df = pd.DataFrame(np.asmatrix(v), index=['t'], columns=self.labels)
        v = np.asarray(v)

        # If normalized,
        # cache this vector as a tpv (transition probability vector).
        if normalized:
            self._tpv_df = v_df
            self._tpv = v

        # Otherwise cache this vector as a tcv (transition count vector).
        else:
            self._tcv_df = v_df
            self._tcv = v

    def get_partial_grid(self, indices):
        """Return another entity grid model with all parameters
            the same but constructing a new entity grid from
            an ordered list of indices. N.B.: Indices start at 1!"""

        # Get new entity grids - preserving salience levels.
        new_grids = [grid[indices] for grid in self.grids]
        # Retrieve sentence permutation
        new_sentences = []
        for i in indices:
            new_sentences.append(self.sentences[i-1])

        return EntityGrid(new_grids, self.trans, new_sentences, history=self.history)


    def pretty_string_from_hypergraph_path(self, hypergraph, path):
        perm = [int(hypergraph.label(p).sents[0].split('_')[1])-1
                for p in path
                if hypergraph.label(p).sents[0] not in ['START', 'END']]
        sents = []
        for p in reversed(perm):
            sents.append(self.sentences[p])

        return "\n".join(sents)

    def pretty_string(self):
        return "\n".join(self.sentences)

    def get_trans_cnt_vctr(self):
        """Get a vector of entity transition counts."""

        if self._tcv is None:
            self._build_vector_rep(self.trans, False)
        return self._tcv

    def get_trans_cnt_dataframe(self):
        """Get a dataframe of entity transition counts."""

        if self._tcv_df is None:
            self._build_vector_rep(self.trans, False)
        return self._tcv_df

    def get_trans_prob_vctr(self):
        """Get a vector of entity transition probabilities."""

        if self._tpv is None:
            self._build_vector_rep(self.trans, True)
        return self._tpv

    def get_trans_prob_dataframe(self):
        """Get a dataframe of entity transition probabilities."""

        if self._tpv_df is None:
            self._build_vector_rep(self.trans, True)
        return self._tpv_df

    def print_grids(self):
        for grid in self.grids:
            print grid
