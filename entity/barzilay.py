#https://github.com/kedz/discourse
import xmltodict
import re
from collections import OrderedDict, defaultdict
from nltk import Tree
import pandas as pd
from discourse.models import entity_grid as eg
from StringIO import StringIO


# Penn TreeBank Noun tags
_noun_tags = set(['NN', 'NNP', 'NNS', 'NNPS'])

_pronoun_tags = set(['PRP', 'PRP$', 'PDT', 'DT'])

# When the same entity occurs more than once in a sentence,
# take the role with the highest precedence.
_role_precedence = {'S': 4, 'O': 3, 'X': 2, '-': 1}

# Maps a grammatical relation to the governor/dependent position and
# it's role in the sentence.
_relation_map = defaultdict(lambda: ('dependent', '-'))
_relation_map['cop'] = ('governor', 'O')
_relation_map['nsubj'] = ('dependent', 'S')
_relation_map['csubj'] = ('dependent', 'S')
_relation_map['dobj'] = ('dependent', 'O')
_relation_map['iobj'] = ('dependent', 'O')
_relation_map['pobj'] = ('dependent', 'X')
_relation_map['prep_'] = ('dependent', 'X')
_relation_map['parat'] = ('dependent', 'X')
_relation_map['nn'] = ('dependent', '-')
_relation_map['dep'] = ('dependent', 'X')


def new_barzilay_model(f, max_salience=2, history=2):
    """Parses file handle to the xml output of the Stanford CoreNLP
        library and creates an EntityGrid in the style of Barzilay
        and Lapata, 2005."""

    # XML map of CoreNLP output (tagging, parses, dependencies, coref).
    xml = xmltodict.parse(f)['root']['document']

    # Map of sentence,word indices to grammatical role.
    sent_word2role = _get_role_map(xml)

    # Map of sentence,word indices to entity.
    # '?' indicates no entity.
    sent_word2ent = _get_entity_map(xml)

    # Get document parses.
    parses = [Tree(sentence['parse'])
              for sentence in xml['sentences']['sentence']]

    # Create dataframe from entity/role maps and parses.
    grid_df = _build_grid(parses, sent_word2ent, sent_word2role)

    # Grab the pretty sentences for pretty printing.
    sentences = [reduce(lambda x, y: x + " " + y, tree.leaves())
                 for tree in parses]

    return eg.new_entity_grid(grid_df, syntax=True,
                              max_salience=max_salience, history=history,
                              sentences=sentences)


def _get_indexed_role(dep):
    """Return the function (governor/dependent) and role
        based on the dependency relation type."""

    gram_relation = dep['@type'][0:5]
    if gram_relation in ["conj", "conj_"]:
        return (-1, 'conj')
    (function, role) = _relation_map[gram_relation]
    return (_iminus_one(dep[function]['@idx']), role)


def _get_role_map(xml):
    """Produces a map of sentence, word indices
        to the word's grammatical role."""

    # Maps sentence, word indices --> grammatical role.
    sent_word2role = defaultdict(lambda: defaultdict(lambda: '-'))

    sentences = xml['sentences']['sentence']

    # Iterate through each sentence, word
    # and retrieve the word's grammatical role.
    for s, sent in enumerate(sentences):
        # Ensure dependencies are not empty.
        if 'dep' in sent['dependencies'][2]:
            dependencies = sent['dependencies'][2]['dep']

            # Ignore edge cases where there is only one dependency.
            # E.g. the sentence has only one token and
            # there is no information here.
            if '@type' not in dependencies:
                # Enter sentence, word assignments into the map
                for d in dependencies:
                    (w, role) = _get_indexed_role(d)
                    sent_word2role[s][w] = role

    return sent_word2role


def _get_entity_map(xml):
    """Map sentence, word indices to entity if
        the word is a phrase head of an entity mention."""

    coref = xml['coreference']['coreference']

    sent_word2ent = defaultdict(lambda: defaultdict(lambda: '?'))

    # Iterate through corenlp xml coref items and map their head indicies to
    for entnum, entity in enumerate(coref, start=1):
        entlabel = "ent{}".format(entnum)
        for mention_dict in entity['mention']:
            snum = _iminus_one(mention_dict['sentence'])
            hnum = _iminus_one(mention_dict['head'])
            sent_word2ent[snum][hnum] = entlabel

    return sent_word2ent


def _build_grid(parses, sent_word2ent, sent_word2role):
    """Create a dataframe from parses, entity, and role maps."""

    # Maps entity to sentence2role map.
    entities = defaultdict(dict)

    # Stores an entities mention form.
    ent_rep = {}

    # For each sentence, and each word, get the entity and role information.
    # If a word meets requirements, add it to the entities map.
    for p, parse in enumerate(parses):
        for w, (word, pos) in enumerate(parse.pos()):
            role = sent_word2role[p][w]
            entity = sent_word2ent[p][w]

            # Label is either the word's entity mention form, or if
            # word is not an entity simply the word itself in caps.
            # Entities encountered for the first time have the current
            # word set as it's mention form.
            label = ''
            if entity == '?':
                label = word.upper()
            elif entity not in ent_rep:
                ent_rep[entity] = 'E:'+word.upper()
                label = ent_rep[entity]
            else:
                label = ent_rep[entity]

            # If a word is an entity and a noun/pronoun add it to the grid.
            # If a word is not an entity (or part of its noun phrase) and
            # it is a noun, add it as well.
            add_word = False
            if entity != '?' and (pos in _noun_tags or pos in _pronoun_tags):
                add_word = True
            elif pos in _noun_tags:
                add_word = True

            # Don't add empties now.
            if role == '-':
                add_word = False

            # Add word, but don't overwrite the same entity in the current
            # sentence if it occurred previously with a more important role.
            # 'S' > 'O' > 'X' > '-'.
            if add_word:
                if p in entities[label]:
                    prev_role = entities[label][p]
                    if _role_precedence[role] > _role_precedence[prev_role]:
                        entities[label][p] = role
                else:
                    entities[label][p] = role

    # Construct an entity by sentence matrix of roles.
    ent_role_table = []

    for ent in entities:
        roles = []
        for i in range(len(parses)):
            if i in entities[ent]:
                roles.append(entities[ent][i])
            else:
                roles.append('-')
        ent_role_table.append(roles)

    # Index each entity by its first occuring head word.
    index = [ent if ent not in ent_rep else ent_rep[ent]
             for ent in entities]
    # Start column labels to 1, 2, ... n.
    columns = [i for i in range(1, len(parses)+1)]

    # Create a dataframe of this entity grid.
    egrid_df = pd.DataFrame(ent_role_table,
                            index=index,
                            columns=columns)

    return egrid_df


def test_barzilay_model():

    f = StringIO(xml)
    model = new_barzilay_model(f)
    for grid in model.grids:
        print grid

xml = """<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet href="CoreNLP-to-HTML.xsl" type="text/xsl"?>
<root>
  <document>
    <sentences>
      <sentence id="1">
        <tokens>
          <token id="1">
            <word>During</word>
            <lemma>during</lemma>
            <CharacterOffsetBegin>0</CharacterOffsetBegin>
            <CharacterOffsetEnd>6</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="2">
            <word>IFR</word>
            <lemma>IFR</lemma>
            <CharacterOffsetBegin>7</CharacterOffsetBegin>
            <CharacterOffsetEnd>10</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>ORGANIZATION</NER>
          </token>
          <token id="3">
            <word>arrival</word>
            <lemma>arrival</lemma>
            <CharacterOffsetBegin>11</CharacterOffsetBegin>
            <CharacterOffsetEnd>18</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="4">
            <word>,</word>
            <lemma>,</lemma>
            <CharacterOffsetBegin>18</CharacterOffsetBegin>
            <CharacterOffsetEnd>19</CharacterOffsetEnd>
            <POS>,</POS>
            <NER>O</NER>
          </token>
          <token id="5">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>20</CharacterOffsetBegin>
            <CharacterOffsetEnd>23</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="6">
            <word>flight</word>
            <lemma>flight</lemma>
            <CharacterOffsetBegin>24</CharacterOffsetBegin>
            <CharacterOffsetEnd>30</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="7">
            <word>was</word>
            <lemma>be</lemma>
            <CharacterOffsetBegin>31</CharacterOffsetBegin>
            <CharacterOffsetEnd>34</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="8">
            <word>cleared</word>
            <lemma>clear</lemma>
            <CharacterOffsetBegin>35</CharacterOffsetBegin>
            <CharacterOffsetEnd>42</CharacterOffsetEnd>
            <POS>VBN</POS>
            <NER>O</NER>
          </token>
          <token id="9">
            <word>for</word>
            <lemma>for</lemma>
            <CharacterOffsetBegin>43</CharacterOffsetBegin>
            <CharacterOffsetEnd>46</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="10">
            <word>an</word>
            <lemma>a</lemma>
            <CharacterOffsetBegin>47</CharacterOffsetBegin>
            <CharacterOffsetEnd>49</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="11">
            <word>ILS</word>
            <lemma>ILS</lemma>
            <CharacterOffsetBegin>50</CharacterOffsetBegin>
            <CharacterOffsetEnd>53</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>O</NER>
          </token>
          <token id="12">
            <word>Runway</word>
            <lemma>runway</lemma>
            <CharacterOffsetBegin>54</CharacterOffsetBegin>
            <CharacterOffsetEnd>60</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="13">
            <word>32</word>
            <lemma>32</lemma>
            <CharacterOffsetBegin>61</CharacterOffsetBegin>
            <CharacterOffsetEnd>63</CharacterOffsetEnd>
            <POS>CD</POS>
            <NER>NUMBER</NER>
            <NormalizedNER>32.0</NormalizedNER>
          </token>
          <token id="14">
            <word>Approach</word>
            <lemma>approach</lemma>
            <CharacterOffsetBegin>64</CharacterOffsetBegin>
            <CharacterOffsetEnd>72</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="15">
            <word>.</word>
            <lemma>.</lemma>
            <CharacterOffsetBegin>72</CharacterOffsetBegin>
            <CharacterOffsetEnd>73</CharacterOffsetEnd>
            <POS>.</POS>
            <NER>O</NER>
          </token>
        </tokens>
        <parse>(ROOT (S (PP (IN During) (NP (NNP IFR) (NN arrival))) (, ,) (NP (DT the) (NN flight)) (VP (VBD was) (VP (VBN cleared) (PP (IN for) (NP (DT an) (NNP ILS) (NN Runway) (CD 32) (NN Approach))))) (. .))) </parse>
        <dependencies type="basic-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="8">cleared</dependent>
          </dep>
          <dep type="prep">
            <governor idx="8">cleared</governor>
            <dependent idx="1">During</dependent>
          </dep>
          <dep type="nn">
            <governor idx="3">arrival</governor>
            <dependent idx="2">IFR</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="1">During</governor>
            <dependent idx="3">arrival</dependent>
          </dep>
          <dep type="det">
            <governor idx="6">flight</governor>
            <dependent idx="5">the</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="8">cleared</governor>
            <dependent idx="6">flight</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="8">cleared</governor>
            <dependent idx="7">was</dependent>
          </dep>
          <dep type="prep">
            <governor idx="8">cleared</governor>
            <dependent idx="9">for</dependent>
          </dep>
          <dep type="det">
            <governor idx="14">Approach</governor>
            <dependent idx="10">an</dependent>
          </dep>
          <dep type="nn">
            <governor idx="14">Approach</governor>
            <dependent idx="11">ILS</dependent>
          </dep>
          <dep type="nn">
            <governor idx="14">Approach</governor>
            <dependent idx="12">Runway</dependent>
          </dep>
          <dep type="num">
            <governor idx="14">Approach</governor>
            <dependent idx="13">32</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="9">for</governor>
            <dependent idx="14">Approach</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="8">cleared</dependent>
          </dep>
          <dep type="nn">
            <governor idx="3">arrival</governor>
            <dependent idx="2">IFR</dependent>
          </dep>
          <dep type="prep_during">
            <governor idx="8">cleared</governor>
            <dependent idx="3">arrival</dependent>
          </dep>
          <dep type="det">
            <governor idx="6">flight</governor>
            <dependent idx="5">the</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="8">cleared</governor>
            <dependent idx="6">flight</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="8">cleared</governor>
            <dependent idx="7">was</dependent>
          </dep>
          <dep type="det">
            <governor idx="14">Approach</governor>
            <dependent idx="10">an</dependent>
          </dep>
          <dep type="nn">
            <governor idx="14">Approach</governor>
            <dependent idx="11">ILS</dependent>
          </dep>
          <dep type="nn">
            <governor idx="14">Approach</governor>
            <dependent idx="12">Runway</dependent>
          </dep>
          <dep type="num">
            <governor idx="14">Approach</governor>
            <dependent idx="13">32</dependent>
          </dep>
          <dep type="prep_for">
            <governor idx="8">cleared</governor>
            <dependent idx="14">Approach</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-ccprocessed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="8">cleared</dependent>
          </dep>
          <dep type="nn">
            <governor idx="3">arrival</governor>
            <dependent idx="2">IFR</dependent>
          </dep>
          <dep type="prep_during">
            <governor idx="8">cleared</governor>
            <dependent idx="3">arrival</dependent>
          </dep>
          <dep type="det">
            <governor idx="6">flight</governor>
            <dependent idx="5">the</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="8">cleared</governor>
            <dependent idx="6">flight</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="8">cleared</governor>
            <dependent idx="7">was</dependent>
          </dep>
          <dep type="det">
            <governor idx="14">Approach</governor>
            <dependent idx="10">an</dependent>
          </dep>
          <dep type="nn">
            <governor idx="14">Approach</governor>
            <dependent idx="11">ILS</dependent>
          </dep>
          <dep type="nn">
            <governor idx="14">Approach</governor>
            <dependent idx="12">Runway</dependent>
          </dep>
          <dep type="num">
            <governor idx="14">Approach</governor>
            <dependent idx="13">32</dependent>
          </dep>
          <dep type="prep_for">
            <governor idx="8">cleared</governor>
            <dependent idx="14">Approach</dependent>
          </dep>
        </dependencies>
      </sentence>
      <sentence id="2">
        <tokens>
          <token id="1">
            <word>The</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>75</CharacterOffsetBegin>
            <CharacterOffsetEnd>78</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="2">
            <word>pilot</word>
            <lemma>pilot</lemma>
            <CharacterOffsetBegin>79</CharacterOffsetBegin>
            <CharacterOffsetEnd>84</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="3">
            <word>-LRB-</word>
            <lemma>-lrb-</lemma>
            <CharacterOffsetBegin>84</CharacterOffsetBegin>
            <CharacterOffsetEnd>85</CharacterOffsetEnd>
            <POS>-LRB-</POS>
            <NER>O</NER>
          </token>
          <token id="4">
            <word>s</word>
            <lemma>s</lemma>
            <CharacterOffsetBegin>85</CharacterOffsetBegin>
            <CharacterOffsetEnd>86</CharacterOffsetEnd>
            <POS>NNS</POS>
            <NER>O</NER>
          </token>
          <token id="5">
            <word>-RRB-</word>
            <lemma>-rrb-</lemma>
            <CharacterOffsetBegin>86</CharacterOffsetBegin>
            <CharacterOffsetEnd>87</CharacterOffsetEnd>
            <POS>-RRB-</POS>
            <NER>O</NER>
          </token>
          <token id="6">
            <word>acknowledged</word>
            <lemma>acknowledge</lemma>
            <CharacterOffsetBegin>88</CharacterOffsetBegin>
            <CharacterOffsetEnd>100</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="7">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>101</CharacterOffsetBegin>
            <CharacterOffsetEnd>104</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="8">
            <word>clearance</word>
            <lemma>clearance</lemma>
            <CharacterOffsetBegin>105</CharacterOffsetBegin>
            <CharacterOffsetEnd>114</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="9">
            <word>&amp;</word>
            <lemma>&amp;</lemma>
            <CharacterOffsetBegin>115</CharacterOffsetBegin>
            <CharacterOffsetEnd>116</CharacterOffsetEnd>
            <POS>CC</POS>
            <NER>O</NER>
          </token>
          <token id="10">
            <word>initiated</word>
            <lemma>initiate</lemma>
            <CharacterOffsetBegin>117</CharacterOffsetBegin>
            <CharacterOffsetEnd>126</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="11">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>127</CharacterOffsetBegin>
            <CharacterOffsetEnd>130</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="12">
            <word>approach</word>
            <lemma>approach</lemma>
            <CharacterOffsetBegin>131</CharacterOffsetBegin>
            <CharacterOffsetEnd>139</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="13">
            <word>.</word>
            <lemma>.</lemma>
            <CharacterOffsetBegin>139</CharacterOffsetBegin>
            <CharacterOffsetEnd>140</CharacterOffsetEnd>
            <POS>.</POS>
            <NER>O</NER>
          </token>
        </tokens>
        <parse>(ROOT (S (NP (NP (DT The) (NN pilot)) (PRN (-LRB- -LRB-) (NNS s) (-RRB- -RRB-))) (VP (VBD acknowledged) (NP (NP (DT the) (NN clearance)) (CC &amp;) (S (VP (VBD initiated) (NP (DT the) (NN approach)))))) (. .))) </parse>
        <dependencies type="basic-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="6">acknowledged</dependent>
          </dep>
          <dep type="det">
            <governor idx="2">pilot</governor>
            <dependent idx="1">The</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="6">acknowledged</governor>
            <dependent idx="2">pilot</dependent>
          </dep>
          <dep type="dep">
            <governor idx="2">pilot</governor>
            <dependent idx="4">s</dependent>
          </dep>
          <dep type="det">
            <governor idx="8">clearance</governor>
            <dependent idx="7">the</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="6">acknowledged</governor>
            <dependent idx="8">clearance</dependent>
          </dep>
          <dep type="cc">
            <governor idx="8">clearance</governor>
            <dependent idx="9">&amp;</dependent>
          </dep>
          <dep type="conj">
            <governor idx="8">clearance</governor>
            <dependent idx="10">initiated</dependent>
          </dep>
          <dep type="det">
            <governor idx="12">approach</governor>
            <dependent idx="11">the</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="10">initiated</governor>
            <dependent idx="12">approach</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="6">acknowledged</dependent>
          </dep>
          <dep type="det">
            <governor idx="2">pilot</governor>
            <dependent idx="1">The</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="6">acknowledged</governor>
            <dependent idx="2">pilot</dependent>
          </dep>
          <dep type="dep">
            <governor idx="2">pilot</governor>
            <dependent idx="4">s</dependent>
          </dep>
          <dep type="det">
            <governor idx="8">clearance</governor>
            <dependent idx="7">the</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="6">acknowledged</governor>
            <dependent idx="8">clearance</dependent>
          </dep>
          <dep type="conj_and">
            <governor idx="8">clearance</governor>
            <dependent idx="10">initiated</dependent>
          </dep>
          <dep type="det">
            <governor idx="12">approach</governor>
            <dependent idx="11">the</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="10">initiated</governor>
            <dependent idx="12">approach</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-ccprocessed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="6">acknowledged</dependent>
          </dep>
          <dep type="det">
            <governor idx="2">pilot</governor>
            <dependent idx="1">The</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="6">acknowledged</governor>
            <dependent idx="2">pilot</dependent>
          </dep>
          <dep type="dep">
            <governor idx="2">pilot</governor>
            <dependent idx="4">s</dependent>
          </dep>
          <dep type="det">
            <governor idx="8">clearance</governor>
            <dependent idx="7">the</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="6">acknowledged</governor>
            <dependent idx="8">clearance</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="6">acknowledged</governor>
            <dependent idx="10">initiated</dependent>
          </dep>
          <dep type="conj_and">
            <governor idx="8">clearance</governor>
            <dependent idx="10">initiated</dependent>
          </dep>
          <dep type="det">
            <governor idx="12">approach</governor>
            <dependent idx="11">the</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="10">initiated</governor>
            <dependent idx="12">approach</dependent>
          </dep>
        </dependencies>
      </sentence>
      <sentence id="3">
        <tokens>
          <token id="1">
            <word>Radio</word>
            <lemma>Radio</lemma>
            <CharacterOffsetBegin>142</CharacterOffsetBegin>
            <CharacterOffsetEnd>147</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>O</NER>
          </token>
          <token id="2">
            <word>transmissions</word>
            <lemma>transmission</lemma>
            <CharacterOffsetBegin>148</CharacterOffsetBegin>
            <CharacterOffsetEnd>161</CharacterOffsetEnd>
            <POS>NNS</POS>
            <NER>O</NER>
          </token>
          <token id="3">
            <word>from</word>
            <lemma>from</lemma>
            <CharacterOffsetBegin>162</CharacterOffsetBegin>
            <CharacterOffsetEnd>166</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="4">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>167</CharacterOffsetBegin>
            <CharacterOffsetEnd>170</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="5">
            <word>airplane</word>
            <lemma>airplane</lemma>
            <CharacterOffsetBegin>171</CharacterOffsetBegin>
            <CharacterOffsetEnd>179</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="6">
            <word>became</word>
            <lemma>become</lemma>
            <CharacterOffsetBegin>180</CharacterOffsetBegin>
            <CharacterOffsetEnd>186</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="7">
            <word>unreadable</word>
            <lemma>unreadable</lemma>
            <CharacterOffsetBegin>187</CharacterOffsetBegin>
            <CharacterOffsetEnd>197</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="8">
            <word>as</word>
            <lemma>as</lemma>
            <CharacterOffsetBegin>198</CharacterOffsetBegin>
            <CharacterOffsetEnd>200</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="9">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>201</CharacterOffsetBegin>
            <CharacterOffsetEnd>204</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="10">
            <word>approach</word>
            <lemma>approach</lemma>
            <CharacterOffsetBegin>205</CharacterOffsetBegin>
            <CharacterOffsetEnd>213</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="11">
            <word>was</word>
            <lemma>be</lemma>
            <CharacterOffsetBegin>214</CharacterOffsetBegin>
            <CharacterOffsetEnd>217</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="12">
            <word>continued</word>
            <lemma>continue</lemma>
            <CharacterOffsetBegin>218</CharacterOffsetBegin>
            <CharacterOffsetEnd>227</CharacterOffsetEnd>
            <POS>VBN</POS>
            <NER>O</NER>
          </token>
          <token id="13">
            <word>.</word>
            <lemma>.</lemma>
            <CharacterOffsetBegin>227</CharacterOffsetBegin>
            <CharacterOffsetEnd>228</CharacterOffsetEnd>
            <POS>.</POS>
            <NER>O</NER>
          </token>
        </tokens>
        <parse>(ROOT (S (NP (NP (NNP Radio) (NNS transmissions)) (PP (IN from) (NP (DT the) (NN airplane)))) (VP (VBD became) (ADJP (JJ unreadable)) (SBAR (IN as) (S (NP (DT the) (NN approach)) (VP (VBD was) (VP (VBN continued)))))) (. .))) </parse>
        <dependencies type="basic-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="7">unreadable</dependent>
          </dep>
          <dep type="nn">
            <governor idx="2">transmissions</governor>
            <dependent idx="1">Radio</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="7">unreadable</governor>
            <dependent idx="2">transmissions</dependent>
          </dep>
          <dep type="prep">
            <governor idx="2">transmissions</governor>
            <dependent idx="3">from</dependent>
          </dep>
          <dep type="det">
            <governor idx="5">airplane</governor>
            <dependent idx="4">the</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="3">from</governor>
            <dependent idx="5">airplane</dependent>
          </dep>
          <dep type="cop">
            <governor idx="7">unreadable</governor>
            <dependent idx="6">became</dependent>
          </dep>
          <dep type="mark">
            <governor idx="12">continued</governor>
            <dependent idx="8">as</dependent>
          </dep>
          <dep type="det">
            <governor idx="10">approach</governor>
            <dependent idx="9">the</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="12">continued</governor>
            <dependent idx="10">approach</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="12">continued</governor>
            <dependent idx="11">was</dependent>
          </dep>
          <dep type="advcl">
            <governor idx="7">unreadable</governor>
            <dependent idx="12">continued</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="7">unreadable</dependent>
          </dep>
          <dep type="nn">
            <governor idx="2">transmissions</governor>
            <dependent idx="1">Radio</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="7">unreadable</governor>
            <dependent idx="2">transmissions</dependent>
          </dep>
          <dep type="det">
            <governor idx="5">airplane</governor>
            <dependent idx="4">the</dependent>
          </dep>
          <dep type="prep_from">
            <governor idx="2">transmissions</governor>
            <dependent idx="5">airplane</dependent>
          </dep>
          <dep type="cop">
            <governor idx="7">unreadable</governor>
            <dependent idx="6">became</dependent>
          </dep>
          <dep type="mark">
            <governor idx="12">continued</governor>
            <dependent idx="8">as</dependent>
          </dep>
          <dep type="det">
            <governor idx="10">approach</governor>
            <dependent idx="9">the</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="12">continued</governor>
            <dependent idx="10">approach</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="12">continued</governor>
            <dependent idx="11">was</dependent>
          </dep>
          <dep type="advcl">
            <governor idx="7">unreadable</governor>
            <dependent idx="12">continued</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-ccprocessed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="7">unreadable</dependent>
          </dep>
          <dep type="nn">
            <governor idx="2">transmissions</governor>
            <dependent idx="1">Radio</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="7">unreadable</governor>
            <dependent idx="2">transmissions</dependent>
          </dep>
          <dep type="det">
            <governor idx="5">airplane</governor>
            <dependent idx="4">the</dependent>
          </dep>
          <dep type="prep_from">
            <governor idx="2">transmissions</governor>
            <dependent idx="5">airplane</dependent>
          </dep>
          <dep type="cop">
            <governor idx="7">unreadable</governor>
            <dependent idx="6">became</dependent>
          </dep>
          <dep type="mark">
            <governor idx="12">continued</governor>
            <dependent idx="8">as</dependent>
          </dep>
          <dep type="det">
            <governor idx="10">approach</governor>
            <dependent idx="9">the</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="12">continued</governor>
            <dependent idx="10">approach</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="12">continued</governor>
            <dependent idx="11">was</dependent>
          </dep>
          <dep type="advcl">
            <governor idx="7">unreadable</governor>
            <dependent idx="12">continued</dependent>
          </dep>
        </dependencies>
      </sentence>
      <sentence id="4">
        <tokens>
          <token id="1">
            <word>Radar</word>
            <lemma>radar</lemma>
            <CharacterOffsetBegin>230</CharacterOffsetBegin>
            <CharacterOffsetEnd>235</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="2">
            <word>data</word>
            <lemma>datum</lemma>
            <CharacterOffsetBegin>236</CharacterOffsetBegin>
            <CharacterOffsetEnd>240</CharacterOffsetEnd>
            <POS>NNS</POS>
            <NER>O</NER>
          </token>
          <token id="3">
            <word>showed</word>
            <lemma>show</lemma>
            <CharacterOffsetBegin>241</CharacterOffsetBegin>
            <CharacterOffsetEnd>247</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="4">
            <word>that</word>
            <lemma>that</lemma>
            <CharacterOffsetBegin>248</CharacterOffsetBegin>
            <CharacterOffsetEnd>252</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="5">
            <word>on</word>
            <lemma>on</lemma>
            <CharacterOffsetBegin>253</CharacterOffsetBegin>
            <CharacterOffsetEnd>255</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="6">
            <word>final</word>
            <lemma>final</lemma>
            <CharacterOffsetBegin>256</CharacterOffsetBegin>
            <CharacterOffsetEnd>261</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="7">
            <word>approach</word>
            <lemma>approach</lemma>
            <CharacterOffsetBegin>262</CharacterOffsetBegin>
            <CharacterOffsetEnd>270</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="8">
            <word>,</word>
            <lemma>,</lemma>
            <CharacterOffsetBegin>270</CharacterOffsetBegin>
            <CharacterOffsetEnd>271</CharacterOffsetEnd>
            <POS>,</POS>
            <NER>O</NER>
          </token>
          <token id="9">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>272</CharacterOffsetBegin>
            <CharacterOffsetEnd>275</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="10">
            <word>airplane</word>
            <lemma>airplane</lemma>
            <CharacterOffsetBegin>276</CharacterOffsetBegin>
            <CharacterOffsetEnd>284</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="11">
            <word>deviated</word>
            <lemma>deviate</lemma>
            <CharacterOffsetBegin>285</CharacterOffsetBegin>
            <CharacterOffsetEnd>293</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="12">
            <word>laterally</word>
            <lemma>laterally</lemma>
            <CharacterOffsetBegin>294</CharacterOffsetBegin>
            <CharacterOffsetEnd>303</CharacterOffsetEnd>
            <POS>RB</POS>
            <NER>O</NER>
          </token>
          <token id="13">
            <word>-LRB-</word>
            <lemma>-lrb-</lemma>
            <CharacterOffsetBegin>304</CharacterOffsetBegin>
            <CharacterOffsetEnd>305</CharacterOffsetEnd>
            <POS>-LRB-</POS>
            <NER>O</NER>
          </token>
          <token id="14">
            <word>S-turned</word>
            <lemma>s-turned</lemma>
            <CharacterOffsetBegin>305</CharacterOffsetBegin>
            <CharacterOffsetEnd>313</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="15">
            <word>-RRB-</word>
            <lemma>-rrb-</lemma>
            <CharacterOffsetBegin>313</CharacterOffsetBegin>
            <CharacterOffsetEnd>314</CharacterOffsetEnd>
            <POS>-RRB-</POS>
            <NER>O</NER>
          </token>
          <token id="16">
            <word>through</word>
            <lemma>through</lemma>
            <CharacterOffsetBegin>315</CharacterOffsetBegin>
            <CharacterOffsetEnd>322</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="17">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>323</CharacterOffsetBegin>
            <CharacterOffsetEnd>326</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="18">
            <word>final</word>
            <lemma>final</lemma>
            <CharacterOffsetBegin>327</CharacterOffsetBegin>
            <CharacterOffsetEnd>332</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="19">
            <word>approach</word>
            <lemma>approach</lemma>
            <CharacterOffsetBegin>333</CharacterOffsetBegin>
            <CharacterOffsetEnd>341</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="20">
            <word>course</word>
            <lemma>course</lemma>
            <CharacterOffsetBegin>342</CharacterOffsetBegin>
            <CharacterOffsetEnd>348</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="21">
            <word>-LRB-</word>
            <lemma>-lrb-</lemma>
            <CharacterOffsetBegin>349</CharacterOffsetBegin>
            <CharacterOffsetEnd>350</CharacterOffsetEnd>
            <POS>-LRB-</POS>
            <NER>O</NER>
          </token>
          <token id="22">
            <word>as</word>
            <lemma>as</lemma>
            <CharacterOffsetBegin>350</CharacterOffsetBegin>
            <CharacterOffsetEnd>352</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="23">
            <word>if</word>
            <lemma>if</lemma>
            <CharacterOffsetBegin>353</CharacterOffsetBegin>
            <CharacterOffsetEnd>355</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="24">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>356</CharacterOffsetBegin>
            <CharacterOffsetEnd>359</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="25">
            <word>pilot</word>
            <lemma>pilot</lemma>
            <CharacterOffsetBegin>360</CharacterOffsetBegin>
            <CharacterOffsetEnd>365</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="26">
            <word>was</word>
            <lemma>be</lemma>
            <CharacterOffsetBegin>366</CharacterOffsetBegin>
            <CharacterOffsetEnd>369</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="27">
            <word>overcorrecting</word>
            <lemma>overcorrect</lemma>
            <CharacterOffsetBegin>370</CharacterOffsetBegin>
            <CharacterOffsetEnd>384</CharacterOffsetEnd>
            <POS>VBG</POS>
            <NER>O</NER>
          </token>
          <token id="28">
            <word>-RRB-</word>
            <lemma>-rrb-</lemma>
            <CharacterOffsetBegin>384</CharacterOffsetBegin>
            <CharacterOffsetEnd>385</CharacterOffsetEnd>
            <POS>-RRB-</POS>
            <NER>O</NER>
          </token>
          <token id="29">
            <word>.</word>
            <lemma>.</lemma>
            <CharacterOffsetBegin>385</CharacterOffsetBegin>
            <CharacterOffsetEnd>386</CharacterOffsetEnd>
            <POS>.</POS>
            <NER>O</NER>
          </token>
        </tokens>
        <parse>(ROOT (S (NP (NN Radar) (NNS data)) (VP (VBD showed) (SBAR (IN that) (S (PP (IN on) (NP (JJ final) (NN approach))) (, ,) (NP (DT the) (NN airplane)) (VP (VBD deviated) (ADVP (ADVP (RB laterally)) (PRN (-LRB- -LRB-) (ADJP (JJ S-turned)) (-RRB- -RRB-))) (PP (IN through) (NP (NP (DT the) (JJ final) (NN approach) (NN course)) (PRN (-LRB- -LRB-) (PP (IN as) (SBAR (IN if) (S (NP (DT the) (NN pilot)) (VP (VBD was) (VP (VBG overcorrecting)))))) (-RRB- -RRB-)))))))) (. .))) </parse>
        <dependencies type="basic-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="3">showed</dependent>
          </dep>
          <dep type="nn">
            <governor idx="2">data</governor>
            <dependent idx="1">Radar</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="3">showed</governor>
            <dependent idx="2">data</dependent>
          </dep>
          <dep type="mark">
            <governor idx="11">deviated</governor>
            <dependent idx="4">that</dependent>
          </dep>
          <dep type="prep">
            <governor idx="11">deviated</governor>
            <dependent idx="5">on</dependent>
          </dep>
          <dep type="amod">
            <governor idx="7">approach</governor>
            <dependent idx="6">final</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="5">on</governor>
            <dependent idx="7">approach</dependent>
          </dep>
          <dep type="det">
            <governor idx="10">airplane</governor>
            <dependent idx="9">the</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="11">deviated</governor>
            <dependent idx="10">airplane</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="3">showed</governor>
            <dependent idx="11">deviated</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="11">deviated</governor>
            <dependent idx="12">laterally</dependent>
          </dep>
          <dep type="dep">
            <governor idx="12">laterally</governor>
            <dependent idx="14">S-turned</dependent>
          </dep>
          <dep type="prep">
            <governor idx="11">deviated</governor>
            <dependent idx="16">through</dependent>
          </dep>
          <dep type="det">
            <governor idx="20">course</governor>
            <dependent idx="17">the</dependent>
          </dep>
          <dep type="amod">
            <governor idx="20">course</governor>
            <dependent idx="18">final</dependent>
          </dep>
          <dep type="nn">
            <governor idx="20">course</governor>
            <dependent idx="19">approach</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="16">through</governor>
            <dependent idx="20">course</dependent>
          </dep>
          <dep type="prep">
            <governor idx="20">course</governor>
            <dependent idx="22">as</dependent>
          </dep>
          <dep type="mark">
            <governor idx="27">overcorrecting</governor>
            <dependent idx="23">if</dependent>
          </dep>
          <dep type="det">
            <governor idx="25">pilot</governor>
            <dependent idx="24">the</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="27">overcorrecting</governor>
            <dependent idx="25">pilot</dependent>
          </dep>
          <dep type="aux">
            <governor idx="27">overcorrecting</governor>
            <dependent idx="26">was</dependent>
          </dep>
          <dep type="pcomp">
            <governor idx="22">as</governor>
            <dependent idx="27">overcorrecting</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="3">showed</dependent>
          </dep>
          <dep type="nn">
            <governor idx="2">data</governor>
            <dependent idx="1">Radar</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="3">showed</governor>
            <dependent idx="2">data</dependent>
          </dep>
          <dep type="mark">
            <governor idx="11">deviated</governor>
            <dependent idx="4">that</dependent>
          </dep>
          <dep type="amod">
            <governor idx="7">approach</governor>
            <dependent idx="6">final</dependent>
          </dep>
          <dep type="prep_on">
            <governor idx="11">deviated</governor>
            <dependent idx="7">approach</dependent>
          </dep>
          <dep type="det">
            <governor idx="10">airplane</governor>
            <dependent idx="9">the</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="11">deviated</governor>
            <dependent idx="10">airplane</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="3">showed</governor>
            <dependent idx="11">deviated</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="11">deviated</governor>
            <dependent idx="12">laterally</dependent>
          </dep>
          <dep type="dep">
            <governor idx="12">laterally</governor>
            <dependent idx="14">S-turned</dependent>
          </dep>
          <dep type="det">
            <governor idx="20">course</governor>
            <dependent idx="17">the</dependent>
          </dep>
          <dep type="amod">
            <governor idx="20">course</governor>
            <dependent idx="18">final</dependent>
          </dep>
          <dep type="nn">
            <governor idx="20">course</governor>
            <dependent idx="19">approach</dependent>
          </dep>
          <dep type="prep_through">
            <governor idx="11">deviated</governor>
            <dependent idx="20">course</dependent>
          </dep>
          <dep type="mark">
            <governor idx="27">overcorrecting</governor>
            <dependent idx="23">if</dependent>
          </dep>
          <dep type="det">
            <governor idx="25">pilot</governor>
            <dependent idx="24">the</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="27">overcorrecting</governor>
            <dependent idx="25">pilot</dependent>
          </dep>
          <dep type="aux">
            <governor idx="27">overcorrecting</governor>
            <dependent idx="26">was</dependent>
          </dep>
          <dep type="prepc_as">
            <governor idx="20">course</governor>
            <dependent idx="27">overcorrecting</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-ccprocessed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="3">showed</dependent>
          </dep>
          <dep type="nn">
            <governor idx="2">data</governor>
            <dependent idx="1">Radar</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="3">showed</governor>
            <dependent idx="2">data</dependent>
          </dep>
          <dep type="mark">
            <governor idx="11">deviated</governor>
            <dependent idx="4">that</dependent>
          </dep>
          <dep type="amod">
            <governor idx="7">approach</governor>
            <dependent idx="6">final</dependent>
          </dep>
          <dep type="prep_on">
            <governor idx="11">deviated</governor>
            <dependent idx="7">approach</dependent>
          </dep>
          <dep type="det">
            <governor idx="10">airplane</governor>
            <dependent idx="9">the</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="11">deviated</governor>
            <dependent idx="10">airplane</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="3">showed</governor>
            <dependent idx="11">deviated</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="11">deviated</governor>
            <dependent idx="12">laterally</dependent>
          </dep>
          <dep type="dep">
            <governor idx="12">laterally</governor>
            <dependent idx="14">S-turned</dependent>
          </dep>
          <dep type="det">
            <governor idx="20">course</governor>
            <dependent idx="17">the</dependent>
          </dep>
          <dep type="amod">
            <governor idx="20">course</governor>
            <dependent idx="18">final</dependent>
          </dep>
          <dep type="nn">
            <governor idx="20">course</governor>
            <dependent idx="19">approach</dependent>
          </dep>
          <dep type="prep_through">
            <governor idx="11">deviated</governor>
            <dependent idx="20">course</dependent>
          </dep>
          <dep type="mark">
            <governor idx="27">overcorrecting</governor>
            <dependent idx="23">if</dependent>
          </dep>
          <dep type="det">
            <governor idx="25">pilot</governor>
            <dependent idx="24">the</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="27">overcorrecting</governor>
            <dependent idx="25">pilot</dependent>
          </dep>
          <dep type="aux">
            <governor idx="27">overcorrecting</governor>
            <dependent idx="26">was</dependent>
          </dep>
          <dep type="prepc_as">
            <governor idx="20">course</governor>
            <dependent idx="27">overcorrecting</dependent>
          </dep>
        </dependencies>
      </sentence>
      <sentence id="5">
        <tokens>
          <token id="1">
            <word>At</word>
            <lemma>at</lemma>
            <CharacterOffsetBegin>388</CharacterOffsetBegin>
            <CharacterOffsetEnd>390</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="2">
            <word>about</word>
            <lemma>about</lemma>
            <CharacterOffsetBegin>391</CharacterOffsetBegin>
            <CharacterOffsetEnd>396</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="3">
            <word>400</word>
            <lemma>400</lemma>
            <CharacterOffsetBegin>397</CharacterOffsetBegin>
            <CharacterOffsetEnd>400</CharacterOffsetEnd>
            <POS>CD</POS>
            <NER>NUMBER</NER>
            <NormalizedNER>~400.0</NormalizedNER>
          </token>
          <token id="4">
            <word>'</word>
            <lemma>'</lemma>
            <CharacterOffsetBegin>400</CharacterOffsetBegin>
            <CharacterOffsetEnd>401</CharacterOffsetEnd>
            <POS>''</POS>
            <NER>O</NER>
          </token>
          <token id="5">
            <word>MSL</word>
            <lemma>MSL</lemma>
            <CharacterOffsetBegin>402</CharacterOffsetBegin>
            <CharacterOffsetEnd>405</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>ORGANIZATION</NER>
          </token>
          <token id="6">
            <word>&amp;</word>
            <lemma>&amp;</lemma>
            <CharacterOffsetBegin>406</CharacterOffsetBegin>
            <CharacterOffsetEnd>407</CharacterOffsetEnd>
            <POS>CC</POS>
            <NER>ORGANIZATION</NER>
          </token>
          <token id="7">
            <word>1</word>
            <lemma>1</lemma>
            <CharacterOffsetBegin>408</CharacterOffsetBegin>
            <CharacterOffsetEnd>409</CharacterOffsetEnd>
            <POS>CD</POS>
            <NER>NUMBER</NER>
            <NormalizedNER>1.0</NormalizedNER>
          </token>
          <token id="8">
            <word>mile</word>
            <lemma>mile</lemma>
            <CharacterOffsetBegin>410</CharacterOffsetBegin>
            <CharacterOffsetEnd>414</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="9">
            <word>from</word>
            <lemma>from</lemma>
            <CharacterOffsetBegin>415</CharacterOffsetBegin>
            <CharacterOffsetEnd>419</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="10">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>420</CharacterOffsetBegin>
            <CharacterOffsetEnd>423</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="11">
            <word>runway</word>
            <lemma>runway</lemma>
            <CharacterOffsetBegin>424</CharacterOffsetBegin>
            <CharacterOffsetEnd>430</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="12">
            <word>,</word>
            <lemma>,</lemma>
            <CharacterOffsetBegin>430</CharacterOffsetBegin>
            <CharacterOffsetEnd>431</CharacterOffsetEnd>
            <POS>,</POS>
            <NER>O</NER>
          </token>
          <token id="13">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>432</CharacterOffsetBegin>
            <CharacterOffsetEnd>435</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="14">
            <word>flight</word>
            <lemma>flight</lemma>
            <CharacterOffsetBegin>436</CharacterOffsetBegin>
            <CharacterOffsetEnd>442</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="15">
            <word>deviated</word>
            <lemma>deviate</lemma>
            <CharacterOffsetBegin>443</CharacterOffsetBegin>
            <CharacterOffsetEnd>451</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="16">
            <word>to</word>
            <lemma>to</lemma>
            <CharacterOffsetBegin>452</CharacterOffsetBegin>
            <CharacterOffsetEnd>454</CharacterOffsetEnd>
            <POS>TO</POS>
            <NER>O</NER>
          </token>
          <token id="17">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>455</CharacterOffsetBegin>
            <CharacterOffsetEnd>458</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="18">
            <word>left</word>
            <lemma>left</lemma>
            <CharacterOffsetBegin>459</CharacterOffsetBegin>
            <CharacterOffsetEnd>463</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="19">
            <word>&amp;</word>
            <lemma>&amp;</lemma>
            <CharacterOffsetBegin>464</CharacterOffsetBegin>
            <CharacterOffsetEnd>465</CharacterOffsetEnd>
            <POS>CC</POS>
            <NER>O</NER>
          </token>
          <token id="20">
            <word>began</word>
            <lemma>begin</lemma>
            <CharacterOffsetBegin>466</CharacterOffsetBegin>
            <CharacterOffsetEnd>471</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="21">
            <word>a</word>
            <lemma>a</lemma>
            <CharacterOffsetBegin>472</CharacterOffsetBegin>
            <CharacterOffsetEnd>473</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="22">
            <word>non-standard</word>
            <lemma>non-standard</lemma>
            <CharacterOffsetBegin>474</CharacterOffsetBegin>
            <CharacterOffsetEnd>486</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="23">
            <word>missed</word>
            <lemma>miss</lemma>
            <CharacterOffsetBegin>487</CharacterOffsetBegin>
            <CharacterOffsetEnd>493</CharacterOffsetEnd>
            <POS>VBN</POS>
            <NER>O</NER>
          </token>
          <token id="24">
            <word>approach</word>
            <lemma>approach</lemma>
            <CharacterOffsetBegin>494</CharacterOffsetBegin>
            <CharacterOffsetEnd>502</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="25">
            <word>.</word>
            <lemma>.</lemma>
            <CharacterOffsetBegin>502</CharacterOffsetBegin>
            <CharacterOffsetEnd>503</CharacterOffsetEnd>
            <POS>.</POS>
            <NER>O</NER>
          </token>
        </tokens>
        <parse>(ROOT (S (PP (IN At) (PP (IN about) (NP (NP (CD 400)) ('' ') (SBAR (S (NP (NP (NNP MSL)) (CC &amp;) (NP (NP (CD 1) (NN mile)) (PP (IN from) (NP (DT the) (NN runway)))) (, ,) (NP (DT the) (NN flight))) (VP (VBD deviated) (PP (TO to) (NP (DT the) (JJ left))))))))) (NP (CC &amp;)) (VP (VBD began) (NP (DT a) (JJ non-standard) (VBN missed) (NN approach))) (. .))) </parse>
        <dependencies type="basic-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="20">began</dependent>
          </dep>
          <dep type="prep">
            <governor idx="20">began</governor>
            <dependent idx="1">At</dependent>
          </dep>
          <dep type="pcomp">
            <governor idx="1">At</governor>
            <dependent idx="2">about</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="2">about</governor>
            <dependent idx="3">400</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="15">deviated</governor>
            <dependent idx="5">MSL</dependent>
          </dep>
          <dep type="cc">
            <governor idx="5">MSL</governor>
            <dependent idx="6">&amp;</dependent>
          </dep>
          <dep type="num">
            <governor idx="8">mile</governor>
            <dependent idx="7">1</dependent>
          </dep>
          <dep type="conj">
            <governor idx="5">MSL</governor>
            <dependent idx="8">mile</dependent>
          </dep>
          <dep type="prep">
            <governor idx="8">mile</governor>
            <dependent idx="9">from</dependent>
          </dep>
          <dep type="det">
            <governor idx="11">runway</governor>
            <dependent idx="10">the</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="9">from</governor>
            <dependent idx="11">runway</dependent>
          </dep>
          <dep type="det">
            <governor idx="14">flight</governor>
            <dependent idx="13">the</dependent>
          </dep>
          <dep type="conj">
            <governor idx="5">MSL</governor>
            <dependent idx="14">flight</dependent>
          </dep>
          <dep type="rcmod">
            <governor idx="3">400</governor>
            <dependent idx="15">deviated</dependent>
          </dep>
          <dep type="prep">
            <governor idx="15">deviated</governor>
            <dependent idx="16">to</dependent>
          </dep>
          <dep type="det">
            <governor idx="18">left</governor>
            <dependent idx="17">the</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="16">to</governor>
            <dependent idx="18">left</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="20">began</governor>
            <dependent idx="19">&amp;</dependent>
          </dep>
          <dep type="det">
            <governor idx="24">approach</governor>
            <dependent idx="21">a</dependent>
          </dep>
          <dep type="amod">
            <governor idx="24">approach</governor>
            <dependent idx="22">non-standard</dependent>
          </dep>
          <dep type="amod">
            <governor idx="24">approach</governor>
            <dependent idx="23">missed</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="20">began</governor>
            <dependent idx="24">approach</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="20">began</dependent>
          </dep>
          <dep type="prep">
            <governor idx="20">began</governor>
            <dependent idx="1">At</dependent>
          </dep>
          <dep type="pcomp">
            <governor idx="1">At</governor>
            <dependent idx="2">about</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="2">about</governor>
            <dependent idx="3">400</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="15">deviated</governor>
            <dependent idx="5">MSL</dependent>
          </dep>
          <dep type="num">
            <governor idx="8">mile</governor>
            <dependent idx="7">1</dependent>
          </dep>
          <dep type="conj_and">
            <governor idx="5">MSL</governor>
            <dependent idx="8">mile</dependent>
          </dep>
          <dep type="det">
            <governor idx="11">runway</governor>
            <dependent idx="10">the</dependent>
          </dep>
          <dep type="prep_from">
            <governor idx="8">mile</governor>
            <dependent idx="11">runway</dependent>
          </dep>
          <dep type="det">
            <governor idx="14">flight</governor>
            <dependent idx="13">the</dependent>
          </dep>
          <dep type="conj_and">
            <governor idx="5">MSL</governor>
            <dependent idx="14">flight</dependent>
          </dep>
          <dep type="rcmod">
            <governor idx="3">400</governor>
            <dependent idx="15">deviated</dependent>
          </dep>
          <dep type="det">
            <governor idx="18">left</governor>
            <dependent idx="17">the</dependent>
          </dep>
          <dep type="prep_to">
            <governor idx="15">deviated</governor>
            <dependent idx="18">left</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="20">began</governor>
            <dependent idx="19">&amp;</dependent>
          </dep>
          <dep type="det">
            <governor idx="24">approach</governor>
            <dependent idx="21">a</dependent>
          </dep>
          <dep type="amod">
            <governor idx="24">approach</governor>
            <dependent idx="22">non-standard</dependent>
          </dep>
          <dep type="amod">
            <governor idx="24">approach</governor>
            <dependent idx="23">missed</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="20">began</governor>
            <dependent idx="24">approach</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-ccprocessed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="20">began</dependent>
          </dep>
          <dep type="prep">
            <governor idx="20">began</governor>
            <dependent idx="1">At</dependent>
          </dep>
          <dep type="pcomp">
            <governor idx="1">At</governor>
            <dependent idx="2">about</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="2">about</governor>
            <dependent idx="3">400</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="15">deviated</governor>
            <dependent idx="5">MSL</dependent>
          </dep>
          <dep type="num">
            <governor idx="8">mile</governor>
            <dependent idx="7">1</dependent>
          </dep>
          <dep type="conj_and">
            <governor idx="5">MSL</governor>
            <dependent idx="8">mile</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="15">deviated</governor>
            <dependent idx="8">mile</dependent>
          </dep>
          <dep type="det">
            <governor idx="11">runway</governor>
            <dependent idx="10">the</dependent>
          </dep>
          <dep type="prep_from">
            <governor idx="8">mile</governor>
            <dependent idx="11">runway</dependent>
          </dep>
          <dep type="det">
            <governor idx="14">flight</governor>
            <dependent idx="13">the</dependent>
          </dep>
          <dep type="conj_and">
            <governor idx="5">MSL</governor>
            <dependent idx="14">flight</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="15">deviated</governor>
            <dependent idx="14">flight</dependent>
          </dep>
          <dep type="rcmod">
            <governor idx="3">400</governor>
            <dependent idx="15">deviated</dependent>
          </dep>
          <dep type="det">
            <governor idx="18">left</governor>
            <dependent idx="17">the</dependent>
          </dep>
          <dep type="prep_to">
            <governor idx="15">deviated</governor>
            <dependent idx="18">left</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="20">began</governor>
            <dependent idx="19">&amp;</dependent>
          </dep>
          <dep type="det">
            <governor idx="24">approach</governor>
            <dependent idx="21">a</dependent>
          </dep>
          <dep type="amod">
            <governor idx="24">approach</governor>
            <dependent idx="22">non-standard</dependent>
          </dep>
          <dep type="amod">
            <governor idx="24">approach</governor>
            <dependent idx="23">missed</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="20">began</governor>
            <dependent idx="24">approach</dependent>
          </dep>
        </dependencies>
      </sentence>
      <sentence id="6">
        <tokens>
          <token id="1">
            <word>An</word>
            <lemma>a</lemma>
            <CharacterOffsetBegin>505</CharacterOffsetBegin>
            <CharacterOffsetEnd>507</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="2">
            <word>alternate</word>
            <lemma>alternate</lemma>
            <CharacterOffsetBegin>508</CharacterOffsetBegin>
            <CharacterOffsetEnd>517</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="3">
            <word>IFR</word>
            <lemma>IFR</lemma>
            <CharacterOffsetBegin>518</CharacterOffsetBegin>
            <CharacterOffsetEnd>521</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>ORGANIZATION</NER>
          </token>
          <token id="4">
            <word>missed</word>
            <lemma>miss</lemma>
            <CharacterOffsetBegin>522</CharacterOffsetBegin>
            <CharacterOffsetEnd>528</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="5">
            <word>approach</word>
            <lemma>approach</lemma>
            <CharacterOffsetBegin>529</CharacterOffsetBegin>
            <CharacterOffsetEnd>537</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="6">
            <word>procedure</word>
            <lemma>procedure</lemma>
            <CharacterOffsetBegin>538</CharacterOffsetBegin>
            <CharacterOffsetEnd>547</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="7">
            <word>was</word>
            <lemma>be</lemma>
            <CharacterOffsetBegin>548</CharacterOffsetBegin>
            <CharacterOffsetEnd>551</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="8">
            <word>issued</word>
            <lemma>issue</lemma>
            <CharacterOffsetBegin>552</CharacterOffsetBegin>
            <CharacterOffsetEnd>558</CharacterOffsetEnd>
            <POS>VBN</POS>
            <NER>O</NER>
          </token>
          <token id="9">
            <word>&amp;</word>
            <lemma>&amp;</lemma>
            <CharacterOffsetBegin>559</CharacterOffsetBegin>
            <CharacterOffsetEnd>560</CharacterOffsetEnd>
            <POS>CC</POS>
            <NER>O</NER>
          </token>
          <token id="10">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>561</CharacterOffsetBegin>
            <CharacterOffsetEnd>564</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="11">
            <word>pilot</word>
            <lemma>pilot</lemma>
            <CharacterOffsetBegin>565</CharacterOffsetBegin>
            <CharacterOffsetEnd>570</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="12">
            <word>-LRB-</word>
            <lemma>-lrb-</lemma>
            <CharacterOffsetBegin>570</CharacterOffsetBegin>
            <CharacterOffsetEnd>571</CharacterOffsetEnd>
            <POS>-LRB-</POS>
            <NER>O</NER>
          </token>
          <token id="13">
            <word>s</word>
            <lemma>s</lemma>
            <CharacterOffsetBegin>571</CharacterOffsetBegin>
            <CharacterOffsetEnd>572</CharacterOffsetEnd>
            <POS>NNS</POS>
            <NER>O</NER>
          </token>
          <token id="14">
            <word>-RRB-</word>
            <lemma>-rrb-</lemma>
            <CharacterOffsetBegin>572</CharacterOffsetBegin>
            <CharacterOffsetEnd>573</CharacterOffsetEnd>
            <POS>-RRB-</POS>
            <NER>O</NER>
          </token>
          <token id="15">
            <word>acknowledged</word>
            <lemma>acknowledge</lemma>
            <CharacterOffsetBegin>574</CharacterOffsetBegin>
            <CharacterOffsetEnd>586</CharacterOffsetEnd>
            <POS>VBN</POS>
            <NER>O</NER>
          </token>
          <token id="16">
            <word>by</word>
            <lemma>by</lemma>
            <CharacterOffsetBegin>587</CharacterOffsetBegin>
            <CharacterOffsetEnd>589</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="17">
            <word>clicking</word>
            <lemma>click</lemma>
            <CharacterOffsetBegin>590</CharacterOffsetBegin>
            <CharacterOffsetEnd>598</CharacterOffsetEnd>
            <POS>VBG</POS>
            <NER>O</NER>
          </token>
          <token id="18">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>599</CharacterOffsetBegin>
            <CharacterOffsetEnd>602</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="19">
            <word>mike</word>
            <lemma>mike</lemma>
            <CharacterOffsetBegin>603</CharacterOffsetBegin>
            <CharacterOffsetEnd>607</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="20">
            <word>button</word>
            <lemma>button</lemma>
            <CharacterOffsetBegin>608</CharacterOffsetBegin>
            <CharacterOffsetEnd>614</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="21">
            <word>;</word>
            <lemma>;</lemma>
            <CharacterOffsetBegin>614</CharacterOffsetBegin>
            <CharacterOffsetEnd>615</CharacterOffsetEnd>
            <POS>:</POS>
            <NER>O</NER>
          </token>
          <token id="22">
            <word>however</word>
            <lemma>however</lemma>
            <CharacterOffsetBegin>616</CharacterOffsetBegin>
            <CharacterOffsetEnd>623</CharacterOffsetEnd>
            <POS>RB</POS>
            <NER>O</NER>
          </token>
          <token id="23">
            <word>,</word>
            <lemma>,</lemma>
            <CharacterOffsetBegin>623</CharacterOffsetBegin>
            <CharacterOffsetEnd>624</CharacterOffsetEnd>
            <POS>,</POS>
            <NER>O</NER>
          </token>
          <token id="24">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>625</CharacterOffsetBegin>
            <CharacterOffsetEnd>628</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="25">
            <word>alternate</word>
            <lemma>alternate</lemma>
            <CharacterOffsetBegin>629</CharacterOffsetBegin>
            <CharacterOffsetEnd>638</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="26">
            <word>missed</word>
            <lemma>miss</lemma>
            <CharacterOffsetBegin>639</CharacterOffsetBegin>
            <CharacterOffsetEnd>645</CharacterOffsetEnd>
            <POS>VBN</POS>
            <NER>O</NER>
          </token>
          <token id="27">
            <word>approach</word>
            <lemma>approach</lemma>
            <CharacterOffsetBegin>646</CharacterOffsetBegin>
            <CharacterOffsetEnd>654</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="28">
            <word>instruction</word>
            <lemma>instruction</lemma>
            <CharacterOffsetBegin>655</CharacterOffsetBegin>
            <CharacterOffsetEnd>666</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="29">
            <word>was</word>
            <lemma>be</lemma>
            <CharacterOffsetBegin>667</CharacterOffsetBegin>
            <CharacterOffsetEnd>670</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="30">
            <word>not</word>
            <lemma>not</lemma>
            <CharacterOffsetBegin>671</CharacterOffsetBegin>
            <CharacterOffsetEnd>674</CharacterOffsetEnd>
            <POS>RB</POS>
            <NER>O</NER>
          </token>
          <token id="31">
            <word>followed</word>
            <lemma>follow</lemma>
            <CharacterOffsetBegin>675</CharacterOffsetBegin>
            <CharacterOffsetEnd>683</CharacterOffsetEnd>
            <POS>VBN</POS>
            <NER>O</NER>
          </token>
          <token id="32">
            <word>.</word>
            <lemma>.</lemma>
            <CharacterOffsetBegin>683</CharacterOffsetBegin>
            <CharacterOffsetEnd>684</CharacterOffsetEnd>
            <POS>.</POS>
            <NER>O</NER>
          </token>
        </tokens>
        <parse>(ROOT (S (S (NP (DT An) (JJ alternate) (NNP IFR)) (VP (VBD missed) (SBAR (S (NP (NN approach) (NN procedure)) (VP (VBD was) (VP (VBN issued) (S (NP (CC &amp;) (NP (NP (DT the) (NN pilot)) (PRN (-LRB- -LRB-) (NNS s) (-RRB- -RRB-)))) (VP (VBN acknowledged) (PP (IN by) (S (VP (VBG clicking) (NP (DT the) (JJ mike) (NN button))))))))))))) (: ;) (S (ADVP (RB however)) (, ,) (NP (DT the) (JJ alternate) (VBN missed) (NN approach) (NN instruction)) (VP (VBD was) (RB not) (VP (VBN followed)))) (. .))) </parse>
        <dependencies type="basic-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="4">missed</dependent>
          </dep>
          <dep type="det">
            <governor idx="3">IFR</governor>
            <dependent idx="1">An</dependent>
          </dep>
          <dep type="amod">
            <governor idx="3">IFR</governor>
            <dependent idx="2">alternate</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="4">missed</governor>
            <dependent idx="3">IFR</dependent>
          </dep>
          <dep type="nn">
            <governor idx="6">procedure</governor>
            <dependent idx="5">approach</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="8">issued</governor>
            <dependent idx="6">procedure</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="8">issued</governor>
            <dependent idx="7">was</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="4">missed</governor>
            <dependent idx="8">issued</dependent>
          </dep>
          <dep type="cc">
            <governor idx="11">pilot</governor>
            <dependent idx="9">&amp;</dependent>
          </dep>
          <dep type="det">
            <governor idx="11">pilot</governor>
            <dependent idx="10">the</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="15">acknowledged</governor>
            <dependent idx="11">pilot</dependent>
          </dep>
          <dep type="dep">
            <governor idx="11">pilot</governor>
            <dependent idx="13">s</dependent>
          </dep>
          <dep type="dep">
            <governor idx="8">issued</governor>
            <dependent idx="15">acknowledged</dependent>
          </dep>
          <dep type="prep">
            <governor idx="15">acknowledged</governor>
            <dependent idx="16">by</dependent>
          </dep>
          <dep type="pcomp">
            <governor idx="16">by</governor>
            <dependent idx="17">clicking</dependent>
          </dep>
          <dep type="det">
            <governor idx="20">button</governor>
            <dependent idx="18">the</dependent>
          </dep>
          <dep type="amod">
            <governor idx="20">button</governor>
            <dependent idx="19">mike</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="17">clicking</governor>
            <dependent idx="20">button</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="31">followed</governor>
            <dependent idx="22">however</dependent>
          </dep>
          <dep type="det">
            <governor idx="28">instruction</governor>
            <dependent idx="24">the</dependent>
          </dep>
          <dep type="amod">
            <governor idx="28">instruction</governor>
            <dependent idx="25">alternate</dependent>
          </dep>
          <dep type="amod">
            <governor idx="28">instruction</governor>
            <dependent idx="26">missed</dependent>
          </dep>
          <dep type="nn">
            <governor idx="28">instruction</governor>
            <dependent idx="27">approach</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="31">followed</governor>
            <dependent idx="28">instruction</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="31">followed</governor>
            <dependent idx="29">was</dependent>
          </dep>
          <dep type="neg">
            <governor idx="31">followed</governor>
            <dependent idx="30">not</dependent>
          </dep>
          <dep type="parataxis">
            <governor idx="4">missed</governor>
            <dependent idx="31">followed</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="4">missed</dependent>
          </dep>
          <dep type="det">
            <governor idx="3">IFR</governor>
            <dependent idx="1">An</dependent>
          </dep>
          <dep type="amod">
            <governor idx="3">IFR</governor>
            <dependent idx="2">alternate</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="4">missed</governor>
            <dependent idx="3">IFR</dependent>
          </dep>
          <dep type="nn">
            <governor idx="6">procedure</governor>
            <dependent idx="5">approach</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="8">issued</governor>
            <dependent idx="6">procedure</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="8">issued</governor>
            <dependent idx="7">was</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="4">missed</governor>
            <dependent idx="8">issued</dependent>
          </dep>
          <dep type="cc">
            <governor idx="11">pilot</governor>
            <dependent idx="9">&amp;</dependent>
          </dep>
          <dep type="det">
            <governor idx="11">pilot</governor>
            <dependent idx="10">the</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="15">acknowledged</governor>
            <dependent idx="11">pilot</dependent>
          </dep>
          <dep type="dep">
            <governor idx="11">pilot</governor>
            <dependent idx="13">s</dependent>
          </dep>
          <dep type="dep">
            <governor idx="8">issued</governor>
            <dependent idx="15">acknowledged</dependent>
          </dep>
          <dep type="prepc_by">
            <governor idx="15">acknowledged</governor>
            <dependent idx="17">clicking</dependent>
          </dep>
          <dep type="det">
            <governor idx="20">button</governor>
            <dependent idx="18">the</dependent>
          </dep>
          <dep type="amod">
            <governor idx="20">button</governor>
            <dependent idx="19">mike</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="17">clicking</governor>
            <dependent idx="20">button</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="31">followed</governor>
            <dependent idx="22">however</dependent>
          </dep>
          <dep type="det">
            <governor idx="28">instruction</governor>
            <dependent idx="24">the</dependent>
          </dep>
          <dep type="amod">
            <governor idx="28">instruction</governor>
            <dependent idx="25">alternate</dependent>
          </dep>
          <dep type="amod">
            <governor idx="28">instruction</governor>
            <dependent idx="26">missed</dependent>
          </dep>
          <dep type="nn">
            <governor idx="28">instruction</governor>
            <dependent idx="27">approach</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="31">followed</governor>
            <dependent idx="28">instruction</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="31">followed</governor>
            <dependent idx="29">was</dependent>
          </dep>
          <dep type="neg">
            <governor idx="31">followed</governor>
            <dependent idx="30">not</dependent>
          </dep>
          <dep type="parataxis">
            <governor idx="4">missed</governor>
            <dependent idx="31">followed</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-ccprocessed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="4">missed</dependent>
          </dep>
          <dep type="det">
            <governor idx="3">IFR</governor>
            <dependent idx="1">An</dependent>
          </dep>
          <dep type="amod">
            <governor idx="3">IFR</governor>
            <dependent idx="2">alternate</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="4">missed</governor>
            <dependent idx="3">IFR</dependent>
          </dep>
          <dep type="nn">
            <governor idx="6">procedure</governor>
            <dependent idx="5">approach</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="8">issued</governor>
            <dependent idx="6">procedure</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="8">issued</governor>
            <dependent idx="7">was</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="4">missed</governor>
            <dependent idx="8">issued</dependent>
          </dep>
          <dep type="cc">
            <governor idx="11">pilot</governor>
            <dependent idx="9">&amp;</dependent>
          </dep>
          <dep type="det">
            <governor idx="11">pilot</governor>
            <dependent idx="10">the</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="15">acknowledged</governor>
            <dependent idx="11">pilot</dependent>
          </dep>
          <dep type="dep">
            <governor idx="11">pilot</governor>
            <dependent idx="13">s</dependent>
          </dep>
          <dep type="dep">
            <governor idx="8">issued</governor>
            <dependent idx="15">acknowledged</dependent>
          </dep>
          <dep type="prepc_by">
            <governor idx="15">acknowledged</governor>
            <dependent idx="17">clicking</dependent>
          </dep>
          <dep type="det">
            <governor idx="20">button</governor>
            <dependent idx="18">the</dependent>
          </dep>
          <dep type="amod">
            <governor idx="20">button</governor>
            <dependent idx="19">mike</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="17">clicking</governor>
            <dependent idx="20">button</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="31">followed</governor>
            <dependent idx="22">however</dependent>
          </dep>
          <dep type="det">
            <governor idx="28">instruction</governor>
            <dependent idx="24">the</dependent>
          </dep>
          <dep type="amod">
            <governor idx="28">instruction</governor>
            <dependent idx="25">alternate</dependent>
          </dep>
          <dep type="amod">
            <governor idx="28">instruction</governor>
            <dependent idx="26">missed</dependent>
          </dep>
          <dep type="nn">
            <governor idx="28">instruction</governor>
            <dependent idx="27">approach</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="31">followed</governor>
            <dependent idx="28">instruction</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="31">followed</governor>
            <dependent idx="29">was</dependent>
          </dep>
          <dep type="neg">
            <governor idx="31">followed</governor>
            <dependent idx="30">not</dependent>
          </dep>
          <dep type="parataxis">
            <governor idx="4">missed</governor>
            <dependent idx="31">followed</dependent>
          </dep>
        </dependencies>
      </sentence>
      <sentence id="7">
        <tokens>
          <token id="1">
            <word>Radar</word>
            <lemma>radar</lemma>
            <CharacterOffsetBegin>686</CharacterOffsetBegin>
            <CharacterOffsetEnd>691</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="2">
            <word>data</word>
            <lemma>datum</lemma>
            <CharacterOffsetBegin>692</CharacterOffsetBegin>
            <CharacterOffsetEnd>696</CharacterOffsetEnd>
            <POS>NNS</POS>
            <NER>O</NER>
          </token>
          <token id="3">
            <word>showed</word>
            <lemma>show</lemma>
            <CharacterOffsetBegin>697</CharacterOffsetBegin>
            <CharacterOffsetEnd>703</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="4">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>704</CharacterOffsetBegin>
            <CharacterOffsetEnd>707</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="5">
            <word>airplane</word>
            <lemma>airplane</lemma>
            <CharacterOffsetBegin>708</CharacterOffsetBegin>
            <CharacterOffsetEnd>716</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="6">
            <word>flew</word>
            <lemma>fly</lemma>
            <CharacterOffsetBegin>717</CharacterOffsetBegin>
            <CharacterOffsetEnd>721</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="7">
            <word>a</word>
            <lemma>a</lemma>
            <CharacterOffsetBegin>722</CharacterOffsetBegin>
            <CharacterOffsetEnd>723</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="8">
            <word>southerly</word>
            <lemma>southerly</lemma>
            <CharacterOffsetBegin>724</CharacterOffsetBegin>
            <CharacterOffsetEnd>733</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="9">
            <word>course</word>
            <lemma>course</lemma>
            <CharacterOffsetBegin>734</CharacterOffsetBegin>
            <CharacterOffsetEnd>740</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="10">
            <word>for</word>
            <lemma>for</lemma>
            <CharacterOffsetBegin>741</CharacterOffsetBegin>
            <CharacterOffsetEnd>744</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="11">
            <word>about</word>
            <lemma>about</lemma>
            <CharacterOffsetBegin>745</CharacterOffsetBegin>
            <CharacterOffsetEnd>750</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="12">
            <word>2-1\/2</word>
            <lemma>2-1\/2</lemma>
            <CharacterOffsetBegin>751</CharacterOffsetBegin>
            <CharacterOffsetEnd>756</CharacterOffsetEnd>
            <POS>CD</POS>
            <NER>NUMBER</NER>
          </token>
          <token id="13">
            <word>miles</word>
            <lemma>mile</lemma>
            <CharacterOffsetBegin>757</CharacterOffsetBegin>
            <CharacterOffsetEnd>762</CharacterOffsetEnd>
            <POS>NNS</POS>
            <NER>O</NER>
          </token>
          <token id="14">
            <word>,</word>
            <lemma>,</lemma>
            <CharacterOffsetBegin>762</CharacterOffsetBegin>
            <CharacterOffsetEnd>763</CharacterOffsetEnd>
            <POS>,</POS>
            <NER>O</NER>
          </token>
          <token id="15">
            <word>then</word>
            <lemma>then</lemma>
            <CharacterOffsetBegin>764</CharacterOffsetBegin>
            <CharacterOffsetEnd>768</CharacterOffsetEnd>
            <POS>RB</POS>
            <NER>O</NER>
          </token>
          <token id="16">
            <word>it</word>
            <lemma>it</lemma>
            <CharacterOffsetBegin>769</CharacterOffsetBegin>
            <CharacterOffsetEnd>771</CharacterOffsetEnd>
            <POS>PRP</POS>
            <NER>O</NER>
          </token>
          <token id="17">
            <word>began</word>
            <lemma>begin</lemma>
            <CharacterOffsetBegin>772</CharacterOffsetBegin>
            <CharacterOffsetEnd>777</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="18">
            <word>turning</word>
            <lemma>turn</lemma>
            <CharacterOffsetBegin>778</CharacterOffsetBegin>
            <CharacterOffsetEnd>785</CharacterOffsetEnd>
            <POS>VBG</POS>
            <NER>O</NER>
          </token>
          <token id="19">
            <word>&amp;</word>
            <lemma>&amp;</lemma>
            <CharacterOffsetBegin>786</CharacterOffsetBegin>
            <CharacterOffsetEnd>787</CharacterOffsetEnd>
            <POS>CC</POS>
            <NER>O</NER>
          </token>
          <token id="20">
            <word>radar</word>
            <lemma>radar</lemma>
            <CharacterOffsetBegin>788</CharacterOffsetBegin>
            <CharacterOffsetEnd>793</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="21">
            <word>contact</word>
            <lemma>contact</lemma>
            <CharacterOffsetBegin>794</CharacterOffsetBegin>
            <CharacterOffsetEnd>801</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="22">
            <word>was</word>
            <lemma>be</lemma>
            <CharacterOffsetBegin>802</CharacterOffsetBegin>
            <CharacterOffsetEnd>805</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="23">
            <word>lost</word>
            <lemma>lose</lemma>
            <CharacterOffsetBegin>806</CharacterOffsetBegin>
            <CharacterOffsetEnd>810</CharacterOffsetEnd>
            <POS>VBN</POS>
            <NER>O</NER>
          </token>
          <token id="24">
            <word>.</word>
            <lemma>.</lemma>
            <CharacterOffsetBegin>810</CharacterOffsetBegin>
            <CharacterOffsetEnd>811</CharacterOffsetEnd>
            <POS>.</POS>
            <NER>O</NER>
          </token>
        </tokens>
        <parse>(ROOT (S (S (NP (NN Radar) (NNS data)) (VP (VBD showed) (SBAR (S (NP (DT the) (NN airplane)) (VP (VBD flew) (NP (DT a) (JJ southerly) (NN course)) (PP (IN for) (PP (IN about) (NP (CD 2-1\/2) (NNS miles))))))))) (, ,) (RB then) (S (NP (PRP it)) (VP (VBD began) (S (VP (VBG turning) (NP (NP (CC &amp;) (NN radar)) (SBAR (S (NP (NN contact)) (VP (VBD was) (VP (VBN lost)))))))))) (. .))) </parse>
        <dependencies type="basic-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="3">showed</dependent>
          </dep>
          <dep type="nn">
            <governor idx="2">data</governor>
            <dependent idx="1">Radar</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="3">showed</governor>
            <dependent idx="2">data</dependent>
          </dep>
          <dep type="det">
            <governor idx="5">airplane</governor>
            <dependent idx="4">the</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="6">flew</governor>
            <dependent idx="5">airplane</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="3">showed</governor>
            <dependent idx="6">flew</dependent>
          </dep>
          <dep type="det">
            <governor idx="9">course</governor>
            <dependent idx="7">a</dependent>
          </dep>
          <dep type="amod">
            <governor idx="9">course</governor>
            <dependent idx="8">southerly</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="6">flew</governor>
            <dependent idx="9">course</dependent>
          </dep>
          <dep type="prep">
            <governor idx="6">flew</governor>
            <dependent idx="10">for</dependent>
          </dep>
          <dep type="pcomp">
            <governor idx="10">for</governor>
            <dependent idx="11">about</dependent>
          </dep>
          <dep type="num">
            <governor idx="13">miles</governor>
            <dependent idx="12">2-1\/2</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="11">about</governor>
            <dependent idx="13">miles</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="3">showed</governor>
            <dependent idx="15">then</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="17">began</governor>
            <dependent idx="16">it</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="3">showed</governor>
            <dependent idx="17">began</dependent>
          </dep>
          <dep type="xcomp">
            <governor idx="17">began</governor>
            <dependent idx="18">turning</dependent>
          </dep>
          <dep type="cc">
            <governor idx="20">radar</governor>
            <dependent idx="19">&amp;</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="18">turning</governor>
            <dependent idx="20">radar</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="23">lost</governor>
            <dependent idx="21">contact</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="23">lost</governor>
            <dependent idx="22">was</dependent>
          </dep>
          <dep type="rcmod">
            <governor idx="20">radar</governor>
            <dependent idx="23">lost</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="3">showed</dependent>
          </dep>
          <dep type="nn">
            <governor idx="2">data</governor>
            <dependent idx="1">Radar</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="3">showed</governor>
            <dependent idx="2">data</dependent>
          </dep>
          <dep type="det">
            <governor idx="5">airplane</governor>
            <dependent idx="4">the</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="6">flew</governor>
            <dependent idx="5">airplane</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="3">showed</governor>
            <dependent idx="6">flew</dependent>
          </dep>
          <dep type="det">
            <governor idx="9">course</governor>
            <dependent idx="7">a</dependent>
          </dep>
          <dep type="amod">
            <governor idx="9">course</governor>
            <dependent idx="8">southerly</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="6">flew</governor>
            <dependent idx="9">course</dependent>
          </dep>
          <dep type="prep">
            <governor idx="6">flew</governor>
            <dependent idx="10">for</dependent>
          </dep>
          <dep type="pcomp">
            <governor idx="10">for</governor>
            <dependent idx="11">about</dependent>
          </dep>
          <dep type="num">
            <governor idx="13">miles</governor>
            <dependent idx="12">2-1\/2</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="11">about</governor>
            <dependent idx="13">miles</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="3">showed</governor>
            <dependent idx="15">then</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="17">began</governor>
            <dependent idx="16">it</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="3">showed</governor>
            <dependent idx="17">began</dependent>
          </dep>
          <dep type="xcomp">
            <governor idx="17">began</governor>
            <dependent idx="18">turning</dependent>
          </dep>
          <dep type="cc">
            <governor idx="20">radar</governor>
            <dependent idx="19">&amp;</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="18">turning</governor>
            <dependent idx="20">radar</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="23">lost</governor>
            <dependent idx="21">contact</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="23">lost</governor>
            <dependent idx="22">was</dependent>
          </dep>
          <dep type="rcmod">
            <governor idx="20">radar</governor>
            <dependent idx="23">lost</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-ccprocessed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="3">showed</dependent>
          </dep>
          <dep type="nn">
            <governor idx="2">data</governor>
            <dependent idx="1">Radar</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="3">showed</governor>
            <dependent idx="2">data</dependent>
          </dep>
          <dep type="det">
            <governor idx="5">airplane</governor>
            <dependent idx="4">the</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="6">flew</governor>
            <dependent idx="5">airplane</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="3">showed</governor>
            <dependent idx="6">flew</dependent>
          </dep>
          <dep type="det">
            <governor idx="9">course</governor>
            <dependent idx="7">a</dependent>
          </dep>
          <dep type="amod">
            <governor idx="9">course</governor>
            <dependent idx="8">southerly</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="6">flew</governor>
            <dependent idx="9">course</dependent>
          </dep>
          <dep type="prep">
            <governor idx="6">flew</governor>
            <dependent idx="10">for</dependent>
          </dep>
          <dep type="pcomp">
            <governor idx="10">for</governor>
            <dependent idx="11">about</dependent>
          </dep>
          <dep type="num">
            <governor idx="13">miles</governor>
            <dependent idx="12">2-1\/2</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="11">about</governor>
            <dependent idx="13">miles</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="3">showed</governor>
            <dependent idx="15">then</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="17">began</governor>
            <dependent idx="16">it</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="3">showed</governor>
            <dependent idx="17">began</dependent>
          </dep>
          <dep type="xcomp">
            <governor idx="17">began</governor>
            <dependent idx="18">turning</dependent>
          </dep>
          <dep type="cc">
            <governor idx="20">radar</governor>
            <dependent idx="19">&amp;</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="18">turning</governor>
            <dependent idx="20">radar</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="23">lost</governor>
            <dependent idx="21">contact</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="23">lost</governor>
            <dependent idx="22">was</dependent>
          </dep>
          <dep type="rcmod">
            <governor idx="20">radar</governor>
            <dependent idx="23">lost</dependent>
          </dep>
        </dependencies>
      </sentence>
      <sentence id="8">
        <tokens>
          <token id="1">
            <word>Witnesses</word>
            <lemma>witness</lemma>
            <CharacterOffsetBegin>813</CharacterOffsetBegin>
            <CharacterOffsetEnd>822</CharacterOffsetEnd>
            <POS>NNS</POS>
            <NER>O</NER>
          </token>
          <token id="2">
            <word>saw</word>
            <lemma>see</lemma>
            <CharacterOffsetBegin>823</CharacterOffsetBegin>
            <CharacterOffsetEnd>826</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="3">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>827</CharacterOffsetBegin>
            <CharacterOffsetEnd>830</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="4">
            <word>airplane</word>
            <lemma>airplane</lemma>
            <CharacterOffsetBegin>831</CharacterOffsetBegin>
            <CharacterOffsetEnd>839</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="5">
            <word>descending</word>
            <lemma>descend</lemma>
            <CharacterOffsetBegin>840</CharacterOffsetBegin>
            <CharacterOffsetEnd>850</CharacterOffsetEnd>
            <POS>VBG</POS>
            <NER>O</NER>
          </token>
          <token id="6">
            <word>through</word>
            <lemma>through</lemma>
            <CharacterOffsetBegin>851</CharacterOffsetBegin>
            <CharacterOffsetEnd>858</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="7">
            <word>fog</word>
            <lemma>fog</lemma>
            <CharacterOffsetBegin>859</CharacterOffsetBegin>
            <CharacterOffsetEnd>862</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="8">
            <word>in</word>
            <lemma>in</lemma>
            <CharacterOffsetBegin>863</CharacterOffsetBegin>
            <CharacterOffsetEnd>865</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="9">
            <word>a</word>
            <lemma>a</lemma>
            <CharacterOffsetBegin>866</CharacterOffsetBegin>
            <CharacterOffsetEnd>867</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="10">
            <word>steep</word>
            <lemma>steep</lemma>
            <CharacterOffsetBegin>868</CharacterOffsetBegin>
            <CharacterOffsetEnd>873</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="11">
            <word>,</word>
            <lemma>,</lemma>
            <CharacterOffsetBegin>873</CharacterOffsetBegin>
            <CharacterOffsetEnd>874</CharacterOffsetEnd>
            <POS>,</POS>
            <NER>O</NER>
          </token>
          <token id="12">
            <word>nose</word>
            <lemma>nose</lemma>
            <CharacterOffsetBegin>875</CharacterOffsetBegin>
            <CharacterOffsetEnd>879</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="13">
            <word>down</word>
            <lemma>down</lemma>
            <CharacterOffsetBegin>880</CharacterOffsetBegin>
            <CharacterOffsetEnd>884</CharacterOffsetEnd>
            <POS>RB</POS>
            <NER>O</NER>
          </token>
          <token id="14">
            <word>,</word>
            <lemma>,</lemma>
            <CharacterOffsetBegin>884</CharacterOffsetBegin>
            <CharacterOffsetEnd>885</CharacterOffsetEnd>
            <POS>,</POS>
            <NER>O</NER>
          </token>
          <token id="15">
            <word>right</word>
            <lemma>right</lemma>
            <CharacterOffsetBegin>886</CharacterOffsetBegin>
            <CharacterOffsetEnd>891</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="16">
            <word>bank</word>
            <lemma>bank</lemma>
            <CharacterOffsetBegin>892</CharacterOffsetBegin>
            <CharacterOffsetEnd>896</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="17">
            <word>attitude</word>
            <lemma>attitude</lemma>
            <CharacterOffsetBegin>897</CharacterOffsetBegin>
            <CharacterOffsetEnd>905</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="18">
            <word>before</word>
            <lemma>before</lemma>
            <CharacterOffsetBegin>906</CharacterOffsetBegin>
            <CharacterOffsetEnd>912</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="19">
            <word>disappearing</word>
            <lemma>disappear</lemma>
            <CharacterOffsetBegin>913</CharacterOffsetBegin>
            <CharacterOffsetEnd>925</CharacterOffsetEnd>
            <POS>VBG</POS>
            <NER>O</NER>
          </token>
          <token id="20">
            <word>from</word>
            <lemma>from</lemma>
            <CharacterOffsetBegin>926</CharacterOffsetBegin>
            <CharacterOffsetEnd>930</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="21">
            <word>view</word>
            <lemma>view</lemma>
            <CharacterOffsetBegin>931</CharacterOffsetBegin>
            <CharacterOffsetEnd>935</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="22">
            <word>.</word>
            <lemma>.</lemma>
            <CharacterOffsetBegin>935</CharacterOffsetBegin>
            <CharacterOffsetEnd>936</CharacterOffsetEnd>
            <POS>.</POS>
            <NER>O</NER>
          </token>
        </tokens>
        <parse>(ROOT (S (NP (NNS Witnesses)) (VP (VBD saw) (NP (NP (NP (DT the) (NN airplane)) (VP (VBG descending) (PP (IN through) (NP (NN fog))) (PP (IN in) (NP (NP (DT a) (JJ steep)) (, ,) (NP (NN nose)))) (ADVP (RB down)))) (, ,) (NP (NP (JJ right) (NN bank) (NN attitude)) (PP (IN before) (S (VP (VBG disappearing) (PP (IN from) (NP (NN view))))))))) (. .))) </parse>
        <dependencies type="basic-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="2">saw</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="2">saw</governor>
            <dependent idx="1">Witnesses</dependent>
          </dep>
          <dep type="det">
            <governor idx="4">airplane</governor>
            <dependent idx="3">the</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="2">saw</governor>
            <dependent idx="4">airplane</dependent>
          </dep>
          <dep type="partmod">
            <governor idx="4">airplane</governor>
            <dependent idx="5">descending</dependent>
          </dep>
          <dep type="prep">
            <governor idx="5">descending</governor>
            <dependent idx="6">through</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="6">through</governor>
            <dependent idx="7">fog</dependent>
          </dep>
          <dep type="prep">
            <governor idx="5">descending</governor>
            <dependent idx="8">in</dependent>
          </dep>
          <dep type="det">
            <governor idx="10">steep</governor>
            <dependent idx="9">a</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="8">in</governor>
            <dependent idx="10">steep</dependent>
          </dep>
          <dep type="appos">
            <governor idx="10">steep</governor>
            <dependent idx="12">nose</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="5">descending</governor>
            <dependent idx="13">down</dependent>
          </dep>
          <dep type="amod">
            <governor idx="17">attitude</governor>
            <dependent idx="15">right</dependent>
          </dep>
          <dep type="nn">
            <governor idx="17">attitude</governor>
            <dependent idx="16">bank</dependent>
          </dep>
          <dep type="appos">
            <governor idx="4">airplane</governor>
            <dependent idx="17">attitude</dependent>
          </dep>
          <dep type="prep">
            <governor idx="17">attitude</governor>
            <dependent idx="18">before</dependent>
          </dep>
          <dep type="pcomp">
            <governor idx="18">before</governor>
            <dependent idx="19">disappearing</dependent>
          </dep>
          <dep type="prep">
            <governor idx="19">disappearing</governor>
            <dependent idx="20">from</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="20">from</governor>
            <dependent idx="21">view</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="2">saw</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="2">saw</governor>
            <dependent idx="1">Witnesses</dependent>
          </dep>
          <dep type="det">
            <governor idx="4">airplane</governor>
            <dependent idx="3">the</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="2">saw</governor>
            <dependent idx="4">airplane</dependent>
          </dep>
          <dep type="partmod">
            <governor idx="4">airplane</governor>
            <dependent idx="5">descending</dependent>
          </dep>
          <dep type="prep_through">
            <governor idx="5">descending</governor>
            <dependent idx="7">fog</dependent>
          </dep>
          <dep type="det">
            <governor idx="10">steep</governor>
            <dependent idx="9">a</dependent>
          </dep>
          <dep type="prep_in">
            <governor idx="5">descending</governor>
            <dependent idx="10">steep</dependent>
          </dep>
          <dep type="appos">
            <governor idx="10">steep</governor>
            <dependent idx="12">nose</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="5">descending</governor>
            <dependent idx="13">down</dependent>
          </dep>
          <dep type="amod">
            <governor idx="17">attitude</governor>
            <dependent idx="15">right</dependent>
          </dep>
          <dep type="nn">
            <governor idx="17">attitude</governor>
            <dependent idx="16">bank</dependent>
          </dep>
          <dep type="appos">
            <governor idx="4">airplane</governor>
            <dependent idx="17">attitude</dependent>
          </dep>
          <dep type="prepc_before">
            <governor idx="17">attitude</governor>
            <dependent idx="19">disappearing</dependent>
          </dep>
          <dep type="prep_from">
            <governor idx="19">disappearing</governor>
            <dependent idx="21">view</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-ccprocessed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="2">saw</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="2">saw</governor>
            <dependent idx="1">Witnesses</dependent>
          </dep>
          <dep type="det">
            <governor idx="4">airplane</governor>
            <dependent idx="3">the</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="2">saw</governor>
            <dependent idx="4">airplane</dependent>
          </dep>
          <dep type="partmod">
            <governor idx="4">airplane</governor>
            <dependent idx="5">descending</dependent>
          </dep>
          <dep type="prep_through">
            <governor idx="5">descending</governor>
            <dependent idx="7">fog</dependent>
          </dep>
          <dep type="det">
            <governor idx="10">steep</governor>
            <dependent idx="9">a</dependent>
          </dep>
          <dep type="prep_in">
            <governor idx="5">descending</governor>
            <dependent idx="10">steep</dependent>
          </dep>
          <dep type="appos">
            <governor idx="10">steep</governor>
            <dependent idx="12">nose</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="5">descending</governor>
            <dependent idx="13">down</dependent>
          </dep>
          <dep type="amod">
            <governor idx="17">attitude</governor>
            <dependent idx="15">right</dependent>
          </dep>
          <dep type="nn">
            <governor idx="17">attitude</governor>
            <dependent idx="16">bank</dependent>
          </dep>
          <dep type="appos">
            <governor idx="4">airplane</governor>
            <dependent idx="17">attitude</dependent>
          </dep>
          <dep type="prepc_before">
            <governor idx="17">attitude</governor>
            <dependent idx="19">disappearing</dependent>
          </dep>
          <dep type="prep_from">
            <governor idx="19">disappearing</governor>
            <dependent idx="21">view</dependent>
          </dep>
        </dependencies>
      </sentence>
      <sentence id="9">
        <tokens>
          <token id="1">
            <word>Engine</word>
            <lemma>Engine</lemma>
            <CharacterOffsetBegin>938</CharacterOffsetBegin>
            <CharacterOffsetEnd>944</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>O</NER>
          </token>
          <token id="2">
            <word>rpm</word>
            <lemma>rpm</lemma>
            <CharacterOffsetBegin>945</CharacterOffsetBegin>
            <CharacterOffsetEnd>948</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="3">
            <word>was</word>
            <lemma>be</lemma>
            <CharacterOffsetBegin>949</CharacterOffsetBegin>
            <CharacterOffsetEnd>952</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="4">
            <word>heard</word>
            <lemma>hear</lemma>
            <CharacterOffsetBegin>953</CharacterOffsetBegin>
            <CharacterOffsetEnd>958</CharacterOffsetEnd>
            <POS>VBN</POS>
            <NER>O</NER>
          </token>
          <token id="5">
            <word>to</word>
            <lemma>to</lemma>
            <CharacterOffsetBegin>959</CharacterOffsetBegin>
            <CharacterOffsetEnd>961</CharacterOffsetEnd>
            <POS>TO</POS>
            <NER>O</NER>
          </token>
          <token id="6">
            <word>increase</word>
            <lemma>increase</lemma>
            <CharacterOffsetBegin>962</CharacterOffsetBegin>
            <CharacterOffsetEnd>970</CharacterOffsetEnd>
            <POS>VB</POS>
            <NER>O</NER>
          </token>
          <token id="7">
            <word>,</word>
            <lemma>,</lemma>
            <CharacterOffsetBegin>970</CharacterOffsetBegin>
            <CharacterOffsetEnd>971</CharacterOffsetEnd>
            <POS>,</POS>
            <NER>O</NER>
          </token>
          <token id="8">
            <word>then</word>
            <lemma>then</lemma>
            <CharacterOffsetBegin>972</CharacterOffsetBegin>
            <CharacterOffsetEnd>976</CharacterOffsetEnd>
            <POS>RB</POS>
            <NER>O</NER>
          </token>
          <token id="9">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>977</CharacterOffsetBegin>
            <CharacterOffsetEnd>980</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="10">
            <word>plane</word>
            <lemma>plane</lemma>
            <CharacterOffsetBegin>981</CharacterOffsetBegin>
            <CharacterOffsetEnd>986</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="11">
            <word>crashed</word>
            <lemma>crash</lemma>
            <CharacterOffsetBegin>987</CharacterOffsetBegin>
            <CharacterOffsetEnd>994</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="12">
            <word>in</word>
            <lemma>in</lemma>
            <CharacterOffsetBegin>995</CharacterOffsetBegin>
            <CharacterOffsetEnd>997</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="13">
            <word>a</word>
            <lemma>a</lemma>
            <CharacterOffsetBegin>998</CharacterOffsetBegin>
            <CharacterOffsetEnd>999</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="14">
            <word>boat</word>
            <lemma>boat</lemma>
            <CharacterOffsetBegin>1000</CharacterOffsetBegin>
            <CharacterOffsetEnd>1004</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="15">
            <word>docking</word>
            <lemma>docking</lemma>
            <CharacterOffsetBegin>1005</CharacterOffsetBegin>
            <CharacterOffsetEnd>1012</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="16">
            <word>area</word>
            <lemma>area</lemma>
            <CharacterOffsetBegin>1013</CharacterOffsetBegin>
            <CharacterOffsetEnd>1017</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="17">
            <word>.</word>
            <lemma>.</lemma>
            <CharacterOffsetBegin>1017</CharacterOffsetBegin>
            <CharacterOffsetEnd>1018</CharacterOffsetEnd>
            <POS>.</POS>
            <NER>O</NER>
          </token>
        </tokens>
        <parse>(ROOT (S (S (NP (NNP Engine) (NN rpm)) (VP (VBD was) (VP (VBN heard) (S (VP (TO to) (VP (VB increase))))))) (, ,) (RB then) (S (NP (DT the) (NN plane)) (VP (VBD crashed) (PP (IN in) (NP (DT a) (NN boat) (NN docking) (NN area))))) (. .))) </parse>
        <dependencies type="basic-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="4">heard</dependent>
          </dep>
          <dep type="nn">
            <governor idx="2">rpm</governor>
            <dependent idx="1">Engine</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="4">heard</governor>
            <dependent idx="2">rpm</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="4">heard</governor>
            <dependent idx="3">was</dependent>
          </dep>
          <dep type="aux">
            <governor idx="6">increase</governor>
            <dependent idx="5">to</dependent>
          </dep>
          <dep type="xcomp">
            <governor idx="4">heard</governor>
            <dependent idx="6">increase</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="4">heard</governor>
            <dependent idx="8">then</dependent>
          </dep>
          <dep type="det">
            <governor idx="10">plane</governor>
            <dependent idx="9">the</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="11">crashed</governor>
            <dependent idx="10">plane</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="4">heard</governor>
            <dependent idx="11">crashed</dependent>
          </dep>
          <dep type="prep">
            <governor idx="11">crashed</governor>
            <dependent idx="12">in</dependent>
          </dep>
          <dep type="det">
            <governor idx="16">area</governor>
            <dependent idx="13">a</dependent>
          </dep>
          <dep type="nn">
            <governor idx="16">area</governor>
            <dependent idx="14">boat</dependent>
          </dep>
          <dep type="nn">
            <governor idx="16">area</governor>
            <dependent idx="15">docking</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="12">in</governor>
            <dependent idx="16">area</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="4">heard</dependent>
          </dep>
          <dep type="nn">
            <governor idx="2">rpm</governor>
            <dependent idx="1">Engine</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="4">heard</governor>
            <dependent idx="2">rpm</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="4">heard</governor>
            <dependent idx="3">was</dependent>
          </dep>
          <dep type="aux">
            <governor idx="6">increase</governor>
            <dependent idx="5">to</dependent>
          </dep>
          <dep type="xcomp">
            <governor idx="4">heard</governor>
            <dependent idx="6">increase</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="4">heard</governor>
            <dependent idx="8">then</dependent>
          </dep>
          <dep type="det">
            <governor idx="10">plane</governor>
            <dependent idx="9">the</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="11">crashed</governor>
            <dependent idx="10">plane</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="4">heard</governor>
            <dependent idx="11">crashed</dependent>
          </dep>
          <dep type="det">
            <governor idx="16">area</governor>
            <dependent idx="13">a</dependent>
          </dep>
          <dep type="nn">
            <governor idx="16">area</governor>
            <dependent idx="14">boat</dependent>
          </dep>
          <dep type="nn">
            <governor idx="16">area</governor>
            <dependent idx="15">docking</dependent>
          </dep>
          <dep type="prep_in">
            <governor idx="11">crashed</governor>
            <dependent idx="16">area</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-ccprocessed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="4">heard</dependent>
          </dep>
          <dep type="nn">
            <governor idx="2">rpm</governor>
            <dependent idx="1">Engine</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="4">heard</governor>
            <dependent idx="2">rpm</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="4">heard</governor>
            <dependent idx="3">was</dependent>
          </dep>
          <dep type="aux">
            <governor idx="6">increase</governor>
            <dependent idx="5">to</dependent>
          </dep>
          <dep type="xcomp">
            <governor idx="4">heard</governor>
            <dependent idx="6">increase</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="4">heard</governor>
            <dependent idx="8">then</dependent>
          </dep>
          <dep type="det">
            <governor idx="10">plane</governor>
            <dependent idx="9">the</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="11">crashed</governor>
            <dependent idx="10">plane</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="4">heard</governor>
            <dependent idx="11">crashed</dependent>
          </dep>
          <dep type="det">
            <governor idx="16">area</governor>
            <dependent idx="13">a</dependent>
          </dep>
          <dep type="nn">
            <governor idx="16">area</governor>
            <dependent idx="14">boat</dependent>
          </dep>
          <dep type="nn">
            <governor idx="16">area</governor>
            <dependent idx="15">docking</dependent>
          </dep>
          <dep type="prep_in">
            <governor idx="11">crashed</governor>
            <dependent idx="16">area</dependent>
          </dep>
        </dependencies>
      </sentence>
      <sentence id="10">
        <tokens>
          <token id="1">
            <word>The</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>1020</CharacterOffsetBegin>
            <CharacterOffsetEnd>1023</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="2">
            <word>left</word>
            <lemma>left</lemma>
            <CharacterOffsetBegin>1024</CharacterOffsetBegin>
            <CharacterOffsetEnd>1028</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="3">
            <word>seat</word>
            <lemma>seat</lemma>
            <CharacterOffsetBegin>1029</CharacterOffsetBegin>
            <CharacterOffsetEnd>1033</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="4">
            <word>pilot</word>
            <lemma>pilot</lemma>
            <CharacterOffsetBegin>1034</CharacterOffsetBegin>
            <CharacterOffsetEnd>1039</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="5">
            <word>had</word>
            <lemma>have</lemma>
            <CharacterOffsetBegin>1040</CharacterOffsetBegin>
            <CharacterOffsetEnd>1043</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="6">
            <word>been</word>
            <lemma>be</lemma>
            <CharacterOffsetBegin>1044</CharacterOffsetBegin>
            <CharacterOffsetEnd>1048</CharacterOffsetEnd>
            <POS>VBN</POS>
            <NER>O</NER>
          </token>
          <token id="7">
            <word>issued</word>
            <lemma>issue</lemma>
            <CharacterOffsetBegin>1049</CharacterOffsetBegin>
            <CharacterOffsetEnd>1055</CharacterOffsetEnd>
            <POS>VBN</POS>
            <NER>O</NER>
          </token>
          <token id="8">
            <word>a</word>
            <lemma>a</lemma>
            <CharacterOffsetBegin>1056</CharacterOffsetBegin>
            <CharacterOffsetEnd>1057</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="9">
            <word>private</word>
            <lemma>private</lemma>
            <CharacterOffsetBegin>1058</CharacterOffsetBegin>
            <CharacterOffsetEnd>1065</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="10">
            <word>pilot</word>
            <lemma>pilot</lemma>
            <CharacterOffsetBegin>1066</CharacterOffsetBegin>
            <CharacterOffsetEnd>1071</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="11">
            <word>certificate</word>
            <lemma>certificate</lemma>
            <CharacterOffsetBegin>1072</CharacterOffsetBegin>
            <CharacterOffsetEnd>1083</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="12">
            <word>based</word>
            <lemma>base</lemma>
            <CharacterOffsetBegin>1084</CharacterOffsetBegin>
            <CharacterOffsetEnd>1089</CharacterOffsetEnd>
            <POS>VBN</POS>
            <NER>O</NER>
          </token>
          <token id="13">
            <word>on</word>
            <lemma>on</lemma>
            <CharacterOffsetBegin>1090</CharacterOffsetBegin>
            <CharacterOffsetEnd>1092</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="14">
            <word>his</word>
            <lemma>he</lemma>
            <CharacterOffsetBegin>1093</CharacterOffsetBegin>
            <CharacterOffsetEnd>1096</CharacterOffsetEnd>
            <POS>PRP$</POS>
            <NER>O</NER>
          </token>
          <token id="15">
            <word>German</word>
            <lemma>german</lemma>
            <CharacterOffsetBegin>1097</CharacterOffsetBegin>
            <CharacterOffsetEnd>1103</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>MISC</NER>
          </token>
          <token id="16">
            <word>certificate</word>
            <lemma>certificate</lemma>
            <CharacterOffsetBegin>1104</CharacterOffsetBegin>
            <CharacterOffsetEnd>1115</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="17">
            <word>;</word>
            <lemma>;</lemma>
            <CharacterOffsetBegin>1115</CharacterOffsetBegin>
            <CharacterOffsetEnd>1116</CharacterOffsetEnd>
            <POS>:</POS>
            <NER>O</NER>
          </token>
          <token id="18">
            <word>he</word>
            <lemma>he</lemma>
            <CharacterOffsetBegin>1117</CharacterOffsetBegin>
            <CharacterOffsetEnd>1119</CharacterOffsetEnd>
            <POS>PRP</POS>
            <NER>O</NER>
          </token>
          <token id="19">
            <word>held</word>
            <lemma>hold</lemma>
            <CharacterOffsetBegin>1120</CharacterOffsetBegin>
            <CharacterOffsetEnd>1124</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="20">
            <word>an</word>
            <lemma>a</lemma>
            <CharacterOffsetBegin>1125</CharacterOffsetBegin>
            <CharacterOffsetEnd>1127</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="21">
            <word>instrument</word>
            <lemma>instrument</lemma>
            <CharacterOffsetBegin>1128</CharacterOffsetBegin>
            <CharacterOffsetEnd>1138</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="22">
            <word>rating</word>
            <lemma>rating</lemma>
            <CharacterOffsetBegin>1139</CharacterOffsetBegin>
            <CharacterOffsetEnd>1145</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="23">
            <word>,</word>
            <lemma>,</lemma>
            <CharacterOffsetBegin>1145</CharacterOffsetBegin>
            <CharacterOffsetEnd>1146</CharacterOffsetEnd>
            <POS>,</POS>
            <NER>O</NER>
          </token>
          <token id="24">
            <word>but</word>
            <lemma>but</lemma>
            <CharacterOffsetBegin>1147</CharacterOffsetBegin>
            <CharacterOffsetEnd>1150</CharacterOffsetEnd>
            <POS>CC</POS>
            <NER>O</NER>
          </token>
          <token id="25">
            <word>did</word>
            <lemma>do</lemma>
            <CharacterOffsetBegin>1151</CharacterOffsetBegin>
            <CharacterOffsetEnd>1154</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="26">
            <word>not</word>
            <lemma>not</lemma>
            <CharacterOffsetBegin>1155</CharacterOffsetBegin>
            <CharacterOffsetEnd>1158</CharacterOffsetEnd>
            <POS>RB</POS>
            <NER>O</NER>
          </token>
          <token id="27">
            <word>have</word>
            <lemma>have</lemma>
            <CharacterOffsetBegin>1159</CharacterOffsetBegin>
            <CharacterOffsetEnd>1163</CharacterOffsetEnd>
            <POS>VB</POS>
            <NER>O</NER>
          </token>
          <token id="28">
            <word>a</word>
            <lemma>a</lemma>
            <CharacterOffsetBegin>1164</CharacterOffsetBegin>
            <CharacterOffsetEnd>1165</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="29">
            <word>current</word>
            <lemma>current</lemma>
            <CharacterOffsetBegin>1166</CharacterOffsetBegin>
            <CharacterOffsetEnd>1173</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>DATE</NER>
            <NormalizedNER>PRESENT_REF</NormalizedNER>
            <Timex tid="t1" type="DATE">PRESENT_REF</Timex>
          </token>
          <token id="30">
            <word>medical</word>
            <lemma>medical</lemma>
            <CharacterOffsetBegin>1174</CharacterOffsetBegin>
            <CharacterOffsetEnd>1181</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="31">
            <word>certificate</word>
            <lemma>certificate</lemma>
            <CharacterOffsetBegin>1182</CharacterOffsetBegin>
            <CharacterOffsetEnd>1193</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="32">
            <word>;</word>
            <lemma>;</lemma>
            <CharacterOffsetBegin>1193</CharacterOffsetBegin>
            <CharacterOffsetEnd>1194</CharacterOffsetEnd>
            <POS>:</POS>
            <NER>O</NER>
          </token>
          <token id="33">
            <word>no</word>
            <lemma>no</lemma>
            <CharacterOffsetBegin>1195</CharacterOffsetBegin>
            <CharacterOffsetEnd>1197</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="34">
            <word>record</word>
            <lemma>record</lemma>
            <CharacterOffsetBegin>1198</CharacterOffsetBegin>
            <CharacterOffsetEnd>1204</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="35">
            <word>was</word>
            <lemma>be</lemma>
            <CharacterOffsetBegin>1205</CharacterOffsetBegin>
            <CharacterOffsetEnd>1208</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="36">
            <word>found</word>
            <lemma>find</lemma>
            <CharacterOffsetBegin>1209</CharacterOffsetBegin>
            <CharacterOffsetEnd>1214</CharacterOffsetEnd>
            <POS>VBN</POS>
            <NER>O</NER>
          </token>
          <token id="37">
            <word>of</word>
            <lemma>of</lemma>
            <CharacterOffsetBegin>1215</CharacterOffsetBegin>
            <CharacterOffsetEnd>1217</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="38">
            <word>his</word>
            <lemma>he</lemma>
            <CharacterOffsetBegin>1218</CharacterOffsetBegin>
            <CharacterOffsetEnd>1221</CharacterOffsetEnd>
            <POS>PRP$</POS>
            <NER>O</NER>
          </token>
          <token id="39">
            <word>previous</word>
            <lemma>previous</lemma>
            <CharacterOffsetBegin>1222</CharacterOffsetBegin>
            <CharacterOffsetEnd>1230</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="40">
            <word>flight</word>
            <lemma>flight</lemma>
            <CharacterOffsetBegin>1231</CharacterOffsetBegin>
            <CharacterOffsetEnd>1237</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="41">
            <word>time</word>
            <lemma>time</lemma>
            <CharacterOffsetBegin>1238</CharacterOffsetBegin>
            <CharacterOffsetEnd>1242</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="42">
            <word>.</word>
            <lemma>.</lemma>
            <CharacterOffsetBegin>1242</CharacterOffsetBegin>
            <CharacterOffsetEnd>1243</CharacterOffsetEnd>
            <POS>.</POS>
            <NER>O</NER>
          </token>
        </tokens>
        <parse>(ROOT (S (S (NP (DT The) (JJ left) (NN seat) (NN pilot)) (VP (VBD had) (VP (VBN been) (VP (VBN issued) (NP (DT a) (JJ private) (NN pilot) (NN certificate)) (PP (VBN based) (PP (IN on) (NP (PRP$ his) (JJ German) (NN certificate)))))))) (: ;) (S (NP (PRP he)) (VP (VP (VBD held) (NP (DT an) (NN instrument) (NN rating))) (, ,) (CC but) (VP (VBD did) (RB not) (VP (VB have) (NP (DT a) (JJ current) (JJ medical) (NN certificate)))))) (: ;) (S (NP (DT no) (NN record)) (VP (VBD was) (VP (VBN found) (PP (IN of) (NP (PRP$ his) (JJ previous) (NN flight) (NN time)))))) (. .))) </parse>
        <dependencies type="basic-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="7">issued</dependent>
          </dep>
          <dep type="det">
            <governor idx="4">pilot</governor>
            <dependent idx="1">The</dependent>
          </dep>
          <dep type="amod">
            <governor idx="4">pilot</governor>
            <dependent idx="2">left</dependent>
          </dep>
          <dep type="nn">
            <governor idx="4">pilot</governor>
            <dependent idx="3">seat</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="7">issued</governor>
            <dependent idx="4">pilot</dependent>
          </dep>
          <dep type="aux">
            <governor idx="7">issued</governor>
            <dependent idx="5">had</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="7">issued</governor>
            <dependent idx="6">been</dependent>
          </dep>
          <dep type="det">
            <governor idx="11">certificate</governor>
            <dependent idx="8">a</dependent>
          </dep>
          <dep type="amod">
            <governor idx="11">certificate</governor>
            <dependent idx="9">private</dependent>
          </dep>
          <dep type="nn">
            <governor idx="11">certificate</governor>
            <dependent idx="10">pilot</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="7">issued</governor>
            <dependent idx="11">certificate</dependent>
          </dep>
          <dep type="prep">
            <governor idx="7">issued</governor>
            <dependent idx="12">based</dependent>
          </dep>
          <dep type="pcomp">
            <governor idx="12">based</governor>
            <dependent idx="13">on</dependent>
          </dep>
          <dep type="poss">
            <governor idx="16">certificate</governor>
            <dependent idx="14">his</dependent>
          </dep>
          <dep type="amod">
            <governor idx="16">certificate</governor>
            <dependent idx="15">German</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="13">on</governor>
            <dependent idx="16">certificate</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="19">held</governor>
            <dependent idx="18">he</dependent>
          </dep>
          <dep type="parataxis">
            <governor idx="7">issued</governor>
            <dependent idx="19">held</dependent>
          </dep>
          <dep type="det">
            <governor idx="22">rating</governor>
            <dependent idx="20">an</dependent>
          </dep>
          <dep type="nn">
            <governor idx="22">rating</governor>
            <dependent idx="21">instrument</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="19">held</governor>
            <dependent idx="22">rating</dependent>
          </dep>
          <dep type="cc">
            <governor idx="19">held</governor>
            <dependent idx="24">but</dependent>
          </dep>
          <dep type="aux">
            <governor idx="27">have</governor>
            <dependent idx="25">did</dependent>
          </dep>
          <dep type="neg">
            <governor idx="27">have</governor>
            <dependent idx="26">not</dependent>
          </dep>
          <dep type="conj">
            <governor idx="19">held</governor>
            <dependent idx="27">have</dependent>
          </dep>
          <dep type="det">
            <governor idx="31">certificate</governor>
            <dependent idx="28">a</dependent>
          </dep>
          <dep type="amod">
            <governor idx="31">certificate</governor>
            <dependent idx="29">current</dependent>
          </dep>
          <dep type="amod">
            <governor idx="31">certificate</governor>
            <dependent idx="30">medical</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="27">have</governor>
            <dependent idx="31">certificate</dependent>
          </dep>
          <dep type="det">
            <governor idx="34">record</governor>
            <dependent idx="33">no</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="36">found</governor>
            <dependent idx="34">record</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="36">found</governor>
            <dependent idx="35">was</dependent>
          </dep>
          <dep type="parataxis">
            <governor idx="7">issued</governor>
            <dependent idx="36">found</dependent>
          </dep>
          <dep type="prep">
            <governor idx="36">found</governor>
            <dependent idx="37">of</dependent>
          </dep>
          <dep type="poss">
            <governor idx="41">time</governor>
            <dependent idx="38">his</dependent>
          </dep>
          <dep type="amod">
            <governor idx="41">time</governor>
            <dependent idx="39">previous</dependent>
          </dep>
          <dep type="nn">
            <governor idx="41">time</governor>
            <dependent idx="40">flight</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="37">of</governor>
            <dependent idx="41">time</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="7">issued</dependent>
          </dep>
          <dep type="det">
            <governor idx="4">pilot</governor>
            <dependent idx="1">The</dependent>
          </dep>
          <dep type="amod">
            <governor idx="4">pilot</governor>
            <dependent idx="2">left</dependent>
          </dep>
          <dep type="nn">
            <governor idx="4">pilot</governor>
            <dependent idx="3">seat</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="7">issued</governor>
            <dependent idx="4">pilot</dependent>
          </dep>
          <dep type="aux">
            <governor idx="7">issued</governor>
            <dependent idx="5">had</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="7">issued</governor>
            <dependent idx="6">been</dependent>
          </dep>
          <dep type="det">
            <governor idx="11">certificate</governor>
            <dependent idx="8">a</dependent>
          </dep>
          <dep type="amod">
            <governor idx="11">certificate</governor>
            <dependent idx="9">private</dependent>
          </dep>
          <dep type="nn">
            <governor idx="11">certificate</governor>
            <dependent idx="10">pilot</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="7">issued</governor>
            <dependent idx="11">certificate</dependent>
          </dep>
          <dep type="prepc_based_on">
            <governor idx="7">issued</governor>
            <dependent idx="13">on</dependent>
          </dep>
          <dep type="poss">
            <governor idx="16">certificate</governor>
            <dependent idx="14">his</dependent>
          </dep>
          <dep type="amod">
            <governor idx="16">certificate</governor>
            <dependent idx="15">German</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="7">issued</governor>
            <dependent idx="16">certificate</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="19">held</governor>
            <dependent idx="18">he</dependent>
          </dep>
          <dep type="parataxis">
            <governor idx="7">issued</governor>
            <dependent idx="19">held</dependent>
          </dep>
          <dep type="det">
            <governor idx="22">rating</governor>
            <dependent idx="20">an</dependent>
          </dep>
          <dep type="nn">
            <governor idx="22">rating</governor>
            <dependent idx="21">instrument</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="19">held</governor>
            <dependent idx="22">rating</dependent>
          </dep>
          <dep type="aux">
            <governor idx="27">have</governor>
            <dependent idx="25">did</dependent>
          </dep>
          <dep type="neg">
            <governor idx="27">have</governor>
            <dependent idx="26">not</dependent>
          </dep>
          <dep type="conj_but">
            <governor idx="19">held</governor>
            <dependent idx="27">have</dependent>
          </dep>
          <dep type="det">
            <governor idx="31">certificate</governor>
            <dependent idx="28">a</dependent>
          </dep>
          <dep type="amod">
            <governor idx="31">certificate</governor>
            <dependent idx="29">current</dependent>
          </dep>
          <dep type="amod">
            <governor idx="31">certificate</governor>
            <dependent idx="30">medical</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="27">have</governor>
            <dependent idx="31">certificate</dependent>
          </dep>
          <dep type="det">
            <governor idx="34">record</governor>
            <dependent idx="33">no</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="36">found</governor>
            <dependent idx="34">record</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="36">found</governor>
            <dependent idx="35">was</dependent>
          </dep>
          <dep type="parataxis">
            <governor idx="7">issued</governor>
            <dependent idx="36">found</dependent>
          </dep>
          <dep type="poss">
            <governor idx="41">time</governor>
            <dependent idx="38">his</dependent>
          </dep>
          <dep type="amod">
            <governor idx="41">time</governor>
            <dependent idx="39">previous</dependent>
          </dep>
          <dep type="nn">
            <governor idx="41">time</governor>
            <dependent idx="40">flight</dependent>
          </dep>
          <dep type="prep_of">
            <governor idx="36">found</governor>
            <dependent idx="41">time</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-ccprocessed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="7">issued</dependent>
          </dep>
          <dep type="det">
            <governor idx="4">pilot</governor>
            <dependent idx="1">The</dependent>
          </dep>
          <dep type="amod">
            <governor idx="4">pilot</governor>
            <dependent idx="2">left</dependent>
          </dep>
          <dep type="nn">
            <governor idx="4">pilot</governor>
            <dependent idx="3">seat</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="7">issued</governor>
            <dependent idx="4">pilot</dependent>
          </dep>
          <dep type="aux">
            <governor idx="7">issued</governor>
            <dependent idx="5">had</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="7">issued</governor>
            <dependent idx="6">been</dependent>
          </dep>
          <dep type="det">
            <governor idx="11">certificate</governor>
            <dependent idx="8">a</dependent>
          </dep>
          <dep type="amod">
            <governor idx="11">certificate</governor>
            <dependent idx="9">private</dependent>
          </dep>
          <dep type="nn">
            <governor idx="11">certificate</governor>
            <dependent idx="10">pilot</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="7">issued</governor>
            <dependent idx="11">certificate</dependent>
          </dep>
          <dep type="prepc_based_on">
            <governor idx="7">issued</governor>
            <dependent idx="13">on</dependent>
          </dep>
          <dep type="poss">
            <governor idx="16">certificate</governor>
            <dependent idx="14">his</dependent>
          </dep>
          <dep type="amod">
            <governor idx="16">certificate</governor>
            <dependent idx="15">German</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="7">issued</governor>
            <dependent idx="16">certificate</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="19">held</governor>
            <dependent idx="18">he</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="27">have</governor>
            <dependent idx="18">he</dependent>
          </dep>
          <dep type="parataxis">
            <governor idx="7">issued</governor>
            <dependent idx="19">held</dependent>
          </dep>
          <dep type="det">
            <governor idx="22">rating</governor>
            <dependent idx="20">an</dependent>
          </dep>
          <dep type="nn">
            <governor idx="22">rating</governor>
            <dependent idx="21">instrument</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="19">held</governor>
            <dependent idx="22">rating</dependent>
          </dep>
          <dep type="aux">
            <governor idx="27">have</governor>
            <dependent idx="25">did</dependent>
          </dep>
          <dep type="neg">
            <governor idx="27">have</governor>
            <dependent idx="26">not</dependent>
          </dep>
          <dep type="parataxis">
            <governor idx="7">issued</governor>
            <dependent idx="27">have</dependent>
          </dep>
          <dep type="conj_but">
            <governor idx="19">held</governor>
            <dependent idx="27">have</dependent>
          </dep>
          <dep type="det">
            <governor idx="31">certificate</governor>
            <dependent idx="28">a</dependent>
          </dep>
          <dep type="amod">
            <governor idx="31">certificate</governor>
            <dependent idx="29">current</dependent>
          </dep>
          <dep type="amod">
            <governor idx="31">certificate</governor>
            <dependent idx="30">medical</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="27">have</governor>
            <dependent idx="31">certificate</dependent>
          </dep>
          <dep type="det">
            <governor idx="34">record</governor>
            <dependent idx="33">no</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="36">found</governor>
            <dependent idx="34">record</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="36">found</governor>
            <dependent idx="35">was</dependent>
          </dep>
          <dep type="parataxis">
            <governor idx="7">issued</governor>
            <dependent idx="36">found</dependent>
          </dep>
          <dep type="poss">
            <governor idx="41">time</governor>
            <dependent idx="38">his</dependent>
          </dep>
          <dep type="amod">
            <governor idx="41">time</governor>
            <dependent idx="39">previous</dependent>
          </dep>
          <dep type="nn">
            <governor idx="41">time</governor>
            <dependent idx="40">flight</dependent>
          </dep>
          <dep type="prep_of">
            <governor idx="36">found</governor>
            <dependent idx="41">time</dependent>
          </dep>
        </dependencies>
      </sentence>
      <sentence id="11">
        <tokens>
          <token id="1">
            <word>The</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>1245</CharacterOffsetBegin>
            <CharacterOffsetEnd>1248</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="2">
            <word>right</word>
            <lemma>right</lemma>
            <CharacterOffsetBegin>1249</CharacterOffsetBegin>
            <CharacterOffsetEnd>1254</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="3">
            <word>seat</word>
            <lemma>seat</lemma>
            <CharacterOffsetBegin>1255</CharacterOffsetBegin>
            <CharacterOffsetEnd>1259</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="4">
            <word>pilot</word>
            <lemma>pilot</lemma>
            <CharacterOffsetBegin>1260</CharacterOffsetBegin>
            <CharacterOffsetEnd>1265</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="5">
            <word>held</word>
            <lemma>hold</lemma>
            <CharacterOffsetBegin>1266</CharacterOffsetBegin>
            <CharacterOffsetEnd>1270</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="6">
            <word>a</word>
            <lemma>a</lemma>
            <CharacterOffsetBegin>1271</CharacterOffsetBegin>
            <CharacterOffsetEnd>1272</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="7">
            <word>commercial</word>
            <lemma>commercial</lemma>
            <CharacterOffsetBegin>1273</CharacterOffsetBegin>
            <CharacterOffsetEnd>1283</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="8">
            <word>&amp;</word>
            <lemma>&amp;</lemma>
            <CharacterOffsetBegin>1284</CharacterOffsetBegin>
            <CharacterOffsetEnd>1285</CharacterOffsetEnd>
            <POS>CC</POS>
            <NER>O</NER>
          </token>
          <token id="9">
            <word>ATP</word>
            <lemma>atp</lemma>
            <CharacterOffsetBegin>1286</CharacterOffsetBegin>
            <CharacterOffsetEnd>1289</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="10">
            <word>certificate</word>
            <lemma>certificate</lemma>
            <CharacterOffsetBegin>1290</CharacterOffsetBegin>
            <CharacterOffsetEnd>1301</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="11">
            <word>-LRB-</word>
            <lemma>-lrb-</lemma>
            <CharacterOffsetBegin>1302</CharacterOffsetBegin>
            <CharacterOffsetEnd>1303</CharacterOffsetEnd>
            <POS>-LRB-</POS>
            <NER>O</NER>
          </token>
          <token id="12">
            <word>with</word>
            <lemma>with</lemma>
            <CharacterOffsetBegin>1303</CharacterOffsetBegin>
            <CharacterOffsetEnd>1307</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="13">
            <word>instrument</word>
            <lemma>instrument</lemma>
            <CharacterOffsetBegin>1308</CharacterOffsetBegin>
            <CharacterOffsetEnd>1318</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="14">
            <word>rating</word>
            <lemma>rating</lemma>
            <CharacterOffsetBegin>1319</CharacterOffsetBegin>
            <CharacterOffsetEnd>1325</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="15">
            <word>-RRB-</word>
            <lemma>-rrb-</lemma>
            <CharacterOffsetBegin>1325</CharacterOffsetBegin>
            <CharacterOffsetEnd>1326</CharacterOffsetEnd>
            <POS>-RRB-</POS>
            <NER>O</NER>
          </token>
          <token id="16">
            <word>&amp;</word>
            <lemma>&amp;</lemma>
            <CharacterOffsetBegin>1327</CharacterOffsetBegin>
            <CharacterOffsetEnd>1328</CharacterOffsetEnd>
            <POS>CC</POS>
            <NER>O</NER>
          </token>
          <token id="17">
            <word>was</word>
            <lemma>be</lemma>
            <CharacterOffsetBegin>1329</CharacterOffsetBegin>
            <CharacterOffsetEnd>1332</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="18">
            <word>presumed</word>
            <lemma>presume</lemma>
            <CharacterOffsetBegin>1333</CharacterOffsetBegin>
            <CharacterOffsetEnd>1341</CharacterOffsetEnd>
            <POS>VBN</POS>
            <NER>O</NER>
          </token>
          <token id="19">
            <word>to</word>
            <lemma>to</lemma>
            <CharacterOffsetBegin>1342</CharacterOffsetBegin>
            <CharacterOffsetEnd>1344</CharacterOffsetEnd>
            <POS>TO</POS>
            <NER>O</NER>
          </token>
          <token id="20">
            <word>be</word>
            <lemma>be</lemma>
            <CharacterOffsetBegin>1345</CharacterOffsetBegin>
            <CharacterOffsetEnd>1347</CharacterOffsetEnd>
            <POS>VB</POS>
            <NER>O</NER>
          </token>
          <token id="21">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>1348</CharacterOffsetBegin>
            <CharacterOffsetEnd>1351</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="22">
            <word>PIC</word>
            <lemma>PIC</lemma>
            <CharacterOffsetBegin>1352</CharacterOffsetBegin>
            <CharacterOffsetEnd>1355</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>O</NER>
          </token>
          <token id="23">
            <word>;</word>
            <lemma>;</lemma>
            <CharacterOffsetBegin>1355</CharacterOffsetBegin>
            <CharacterOffsetEnd>1356</CharacterOffsetEnd>
            <POS>:</POS>
            <NER>O</NER>
          </token>
          <token id="24">
            <word>a</word>
            <lemma>a</lemma>
            <CharacterOffsetBegin>1357</CharacterOffsetBegin>
            <CharacterOffsetEnd>1358</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="25">
            <word>review</word>
            <lemma>review</lemma>
            <CharacterOffsetBegin>1359</CharacterOffsetBegin>
            <CharacterOffsetEnd>1365</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="26">
            <word>of</word>
            <lemma>of</lemma>
            <CharacterOffsetBegin>1366</CharacterOffsetBegin>
            <CharacterOffsetEnd>1368</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="27">
            <word>his</word>
            <lemma>he</lemma>
            <CharacterOffsetBegin>1369</CharacterOffsetBegin>
            <CharacterOffsetEnd>1372</CharacterOffsetEnd>
            <POS>PRP$</POS>
            <NER>O</NER>
          </token>
          <token id="28">
            <word>log</word>
            <lemma>log</lemma>
            <CharacterOffsetBegin>1373</CharacterOffsetBegin>
            <CharacterOffsetEnd>1376</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="29">
            <word>book</word>
            <lemma>book</lemma>
            <CharacterOffsetBegin>1377</CharacterOffsetBegin>
            <CharacterOffsetEnd>1381</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="30">
            <word>revealed</word>
            <lemma>reveal</lemma>
            <CharacterOffsetBegin>1382</CharacterOffsetBegin>
            <CharacterOffsetEnd>1390</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="31">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>1391</CharacterOffsetBegin>
            <CharacterOffsetEnd>1394</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="32">
            <word>last</word>
            <lemma>last</lemma>
            <CharacterOffsetBegin>1395</CharacterOffsetBegin>
            <CharacterOffsetEnd>1399</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="33">
            <word>recorded</word>
            <lemma>recorded</lemma>
            <CharacterOffsetBegin>1400</CharacterOffsetBegin>
            <CharacterOffsetEnd>1408</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
          </token>
          <token id="34">
            <word>flight</word>
            <lemma>flight</lemma>
            <CharacterOffsetBegin>1409</CharacterOffsetBegin>
            <CharacterOffsetEnd>1415</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="35">
            <word>in</word>
            <lemma>in</lemma>
            <CharacterOffsetBegin>1416</CharacterOffsetBegin>
            <CharacterOffsetEnd>1418</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="36">
            <word>this</word>
            <lemma>this</lemma>
            <CharacterOffsetBegin>1419</CharacterOffsetBegin>
            <CharacterOffsetEnd>1423</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
          </token>
          <token id="37">
            <word>make</word>
            <lemma>make</lemma>
            <CharacterOffsetBegin>1424</CharacterOffsetBegin>
            <CharacterOffsetEnd>1428</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="38">
            <word>&amp;</word>
            <lemma>&amp;</lemma>
            <CharacterOffsetBegin>1429</CharacterOffsetBegin>
            <CharacterOffsetEnd>1430</CharacterOffsetEnd>
            <POS>CC</POS>
            <NER>O</NER>
          </token>
          <token id="39">
            <word>model</word>
            <lemma>model</lemma>
            <CharacterOffsetBegin>1431</CharacterOffsetBegin>
            <CharacterOffsetEnd>1436</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="40">
            <word>of</word>
            <lemma>of</lemma>
            <CharacterOffsetBegin>1437</CharacterOffsetBegin>
            <CharacterOffsetEnd>1439</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="41">
            <word>airplane</word>
            <lemma>airplane</lemma>
            <CharacterOffsetBegin>1440</CharacterOffsetBegin>
            <CharacterOffsetEnd>1448</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
          </token>
          <token id="42">
            <word>was</word>
            <lemma>be</lemma>
            <CharacterOffsetBegin>1449</CharacterOffsetBegin>
            <CharacterOffsetEnd>1452</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
          </token>
          <token id="43">
            <word>on</word>
            <lemma>on</lemma>
            <CharacterOffsetBegin>1453</CharacterOffsetBegin>
            <CharacterOffsetEnd>1455</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
          </token>
          <token id="44">
            <word>3\/27\/88</word>
            <lemma>3\/27\/88</lemma>
            <CharacterOffsetBegin>1456</CharacterOffsetBegin>
            <CharacterOffsetEnd>1463</CharacterOffsetEnd>
            <POS>CD</POS>
            <NER>DATE</NER>
            <NormalizedNER>XX88-03-27</NormalizedNER>
            <Timex tid="t1" type="DATE">XX88-03-27</Timex>
          </token>
          <token id="45">
            <word>.</word>
            <lemma>.</lemma>
            <CharacterOffsetBegin>1463</CharacterOffsetBegin>
            <CharacterOffsetEnd>1464</CharacterOffsetEnd>
            <POS>.</POS>
            <NER>O</NER>
          </token>
        </tokens>
        <parse>(ROOT (S (S (NP (DT The) (JJ right) (NN seat) (NN pilot)) (VP (VBD held) (NP (NP (DT a) (JJ commercial)) (CC &amp;) (NP (NP (NN ATP) (NN certificate)) (PRN (-LRB- -LRB-) (PP (IN with) (NP (NN instrument) (NN rating))) (-RRB- -RRB-)) (SBAR (S (NP (CC &amp;)) (VP (VBD was) (VP (VBN presumed) (S (VP (TO to) (VP (VB be) (NP (DT the) (NNP PIC))))))))))))) (: ;) (S (NP (NP (DT a) (NN review)) (PP (IN of) (NP (PRP$ his) (NN log) (NN book)))) (VP (VBD revealed) (SBAR (S (NP (NP (DT the) (JJ last) (JJ recorded) (NN flight)) (PP (IN in) (NP (NP (DT this) (NN make)) (CC &amp;) (NP (NP (NN model)) (PP (IN of) (NP (NN airplane))))))) (VP (VBD was) (PP (IN on) (NP (CD 3\/27\/88)))))))) (. .))) </parse>
        <dependencies type="basic-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="5">held</dependent>
          </dep>
          <dep type="det">
            <governor idx="4">pilot</governor>
            <dependent idx="1">The</dependent>
          </dep>
          <dep type="amod">
            <governor idx="4">pilot</governor>
            <dependent idx="2">right</dependent>
          </dep>
          <dep type="nn">
            <governor idx="4">pilot</governor>
            <dependent idx="3">seat</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="5">held</governor>
            <dependent idx="4">pilot</dependent>
          </dep>
          <dep type="det">
            <governor idx="7">commercial</governor>
            <dependent idx="6">a</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="5">held</governor>
            <dependent idx="7">commercial</dependent>
          </dep>
          <dep type="cc">
            <governor idx="7">commercial</governor>
            <dependent idx="8">&amp;</dependent>
          </dep>
          <dep type="nn">
            <governor idx="10">certificate</governor>
            <dependent idx="9">ATP</dependent>
          </dep>
          <dep type="conj">
            <governor idx="7">commercial</governor>
            <dependent idx="10">certificate</dependent>
          </dep>
          <dep type="prep">
            <governor idx="10">certificate</governor>
            <dependent idx="12">with</dependent>
          </dep>
          <dep type="nn">
            <governor idx="14">rating</governor>
            <dependent idx="13">instrument</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="12">with</governor>
            <dependent idx="14">rating</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="18">presumed</governor>
            <dependent idx="16">&amp;</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="18">presumed</governor>
            <dependent idx="17">was</dependent>
          </dep>
          <dep type="rcmod">
            <governor idx="10">certificate</governor>
            <dependent idx="18">presumed</dependent>
          </dep>
          <dep type="aux">
            <governor idx="22">PIC</governor>
            <dependent idx="19">to</dependent>
          </dep>
          <dep type="cop">
            <governor idx="22">PIC</governor>
            <dependent idx="20">be</dependent>
          </dep>
          <dep type="det">
            <governor idx="22">PIC</governor>
            <dependent idx="21">the</dependent>
          </dep>
          <dep type="xcomp">
            <governor idx="18">presumed</governor>
            <dependent idx="22">PIC</dependent>
          </dep>
          <dep type="det">
            <governor idx="25">review</governor>
            <dependent idx="24">a</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="30">revealed</governor>
            <dependent idx="25">review</dependent>
          </dep>
          <dep type="prep">
            <governor idx="25">review</governor>
            <dependent idx="26">of</dependent>
          </dep>
          <dep type="poss">
            <governor idx="29">book</governor>
            <dependent idx="27">his</dependent>
          </dep>
          <dep type="nn">
            <governor idx="29">book</governor>
            <dependent idx="28">log</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="26">of</governor>
            <dependent idx="29">book</dependent>
          </dep>
          <dep type="parataxis">
            <governor idx="5">held</governor>
            <dependent idx="30">revealed</dependent>
          </dep>
          <dep type="det">
            <governor idx="34">flight</governor>
            <dependent idx="31">the</dependent>
          </dep>
          <dep type="amod">
            <governor idx="34">flight</governor>
            <dependent idx="32">last</dependent>
          </dep>
          <dep type="amod">
            <governor idx="34">flight</governor>
            <dependent idx="33">recorded</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="42">was</governor>
            <dependent idx="34">flight</dependent>
          </dep>
          <dep type="prep">
            <governor idx="34">flight</governor>
            <dependent idx="35">in</dependent>
          </dep>
          <dep type="det">
            <governor idx="37">make</governor>
            <dependent idx="36">this</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="35">in</governor>
            <dependent idx="37">make</dependent>
          </dep>
          <dep type="cc">
            <governor idx="37">make</governor>
            <dependent idx="38">&amp;</dependent>
          </dep>
          <dep type="conj">
            <governor idx="37">make</governor>
            <dependent idx="39">model</dependent>
          </dep>
          <dep type="prep">
            <governor idx="39">model</governor>
            <dependent idx="40">of</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="40">of</governor>
            <dependent idx="41">airplane</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="30">revealed</governor>
            <dependent idx="42">was</dependent>
          </dep>
          <dep type="prep">
            <governor idx="42">was</governor>
            <dependent idx="43">on</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="43">on</governor>
            <dependent idx="44">3\/27\/88</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="5">held</dependent>
          </dep>
          <dep type="det">
            <governor idx="4">pilot</governor>
            <dependent idx="1">The</dependent>
          </dep>
          <dep type="amod">
            <governor idx="4">pilot</governor>
            <dependent idx="2">right</dependent>
          </dep>
          <dep type="nn">
            <governor idx="4">pilot</governor>
            <dependent idx="3">seat</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="5">held</governor>
            <dependent idx="4">pilot</dependent>
          </dep>
          <dep type="det">
            <governor idx="7">commercial</governor>
            <dependent idx="6">a</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="5">held</governor>
            <dependent idx="7">commercial</dependent>
          </dep>
          <dep type="nn">
            <governor idx="10">certificate</governor>
            <dependent idx="9">ATP</dependent>
          </dep>
          <dep type="conj_and">
            <governor idx="7">commercial</governor>
            <dependent idx="10">certificate</dependent>
          </dep>
          <dep type="nn">
            <governor idx="14">rating</governor>
            <dependent idx="13">instrument</dependent>
          </dep>
          <dep type="prep_with">
            <governor idx="10">certificate</governor>
            <dependent idx="14">rating</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="18">presumed</governor>
            <dependent idx="16">&amp;</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="18">presumed</governor>
            <dependent idx="17">was</dependent>
          </dep>
          <dep type="rcmod">
            <governor idx="10">certificate</governor>
            <dependent idx="18">presumed</dependent>
          </dep>
          <dep type="aux">
            <governor idx="22">PIC</governor>
            <dependent idx="19">to</dependent>
          </dep>
          <dep type="cop">
            <governor idx="22">PIC</governor>
            <dependent idx="20">be</dependent>
          </dep>
          <dep type="det">
            <governor idx="22">PIC</governor>
            <dependent idx="21">the</dependent>
          </dep>
          <dep type="xcomp">
            <governor idx="18">presumed</governor>
            <dependent idx="22">PIC</dependent>
          </dep>
          <dep type="det">
            <governor idx="25">review</governor>
            <dependent idx="24">a</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="30">revealed</governor>
            <dependent idx="25">review</dependent>
          </dep>
          <dep type="poss">
            <governor idx="29">book</governor>
            <dependent idx="27">his</dependent>
          </dep>
          <dep type="nn">
            <governor idx="29">book</governor>
            <dependent idx="28">log</dependent>
          </dep>
          <dep type="prep_of">
            <governor idx="25">review</governor>
            <dependent idx="29">book</dependent>
          </dep>
          <dep type="parataxis">
            <governor idx="5">held</governor>
            <dependent idx="30">revealed</dependent>
          </dep>
          <dep type="det">
            <governor idx="34">flight</governor>
            <dependent idx="31">the</dependent>
          </dep>
          <dep type="amod">
            <governor idx="34">flight</governor>
            <dependent idx="32">last</dependent>
          </dep>
          <dep type="amod">
            <governor idx="34">flight</governor>
            <dependent idx="33">recorded</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="42">was</governor>
            <dependent idx="34">flight</dependent>
          </dep>
          <dep type="det">
            <governor idx="37">make</governor>
            <dependent idx="36">this</dependent>
          </dep>
          <dep type="prep_in">
            <governor idx="34">flight</governor>
            <dependent idx="37">make</dependent>
          </dep>
          <dep type="conj_and">
            <governor idx="37">make</governor>
            <dependent idx="39">model</dependent>
          </dep>
          <dep type="prep_of">
            <governor idx="39">model</governor>
            <dependent idx="41">airplane</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="30">revealed</governor>
            <dependent idx="42">was</dependent>
          </dep>
          <dep type="prep_on">
            <governor idx="42">was</governor>
            <dependent idx="44">3\/27\/88</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-ccprocessed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="5">held</dependent>
          </dep>
          <dep type="det">
            <governor idx="4">pilot</governor>
            <dependent idx="1">The</dependent>
          </dep>
          <dep type="amod">
            <governor idx="4">pilot</governor>
            <dependent idx="2">right</dependent>
          </dep>
          <dep type="nn">
            <governor idx="4">pilot</governor>
            <dependent idx="3">seat</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="5">held</governor>
            <dependent idx="4">pilot</dependent>
          </dep>
          <dep type="det">
            <governor idx="7">commercial</governor>
            <dependent idx="6">a</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="5">held</governor>
            <dependent idx="7">commercial</dependent>
          </dep>
          <dep type="nn">
            <governor idx="10">certificate</governor>
            <dependent idx="9">ATP</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="5">held</governor>
            <dependent idx="10">certificate</dependent>
          </dep>
          <dep type="conj_and">
            <governor idx="7">commercial</governor>
            <dependent idx="10">certificate</dependent>
          </dep>
          <dep type="nn">
            <governor idx="14">rating</governor>
            <dependent idx="13">instrument</dependent>
          </dep>
          <dep type="prep_with">
            <governor idx="10">certificate</governor>
            <dependent idx="14">rating</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="18">presumed</governor>
            <dependent idx="16">&amp;</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="18">presumed</governor>
            <dependent idx="17">was</dependent>
          </dep>
          <dep type="rcmod">
            <governor idx="10">certificate</governor>
            <dependent idx="18">presumed</dependent>
          </dep>
          <dep type="aux">
            <governor idx="22">PIC</governor>
            <dependent idx="19">to</dependent>
          </dep>
          <dep type="cop">
            <governor idx="22">PIC</governor>
            <dependent idx="20">be</dependent>
          </dep>
          <dep type="det">
            <governor idx="22">PIC</governor>
            <dependent idx="21">the</dependent>
          </dep>
          <dep type="xcomp">
            <governor idx="18">presumed</governor>
            <dependent idx="22">PIC</dependent>
          </dep>
          <dep type="det">
            <governor idx="25">review</governor>
            <dependent idx="24">a</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="30">revealed</governor>
            <dependent idx="25">review</dependent>
          </dep>
          <dep type="poss">
            <governor idx="29">book</governor>
            <dependent idx="27">his</dependent>
          </dep>
          <dep type="nn">
            <governor idx="29">book</governor>
            <dependent idx="28">log</dependent>
          </dep>
          <dep type="prep_of">
            <governor idx="25">review</governor>
            <dependent idx="29">book</dependent>
          </dep>
          <dep type="parataxis">
            <governor idx="5">held</governor>
            <dependent idx="30">revealed</dependent>
          </dep>
          <dep type="det">
            <governor idx="34">flight</governor>
            <dependent idx="31">the</dependent>
          </dep>
          <dep type="amod">
            <governor idx="34">flight</governor>
            <dependent idx="32">last</dependent>
          </dep>
          <dep type="amod">
            <governor idx="34">flight</governor>
            <dependent idx="33">recorded</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="42">was</governor>
            <dependent idx="34">flight</dependent>
          </dep>
          <dep type="det">
            <governor idx="37">make</governor>
            <dependent idx="36">this</dependent>
          </dep>
          <dep type="prep_in">
            <governor idx="34">flight</governor>
            <dependent idx="37">make</dependent>
          </dep>
          <dep type="prep_in">
            <governor idx="34">flight</governor>
            <dependent idx="39">model</dependent>
          </dep>
          <dep type="conj_and">
            <governor idx="37">make</governor>
            <dependent idx="39">model</dependent>
          </dep>
          <dep type="prep_of">
            <governor idx="39">model</governor>
            <dependent idx="41">airplane</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="30">revealed</governor>
            <dependent idx="42">was</dependent>
          </dep>
          <dep type="prep_on">
            <governor idx="42">was</governor>
            <dependent idx="44">3\/27\/88</dependent>
          </dep>
        </dependencies>
      </sentence>
    </sentences>
    <coreference>
      <coreference>
        <mention representative="true">
          <sentence>1</sentence>
          <start>5</start>
          <end>7</end>
          <head>6</head>
        </mention>
        <mention>
          <sentence>5</sentence>
          <start>13</start>
          <end>15</end>
          <head>14</head>
        </mention>
      </coreference>
      <coreference>
        <mention representative="true">
          <sentence>1</sentence>
          <start>10</start>
          <end>15</end>
          <head>14</head>
        </mention>
        <mention>
          <sentence>2</sentence>
          <start>11</start>
          <end>13</end>
          <head>12</head>
        </mention>
        <mention>
          <sentence>3</sentence>
          <start>9</start>
          <end>11</end>
          <head>10</head>
        </mention>
      </coreference>
      <coreference>
        <mention representative="true">
          <sentence>2</sentence>
          <start>1</start>
          <end>6</end>
          <head>2</head>
        </mention>
        <mention>
          <sentence>4</sentence>
          <start>24</start>
          <end>26</end>
          <head>25</head>
        </mention>
      </coreference>
      <coreference>
        <mention representative="true">
          <sentence>3</sentence>
          <start>4</start>
          <end>6</end>
          <head>5</head>
        </mention>
        <mention>
          <sentence>4</sentence>
          <start>9</start>
          <end>11</end>
          <head>10</head>
        </mention>
        <mention>
          <sentence>7</sentence>
          <start>4</start>
          <end>6</end>
          <head>5</head>
        </mention>
        <mention>
          <sentence>7</sentence>
          <start>16</start>
          <end>17</end>
          <head>16</head>
        </mention>
      </coreference>
      <coreference>
        <mention representative="true">
          <sentence>4</sentence>
          <start>1</start>
          <end>3</end>
          <head>2</head>
        </mention>
        <mention>
          <sentence>7</sentence>
          <start>1</start>
          <end>3</end>
          <head>2</head>
        </mention>
      </coreference>
      <coreference>
        <mention representative="true">
          <sentence>5</sentence>
          <start>5</start>
          <end>7</end>
          <head>6</head>
        </mention>
        <mention>
          <sentence>5</sentence>
          <start>19</start>
          <end>20</end>
          <head>19</head>
        </mention>
        <mention>
          <sentence>11</sentence>
          <start>16</start>
          <end>17</end>
          <head>16</head>
        </mention>
      </coreference>
      <coreference>
        <mention representative="true">
          <sentence>5</sentence>
          <start>5</start>
          <end>15</end>
          <head>5</head>
        </mention>
        <mention>
          <sentence>5</sentence>
          <start>7</start>
          <end>12</end>
          <head>8</head>
        </mention>
      </coreference>
      <coreference>
        <mention representative="true">
          <sentence>8</sentence>
          <start>3</start>
          <end>22</end>
          <head>4</head>
        </mention>
        <mention>
          <sentence>8</sentence>
          <start>3</start>
          <end>14</end>
          <head>4</head>
        </mention>
      </coreference>
      <coreference>
        <mention representative="true">
          <sentence>8</sentence>
          <start>9</start>
          <end>13</end>
          <head>10</head>
        </mention>
        <mention>
          <sentence>8</sentence>
          <start>9</start>
          <end>11</end>
          <head>10</head>
        </mention>
      </coreference>
      <coreference>
        <mention representative="true">
          <sentence>10</sentence>
          <start>20</start>
          <end>23</end>
          <head>22</head>
        </mention>
        <mention>
          <sentence>11</sentence>
          <start>13</start>
          <end>15</end>
          <head>14</head>
        </mention>
      </coreference>
      <coreference>
        <mention representative="true">
          <sentence>10</sentence>
          <start>1</start>
          <end>5</end>
          <head>4</head>
        </mention>
        <mention>
          <sentence>10</sentence>
          <start>14</start>
          <end>15</end>
          <head>14</head>
        </mention>
        <mention>
          <sentence>10</sentence>
          <start>18</start>
          <end>19</end>
          <head>18</head>
        </mention>
        <mention>
          <sentence>10</sentence>
          <start>38</start>
          <end>39</end>
          <head>38</head>
        </mention>
      </coreference>
      <coreference>
        <mention representative="true">
          <sentence>11</sentence>
          <start>1</start>
          <end>5</end>
          <head>4</head>
        </mention>
        <mention>
          <sentence>11</sentence>
          <start>27</start>
          <end>28</end>
          <head>27</head>
        </mention>
      </coreference>
      <coreference>
        <mention representative="true">
          <sentence>11</sentence>
          <start>36</start>
          <end>42</end>
          <head>37</head>
        </mention>
        <mention>
          <sentence>11</sentence>
          <start>36</start>
          <end>38</end>
          <head>37</head>
        </mention>
      </coreference>
    </coreference>
  </document>
</root>"""


def _iminus_one(i):
    return int(i) - 1
