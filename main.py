import csv
import json
import logging
import re

import sklearn
from allennlp import pretrained
from allennlp.modules.elmo import Elmo, batch_to_ids

from ner import NER
from utils import Utils

logging.basicConfig(level=logging.INFO)


class Node:
    def __init__(self, parent, mapping, sentence):
        self.parent = parent
        self.nodeType = str(mapping['nodeType'])
        self.word = str(mapping['word'])
        self.attributes = mapping['attributes']
        self.link = mapping['link']
        self.children = []
        self.is_fuzzy_matched = False
        self.fuzzy_word = ''
        self.role = ''
        if parent is None:
            self.index = 10000
        else:
            self.index = parent.index + parent.word.index(self.word) * 10
        if 'children' in mapping:
            raw_children = mapping['children']
            for raw_child in raw_children:
                child = Node(self, raw_child, sentence)
                self.children.append(child)
        self.is_a_leaf = len(self.children) == 0

    def is_it_a_leaf(self):
        return self.is_a_leaf

    def get_children(self):
        return self.children

    def get_node_type(self):
        return self.nodeType

    def get_word(self):
        return self.word

    def get_parent(self):
        return self.parent

    def is_a_stop_word(self):
        return self.word.lower().strip() in Utils.stop_words

    def is_a_noun(self):
        if self.is_a_stop_word():
            return False
        if self.nodeType.startswith("N"):
            return True
        return False

    def is_a_verb(self):
        if self.is_a_stop_word():
            return False
        if self.nodeType.startswith("VB") and ' ' not in self.word:
            return True
        return False

    def is_unknown(self):
        return self.role == ''

    def is_a_wh(self):
        if self.nodeType.startswith("WH") and self.word.strip() != 'that':
            return True
        return False

    def is_a_pp(self):
        if self.nodeType.startswith("PP"):
            return True
        return False

    def is_a_in(self):
        if self.nodeType.startswith("IN"):
            return True
        return False

    def is_a_adj(self):
        if self.nodeType.startswith("JJ"):
            return True
        return False

    def get_leaves(self):
        leaves = []
        if self.is_a_leaf:
            leaves.append(self)
        else:
            for child in self.children:
                c_leaves = child.get_leaves()
                if c_leaves is not None:
                    leaves.extend(c_leaves)
        return leaves

    def is_your_child(self, node):
        if node.word in self.word:
            if self.__eq__(node):
                return True
            else:
                for child in self.children:
                    res = child.is_your_child(node)
                    if res:
                        return True
        return False

    def get_siblings(self):
        siblings = []
        if self.parent is None:
            return siblings
        for child in self.parent.children:
            if not child == self:
                siblings.append(child)
        return siblings

    def analyze(self):
        logging.debug('analyzing node called {}'.format(self.__repr__()))
        cond = False
        if self.is_a_noun():
            logging.debug('noun phrase has been found {}'.format(self.word))
        elif self.is_a_verb():
            logging.debug('verb phrase has been found {}'.format(self.word))
        if cond:
            return True
        for child in self.children:
            child.analyze()
        return True

    def find_fuzzy(self, string):
        nodes = []
        if self.word.strip().lower().startswith(string.strip().lower()):
            self.is_fuzzy_matched = True
            self.fuzzy_word = string
            nodes.append(self)
        for child in self.children:
            res = child.find_fuzzy(string)
            if res is not None and len(res) > 0:
                nodes.extend(res)
        return nodes

    def find(self, string):
        nodes = []
        if self.word.strip().lower() == string.strip().lower():
            nodes.append(self)
        for child in self.children:
            res = child.find(string)
            if res is not None:
                nodes.extend(res)
        return nodes

    def iterate(self):
        logging.debug('word: {} and type: {}'.format(self.word, self.nodeType))
        for child in self.children:
            child.iterate()

    def iterate_nouns(self):
        nouns = []
        if self.is_a_noun():
            logging.debug('noun is: {}'.format(self.word))
            nouns.append(self)
        for child in self.children:
            nouns.extend(child.iterate_nouns())
        return nouns

    def iterate_verbs(self):
        verbs = []
        if self.is_a_verb():
            logging.debug('verb is: {}'.format(self.word))
            verbs.append(self)
        for child in self.children:
            verbs.extend(child.iterate_verbs())
        return verbs

    def iterate_adjectives(self):
        adjs = []
        if self.is_a_adj():
            logging.debug('adjective: {}'.format(self.word))
            adjs.append(self)
        for child in self.children:
            adjs.extend(child.iterate_adjectives())
        return adjs

    def iterate_pps(self):
        pps = []
        if self.is_a_pp():
            logging.debug('pp is: {}'.format(self.word))
            pps.append(self)
        for child in self.children:
            pps.extend(child.iterate_pps())
        return pps

    def get_in_in_pp(self):
        if self.is_a_in():
            logging.debug('IN found: {}'.format(self.word))
            return self
        for child in self.children:
            return child.get_in_in_pp()

    def iterate_wh(self):
        whs = []
        if self.is_a_wh():
            logging.debug('wh found, {}'.format(self.word))

            whs.append(self)
            return whs
        else:
            for child in self.children:
                whs.extend(child.iterate_wh())
        return whs

    def __repr__(self):
        representation = ""
        representation += str(self.nodeType)
        if len(self.children) > 0:
            representation += "("
            for child in self.children:
                representation += child.__repr__()
                representation += ", "
            representation = representation[:-2]
            representation += ")"

        return representation

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Node):
            return self.word == other.word and self.index == other.index
        return False

    def __hash__(self):
        return hash(self.index) + hash(self.word)


# parse tree is a class that capture root node and enable us to iterate over the nodes...


class ParseTree:
    def __init__(self, root, sentence):
        self.root = Node(None, root, sentence)

    def get_root(self):
        return self.root

    def get_type(self):
        return self.root.get_node_type()

    def analyze(self):
        self.root.analyze()

    def get_intent(self):
        return self.root.iterate_wh()

    def find(self, string):
        nodes = self.root.find(string)
        if nodes is None or len(nodes) == 0:
            fuzzy_result = self.root.find_fuzzy(string)
            return fuzzy_result
        return nodes

    def get_leaves(self):
        return self.root.get_leaves()

    def iterate(self):
        logging.info('iterating and printing roles')
        self.root.iterate()
        logging.info('iterating nouns: ')
        self.root.iterate_nouns()
        logging.info('iterating verbs: ')
        self.root.iterate_verbs()

    def get_nouns(self):
        return self.root.iterate_nouns()

    def get_verbs(self):
        return self.root.iterate_verbs()

    def get_pps(self):
        return self.root.iterate_pps()

    def get_adjectives(self):
        return self.root.iterate_adjectives()


def is_inside(string, list):
    for l in list:
        if string in l or l in string:
            return True
    return False


def is_left_inside(string, list):
    for l in list:
        if string.lower().strip() == l.lower().strip():
            return True
    return False


class GeoAnalyticalQuestion:
    def __init__(self, id, source, year, title, question,
                 q_type):  # change the source and add taxonomy! also put it result
        self.id = id
        self.source = source
        self.q_type = q_type
        self.year = year
        self.question = question.replace('?', '').replace('\'s', '')
        self.title = title
        self.parser = ''

    def analyze(self):
        s = Sentence(self.question, self.source, self.q_type)
        self.parser = s.analyze()
        return self.parser


class Sentence():
    def __init__(self, sentence, source, s_type):
        self.sentence = sentence
        self.source = source
        self.s_type = s_type

    @staticmethod
    def is_ambiguous(intent_list, intent_code):
        if len(intent_list) == 1 and ('o' not in intent_code or 't' not in intent_code):
            return True
        return False

    @staticmethod
    def resolving_intent(desc_list_info):
        flag = False
        val = ''
        resolved_intent_list = []
        resolved_intent_code = ''
        for elem in desc_list_info:
            if elem['tag'] in ['o', 't']:
                if not flag:
                    val = elem['tag']
                    flag = True
                if elem['tag'] == val:
                    resolved_intent_code += val
                    resolved_intent_list.append({'tag': elem['tag'], 'value': elem['value']})

            elif elem['tag'] in ['q', 'p']:
                if flag:
                    break
                else:
                    resolved_intent_code += val
                    resolved_intent_list.append({'tag': elem['tag'], 'value': elem['value']})

            elif resolved_intent_code != '':
                break;
        result = {'list': resolved_intent_list, 'code': resolved_intent_code}
        return result

    def analyze(self):
        logging.info('*******************************************************')
        result_dict = {}
        result_dict['source'] = self.source.strip().lower()
        result_dict['q_type'] = self.s_type.strip().lower()
        res = model.predict(sentence=self.sentence)
        root_dict = res['hierplane_tree']['root']
        logging.info('sentence {} parsed as {}'.format(self.sentence, root_dict))

        emb = elmo(batch_to_ids([self.sentence.split()]))['elmo_representations'][0].detach().numpy()

        parse_tree = ParseTree(root_dict, self.sentence)
        # logging.info('ParseTree type is: {}'.format(parse_tree.get_type()))
        # parse_tree.iterate()
        logging.info('Now it\'s time to check the string representation \n{}'.format(str(parse_tree.root)))
        # parse_tree.analyze()
        logging.info('extracting information')
        all_nodes = set()
        all_intent_nodes = set()
        all_desc_nodes = set()
        toponyms = NER.extract_place_names(self.sentence)
        result_dict['pnames'] = toponyms
        topo_nodes = set()
        for t in toponyms:
            logging.info('\ttoponym:\t{}'.format(t))
            nodes = parse_tree.find(t)
            if nodes is None:
                logging.info('An error in finding nodes')
            else:
                for n in nodes:
                    n.role = 'n'
                    topo_nodes.add(n)
        for t_node in topo_nodes:
            logging.info('\t**Found Node: {} and index {}'.format(t_node.word, t_node.index))
        all_nodes = all_nodes.union(topo_nodes)
        all_desc_nodes = all_desc_nodes.union(topo_nodes)

        dates = NER.extract_dates(self.sentence)
        result_dict['dates'] = dates
        dates_nodes = set()
        for d in dates:
            logging.info('\tdate:\t{}'.format(d))
            nodes = parse_tree.find(d)
            if nodes is None:
                logging.info('An error in finding nodes')
            else:
                for n in nodes:
                    n.role = 'd'
                    dates_nodes.add(n)

        for d_node in dates_nodes:
            logging.info('\t**Found Node: {} and index {}'.format(d_node.word, d_node.index))
        all_nodes = all_nodes.union(dates_nodes)
        all_desc_nodes = all_desc_nodes.union(dates_nodes)

        whs_nodes = parse_tree.get_intent()
        whs = []
        for wh_node in whs_nodes:
            wh_node.role = intent_encoding(wh_node, PRONOUN)
            whs.append(wh_node.word)
        for w in whs:
            logging.info('intent is: {}'.format(w))
        all_nodes = all_nodes.union(whs_nodes)
        all_intent_nodes = all_intent_nodes.union(whs_nodes)
        result_dict['intents'] = whs
        a_entities_set = set()
        a_entities_nodes = set()
        a_types = []
        a_types_nodes = set()
        for whs_node in whs_nodes:
            wh_nouns = whs_node.iterate_nouns()
            wh_nouns.sort(key=sort_function, reverse=True)
            for n in wh_nouns:
                if not is_inside(n.word, toponyms) and not is_inside(n.word, dates) and not is_left_inside(
                        n.word, a_types) and is_a_new_one(a_types_nodes, n):
                    if is_left_inside(n.word.lower().strip(), pt_set) or is_left_inside(n.word.lower().strip(),
                                                                                        pt_dict.keys()):
                        a_types.append(n.word)
                        n.role = 't'
                        a_types_nodes.add(n)
                    elif ' ' not in n.word.strip() and len(n.word) > 2:
                        a_entities_set.add(n.word)
                        n.role = 'o'
                        a_entities_nodes.add(n)
        for t in a_types:
            logging.info('\ttype in intent:\t{}'.format(t))
        a_entities = list(a_entities_set)
        for e in a_entities:
            logging.info('\tentity in intent:\t{}'.format(e))
        all_nodes = all_nodes.union(a_types_nodes)
        all_intent_nodes = all_intent_nodes.union(a_types_nodes)
        all_nodes = all_nodes.union(a_entities_nodes)
        all_intent_nodes = all_intent_nodes.union(a_entities_nodes)
        result_dict['i_objects'] = a_entities
        result_dict['i_ptypes'] = a_types
        nouns = parse_tree.get_nouns()
        nouns.sort(key=sort_function, reverse=True)
        types = []
        types_nodes = set()
        entities_set = set()
        entities_nodes = set()
        for n in nouns:
            if not is_inside(n.word, toponyms) and not is_inside(n.word, dates) and not is_inside(
                    n.word, whs) and not is_left_inside(n.word, types) and is_a_new_one(types_nodes, n):
                if is_left_inside(n.word.lower().strip(), pt_set) or is_left_inside(n.word.lower().strip(),
                                                                                    pt_dict.keys()):
                    types.append(n.word)
                    n.role = 't'
                    types_nodes.add(n)
                elif ' ' not in n.word.strip() and len(n.word) > 2:
                    entities_set.add(n.word)
                    n.role = 'o'
                    entities_nodes.add(n)
        for t in types:
            logging.info('\ttype:\t{}'.format(t))
        entities = list(entities_set)
        for e in entities:
            logging.info('\tentity:\t{}'.format(e))
        all_nodes = all_nodes.union(types_nodes)
        all_desc_nodes = all_desc_nodes.union(types_nodes)
        all_nodes = all_nodes.union(entities_nodes)
        all_desc_nodes = all_desc_nodes.union(entities_nodes)
        result_dict['objects'] = entities
        result_dict['ptypes'] = types
        verbs = parse_tree.get_verbs()
        situations = []
        situations_nodes = set()
        activities = []
        activities_nodes = set()
        unknowns = []
        unknowns_nodes = set()
        for v in verbs:
            v_index = self.sentence.split().index(v.word)
            v_emb = [emb[0][v_index]]
            logging.debug('verb is {} and len of emb is {}'.format(v.word, len(v_emb)))
            decision = verb_encoding(v_emb, actv_emb, stav_emb)
            if decision == "a":
                activities.append(v.word)
                v.role = 'a'
                activities_nodes.add(v)
            elif decision == "s":
                situations.append(v.word)
                v.role = 's'
                situations_nodes.add(v)
            else:
                unknowns.append(v.word)
                unknowns_nodes.add(v)
        for s in situations:
            logging.info('\tsituation: {}'.format(s))
        for a in activities:
            logging.info('\tactivities: {}'.format(a))
        for u in unknowns:
            logging.info('\tunknown: {}'.format(u))
        all_nodes = all_nodes.union(activities_nodes)
        all_desc_nodes = all_desc_nodes.union(activities_nodes)
        all_nodes = all_nodes.union(situations_nodes)
        all_desc_nodes = all_desc_nodes.union(situations_nodes)
        result_dict['situations'] = situations
        result_dict['activities'] = activities
        pps = parse_tree.get_pps()
        relations = []
        relation_nodes = set()
        for pp in pps:
            for n in toponyms:
                if 'with' in pp.word.lower():
                    is_within = is_within_phrase(pp.word)
                    if is_within is not None:
                        in_pp = pp.get_in_in_pp()
                        if in_pp is not None:
                            relations.append(in_pp.word)
                            in_pp.role = 'r'
                            relation_nodes.add(in_pp)
                if n in pp.word and not is_inside_right(pp.word, entities) and not is_inside_right(pp.word, a_entities):
                    in_pp = pp.get_in_in_pp()
                    if in_pp is not None:
                        relations.append(in_pp.word)
                        in_pp.role = 'r'
                        relation_nodes.add(in_pp)
                        break
            for t in types:
                if t in pp.word:
                    in_pp = pp.get_in_in_pp()
                    if in_pp is not None:
                        relations.append(in_pp.word)
                        in_pp.role = 'r'
                        relation_nodes.add(in_pp)
                        break
        all_nodes = all_nodes.union(relation_nodes)
        all_desc_nodes = all_desc_nodes.union(relation_nodes)
        for relation in relations:
            logging.info('\trelation: {}'.format(relation))
        result_dict['relations'] = relations

        adjs = parse_tree.get_adjectives()
        qualities = []
        qualities_nodes = set()
        object_qualities = []
        object_qualities_nodes = set()
        for adj in adjs:
            siblings = adj.get_siblings()
            for sibling in siblings:
                if is_inside(sibling.word, toponyms) or is_inside(sibling.word, types) or is_inside(sibling.word,
                                                                                                    a_types):
                    if not is_inside(adj.word, types) and not is_inside(adj.word, a_types):
                        qualities.append(adj.word)
                        adj.role = 'q'
                        qualities_nodes.add(adj)
                        break
                elif is_inside(sibling.word, entities) or is_inside(sibling.word, a_entities):
                    object_qualities.append(adj.word)
                    adj.role = 'p'
                    object_qualities_nodes.add(adj)
                    break
        all_nodes = all_nodes.union(qualities_nodes)
        all_desc_nodes = all_desc_nodes.union(qualities_nodes)
        all_nodes = all_nodes.union(object_qualities_nodes)
        all_desc_nodes = all_desc_nodes.union(object_qualities_nodes)
        for q in qualities:
            logging.info('\tquality: {}'.format(q))
        for oq in object_qualities:
            logging.info('\tobject quality: {}'.format(oq))
        result_dict['pqualities'] = qualities
        result_dict['oqualities'] = object_qualities
        # coding schema: where: 1, what: 2, which: 3, why: 4, how: 5, how+adj: 6 etc. make it complete... other:0...
        # ...activity: a, situation: s, quality: q, object_quality: p, relation: r, toponym: n, type: t, date: d
        ignored_nodes = []
        leaves = parse_tree.get_leaves()
        for leaf in leaves:
            if leaf.is_unknown():
                ignored_nodes.append(leaf)

        temp = []

        for leaf in ignored_nodes:
            for n in all_nodes:
                flag = True
                if n.is_fuzzy_matched:
                    if leaf.word in n.word:
                        flag = False
                        break
                else:
                    if n.is_your_child(leaf):
                        flag = False
                        break
            if flag:
                temp.append(leaf)
                all_nodes.add(leaf)
        # ignored_nodes = temp

        all_list = list(all_nodes)
        intent_list = list(all_intent_nodes)
        description_list = list(all_desc_nodes)
        all_list.sort(key=lambda x: x.index, reverse=False)
        intent_list.sort(key=lambda x: x.index, reverse=False)
        description_list.sort(key=lambda x: x.index, reverse=False)
        intent_code = ''
        intent_info = []
        for node in intent_list:
            intent_code += node.role
            if node.is_fuzzy_matched:
                intent_info.append({'tag': node.role, 'value': node.fuzzy_word})
            else:
                intent_info.append({'tag': node.role, 'value': node.word})

        desc_code = ''
        desc_info = []
        for node in description_list:
            desc_code += node.role
            if node.is_fuzzy_matched:
                desc_info.append({'tag': node.role, 'value': node.fuzzy_word})
            else:
                desc_info.append({'tag': node.role, 'value': node.word})

        if Sentence.is_ambiguous(intent_list, intent_code):
            logging.info('the intention is ambiguous, code: {}'.format(intent_code))
            resolved = Sentence.resolving_intent(desc_info)
            result_dict['resolved_intent'] = resolved
            if resolved['code'] != '':
                intent_code += resolved['code']
                intent_info.extend(resolved['list'])
                desc_temp_list = []
                for d in desc_info:
                    if d not in resolved['list']:
                        desc_temp_list.append(d)
                    else:
                        logging.debug('found!')
                desc_code = desc_code.replace(resolved['code'], '', 1)
                desc_info = desc_temp_list
                logging.debug('updated...')

        result_dict['intent_code'] = intent_code
        result_dict['intent_info'] = intent_info
        result_dict['desc_code'] = desc_code
        result_dict['desc_info'] = desc_info
        all_code = ''
        all_info = []
        for node in all_list:
            all_code += node.role
            if node.is_fuzzy_matched:
                all_info.append({'tag': node.role, 'value': node.fuzzy_word})
            else:
                all_info.append({'tag': node.role, 'value': node.word})
        result_dict['all_code'] = all_code
        result_dict['all_info'] = all_info
        logging.info('\tintent code is: {}'.format(intent_code))
        logging.info('\tdesc code is: {}'.format(desc_code))
        logging.info('\tall code is: {}'.format(all_code))
        logging.info('*******************************************************')
        return result_dict


def is_a_new_one(list, node):
    for l in list:
        if l.is_your_child(node):
            return False
    return True


def sort_function(node):
    return len(node.word)


# load place type
def load_pt(fpt):
    pt_set = set()
    pt_dict = dict()
    fpt = open(fpt, 'r', encoding="utf8")
    for line in fpt.readlines():
        pt_set.add(line.strip())
        pt_dict[line.strip()] = 1
    fpt.close()
    return pt_set, pt_dict


def is_inside_right(string, list):
    for l in list:
        if l in string:
            return True
    return False


def is_within_phrase(string):
    m_object = re.match(r'^[w|W]ithi?n?\s\d+[\.]?\d*\s?[\w|\W]+\Z', string + '\n')
    if m_object:
        return m_object.group()
    return None


# load word
def load_word(fword):
    words = set()
    fword = open(fword, 'r', encoding="utf8")
    for line in fword.readlines():
        word = line.strip()
        words.add(word)
    fword.close()
    return words


def list_node_to_list_word(list_node):
    list_word = []
    for n in list_node:
        list_word.append(n.word)
    return list_word


def verb_encoding(verb_emb, activity_embs, situation_embs):
    stav_similar = sklearn.metrics.pairwise.cosine_similarity(situation_embs.squeeze(), verb_emb).max()
    actv_similar = sklearn.metrics.pairwise.cosine_similarity(activity_embs.squeeze(), verb_emb).max()
    if actv_similar > max(stav_similar, 0.35):
        return "a"
    elif stav_similar > max(actv_similar, 0.35):
        return "s"
    return "u"


def intent_encoding(intent_node, pronoun_dict):
    for key, val in pronoun_dict.items():
        if key.lower().strip() in intent_node.word.lower().strip():
            return val
    if 'how' in intent_node.word.lower().strip():
        if 'JJ' in str(intent_node):
            return '6'
        else:
            return '5'
    return '0'


def read_file(fname):
    res = []
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')

        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                # print(f'id: \t{row[0]}, source: {row[1]}, year: {row[3]}, title: {row[4]} and question: {row[5]}.')
                geoaq = GeoAnalyticalQuestion(row[0], row[7], row[3], row[4], row[5], row[6])
                res.append(geoaq)
                line_count += 1
        print(f'Processed {line_count} lines.')
        return res


def read_file_201_dataset(fname):
    res = []
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                # print(f'id: \t{row[0]}, source: {row[1]}, year: {row[3]}, title: {row[4]} and question: {row[5]}.')
                geoaq = GeoAnalyticalQuestion(line_count, '2019', '201 GeoQA Gold Standard Question',
                                              'The other dataset', row[0])
                res.append(geoaq)
                line_count += 1
        print(f'Processed {line_count} lines.')
        return res


PRONOUN = dict(
    {'where': '1', 'what': '2', 'which': '3', 'when': '4', 'who': '6', 'why': '7'})
model = pretrained.span_based_constituency_parsing_with_elmo_joshi_2018()

fpt = 'data/place_type/type-set'
factv = 'data/verb/action_verb.txt'
fstav = 'data/verb/stative_verb.txt'

pt_set, pt_dict = load_pt(fpt)
actv = load_word(factv)
stav = load_word(fstav)

# loading ELMo pretrained word embedding model
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)

# Verb Elmo representation
actv_emb = elmo(batch_to_ids([[v] for v in actv]))['elmo_representations'][0].detach().numpy()
stav_emb = elmo(batch_to_ids([[v] for v in stav]))['elmo_representations'][0].detach().numpy()

###############SENTENCE LEVEL ANALYSIS###############
# sentence = "How many poll workers do each precinct need given the number of households that fall within each precinct"
# sentence = "how many people live in the neighborhood in Amsterdam that has the highest crime rate in Amsterdam between 1990 and 2000"
# sentence = "Where are the houses for sale and built between 1990 and 2000 in Utrecht?"
#####################################################

# geoaqs = read_file_201_dataset('data/geoaq/201-dataset.csv')
# geoaqs = read_file('data/geoaq/Question corpus_UU.csv') # old version of the dataset...
# geoaqs = read_file('data/geoaq/Question corpus_20190918.csv')
geoaqs = read_file('data/geoaq/Question corpus_Qtype_Stypes_20190918.csv')  # has source and type
questions_incorrect = []
results = []
count = 0
for geoaq in geoaqs:
    try:
        result = {}
        result['question'] = geoaq.question
        result['id'] = geoaq.id
        result['source'] = geoaq.source
        result['title'] = geoaq.title
        parser = geoaq.analyze()
        result.update(parser)
        results.append(result)
        # count += 1
        # if count == 20:
        #     break
    except:
        logging.error('this is an error in analyzing geoaq {}'.format(geoaq.question))
        count += 1
        questions_incorrect.append(geoaq.question)
logging.info('number of errors: {}'.format(count))
with open('data/geoaq/analyzed_question_new_parser_source_and_type.json', 'w') as outfile:
    json.dump(results, outfile)
with open('data/geoaq/analyzed_question_new_parser_issues.json', 'w') as outfile:
    json.dump(questions_incorrect, outfile)
logging.debug('finally finished :D')
