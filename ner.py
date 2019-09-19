import logging

from allennlp import pretrained

logging.basicConfig(level=logging.INFO)
model = pretrained.fine_grained_named_entity_recognition_with_elmo_peters_2018()
up_name_tags = ['U-GPE', 'U-LOC', 'U-FAC', 'U-ORG']
cp_name_tags = ['B-GPE', 'B-LOC', 'B-FAC', 'B-ORG', 'I-GPE', 'I-LOC', 'I-FAC', 'I-ORG', 'L-GPE', 'L-LOC', 'L-FAC',
                'L-ORG']

u_date_tags = ['U-DATE']
cp_date_tags = ['B-DATE', 'I-DATE', 'L-DATE']


class NER:
    @staticmethod
    def parse(sentence):
        res = model.predict(sentence=sentence)
        return res

    @staticmethod
    def extract_entities(sentence, u_list, cp_list):
        entities = []
        parsed = NER.parse(sentence)
        current = ''
        for i in range(0, len(parsed['tags'])):
            logging.debug('i: {} word: {} and tag: {}'.format(i, parsed['words'][i], parsed['tags'][i]))
            if parsed['tags'][i] in u_list:
                entities.append(parsed['words'][i])
            elif parsed['tags'][i] in cp_list:
                if parsed['tags'][i].startswith('B-'):
                    current = parsed['words'][i] + ' '
                elif parsed['tags'][i].startswith('L-'):
                    current += parsed['words'][i]
                    entities.append(current)
                else:
                    current += parsed['words'][i] + ' '
        return entities

    @staticmethod
    def extract_place_names(sentence):
        return NER.extract_entities(sentence, up_name_tags, cp_name_tags)

    @staticmethod
    def extract_dates(sentence):
        return NER.extract_entities(sentence, u_date_tags, cp_date_tags)

# sentence = "How much is Amsterdam covered with green?"
# sentence = "I was in National Park of Australia and I saw her near the Danube River in United States of America"
# sentence = "Where are the houses for sale and built between 1990 and 2000 in Utrecht?"
# logging.info(NER.parse(sentence))
# p_names = NER.extract_place_names(sentence)
# for p_name in p_names:
#     logging.info('extracted toponym: {}'.format(p_name))
# d_entities = NER.extract_dates(sentence)
# for d_entity in d_entities:
#     logging.info('extracted date: {}'.format(d_entity))
