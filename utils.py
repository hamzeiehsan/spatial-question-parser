import logging
import os
import re

import geocoder

os.environ["GOOGLE_API_KEY"] = "api_key_from_google_cloud_platform"
logging.basicConfig(level=logging.INFO)


class Utils:
    stop_words = ['i', 'am', 'we', 'are', 'he', 'she', 'is', 'they', 'was', 'where', 'do', 'does', 'did', 'done', 'has',
                  'have', 'had', 'be', 'been']

    @staticmethod
    def regex_checker(string, reg):
        prog = re.compile(reg)
        return prog.match(string)

    @staticmethod
    def noun_resolver(string):
        geonames = geocoder.geonames(string, maxRows=5, key='e_hamzei', fuzzy=1)
        logging.info(geonames)
        print([(r.address, r.country, r.latlng) for r in geonames])
        if len(geonames) > 0:
            return True
        return False

# name = 'Melbourne'
# if Utils.noun_resolver(name):
#     logging.info('{} is a place!'.format(name))
# name = 'Asdkq24'
# if Utils.noun_resolver(name):
#     logging.info('{} is a place!'.format(name))
# name = 'green'
# if Utils.noun_resolver(name):
#     logging.info('{} is a place!'.format(name))
