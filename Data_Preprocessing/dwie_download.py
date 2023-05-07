# downloads and check the hashes
import argparse
import hashlib
import json
import os
from collections import OrderedDict
from time import sleep

import requests

import re

import spacy
from spacy.tokens.token import Token


class TokenizerCPN:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def tokenize(self, text):
        all_tokens_data = []
        doc = self.nlp(text)
        tok_id = 0

        for tok in doc:
            tokp: Token = tok

            found_year = re.findall('([1-3][0-9]{3})', tokp.text)
            found_s_end_date = re.findall('([0-9]{1,4}s)', tokp.text)

            found_dot_after_month = re.findall('(?i)((january|'
                                               'february|'
                                               'march|'
                                               'april|may|june|july|august|september|october|'
                                               'november|december)\\.)', tokp.text)

            # this particular case has to be hardcoded (ex: "2012/2013"), if not just taken as single token
            # something that produces inconsistencies later in dataset (one token pointing to two concepts).
            if len(found_year) == 2 and len(str(tokp)) == 9 and '/' in str(tokp) and str(tokp).index('/') == 4:
                # print('processing dates separated by slash case: ', str(tokp))
                tok1 = tokp.text[0:4]
                tok2 = tokp.text[4:5]
                tok3 = tokp.text[5:9]
                all_tokens_data.append({'tokid_begin_char': tokp.idx, 'tokid_end_char': tokp.idx + 4,
                                        'tok_id': tok_id, 'tok_text': tok1})
                tok_id += 1
                all_tokens_data.append({'tokid_begin_char': tokp.idx + 4, 'tokid_end_char': tokp.idx + 5,
                                        'tok_id': tok_id, 'tok_text': tok2})
                tok_id += 1
                all_tokens_data.append({'tokid_begin_char': tokp.idx + 5, 'tokid_end_char': tokp.idx + 9,
                                        'tok_id': tok_id, 'tok_text': tok3})
                tok_id += 1
            # this particular case happens when date (year for example) is finished in 's', ex: 1980s, 90s, etc.
            # it has to be parsed as two different tokens (ex: ['1980', 's']), with spacy version it gets parsed
            # as single token which produces difference in scores with Johannes version later on
            elif len(found_s_end_date) == 1 and len(found_s_end_date[0]) == len(str(tokp)):
                # print('processing date ending in s: ', str(tokp))
                tok1 = str(tokp)[0:len(str(tokp)) - 1]
                tok2 = str(tokp)[len(str(tokp)) - 1: len(str(tokp))]
                all_tokens_data.append(
                    {'tokid_begin_char': tokp.idx, 'tokid_end_char': tokp.idx + len(str(tokp)) - 1,
                     'tok_id': tok_id, 'tok_text': tok1})
                tok_id += 1
                all_tokens_data.append({'tokid_begin_char': tokp.idx + len(str(tokp)) - 1,
                                        'tokid_end_char': tokp.idx + len(str(tokp)),
                                        'tok_id': tok_id, 'tok_text': tok2})
                tok_id += 1
            # this particular case happens when there is a dot after month, sometimes spacy doesn't separate the
            # dot from the month, ex 'May.' gets parsed as ['May.'] and not ['May','.']
            elif len(found_dot_after_month) == 1 and len(found_dot_after_month[0][0]) == len(str(tokp)):
                # print('processing month name ending in .', str(tokp))
                tok1 = str(tokp)[0:len(str(tokp)) - 1]
                tok2 = str(tokp)[len(str(tokp)) - 1: len(str(tokp))]
                all_tokens_data.append({'tokid_begin_char': tokp.idx, 'tokid_end_char': tokp.idx + len(str(tokp)) - 1,
                                        'tok_id': tok_id, 'tok_text': tok1})
                tok_id += 1
                all_tokens_data.append({'tokid_begin_char': tokp.idx + len(str(tokp)) - 1,
                                        'tokid_end_char': tokp.idx + len(str(tokp)),
                                        'tok_id': tok_id, 'tok_text': tok2})
                tok_id += 1

            else:
                token_data = {'tokid_begin_char': tokp.idx, 'tokid_end_char': tokp.idx + len(tokp.text),
                              'tok_id': tok_id, 'tok_text': tokp.text}
                all_tokens_data.append(token_data)
                tok_id += 1
        return [{'offset': tok_info['tokid_begin_char'],
                 'length': tok_info['tokid_end_char'] - tok_info['tokid_begin_char'],
                 'token': tok_info['tok_text']} for tok_info in all_tokens_data]
def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--article_to_url_path',
                        default='DWIE/data/article_id_to_url.json',
                        help='Path to the file that contains the list of article ids with '
                             'respective url paths to api.dw')

    parser.add_argument('--annos_path',
                        default='DWIE/data/annos',
                        help='The path to where the annotations without the content are located')

    parser.add_argument('--output_path',
                        default='DWIE/data/annos_with_content',
                        help='The path to where the annotations without the content are located')

    parser.add_argument('--tokenize',
                        default=False,
                        type='bool',
                        help='Whether to tokenize the content using the default tokenizer (the one used in the paper)')

    args = parser.parse_args()

    article_id_to_url_path = args.article_to_url_path
    annos_path = args.annos_path
    output_path = args.output_path
    should_tokenize = args.tokenize
    if should_tokenize:
        tokenizer = TokenizerCPN()

    os.makedirs(output_path, exist_ok=True)

    ids_to_new_ids = dict()
    # some ids seem to be different, for now only this one:
    ids_to_new_ids[18525950] = 19026607

    content_to_new_content = {'DW_40663341': [('starting with Sunday\'s', 'starting Sunday\'s'),
                                              ('$1 million (€840,000)', 'one million dollars (840,000 euros)'),
                                              ('who kneel in protest during', 'to kneel in protest during')]}

    article_id_to_url_json = json.load(open(article_id_to_url_path))

    articles_done = 0
    total_articles = len(article_id_to_url_json)
    problematic_articles = set()
    problematic_hash_articles = set()

    for curr_article in article_id_to_url_json:
        article_id = curr_article['id']
        article_url = curr_article['url']
        article_id_nr = int(article_id[3:])
        if article_id_nr in ids_to_new_ids:
            article_url = article_url.replace(str(article_id_nr), str(ids_to_new_ids[article_id_nr]))
        article_hash = curr_article['hash']
        print('fetching {} out of {} articles -'.format(articles_done, total_articles), curr_article)

        annos_only_art_path = os.path.join(annos_path, '{}.json'.format(article_id))
        annos_only_json = json.load(open(annos_only_art_path))

        done = False
        attempts = 0
        while not done and attempts <= 3:
            # try:
            a = requests.get(article_url, allow_redirects=True).json()
            if 'name' in a:
                article_title = a['name']
            else:
                print('WARNING: no name detected for ', article_id)
                article_title = ''
            if 'teaser' in a:
                article_teaser = a['teaser']
            else:
                print('WARNING: no teaser detected for ', article_id)
                article_teaser = ''

            if 'text' in a:
                article_text = a['text']
            else:
                print('WARNING: no text detected for ', article_id)
                article_text = ''

            article_content_no_strip = '{}\n{}\n{}'.format(article_title, article_teaser, article_text)
            article_content = article_content_no_strip

            if article_id in content_to_new_content:
                for str_dw, str_dwie in content_to_new_content[article_id]:
                    article_content = article_content.replace(str_dw, str_dwie)

            if 'mentions' in annos_only_json:
                for idx_mention, curr_mention in enumerate(annos_only_json['mentions']):
                    curr_mention_text = curr_mention['text'].replace(' ', ' ')
                    curr_mention_text = curr_mention_text.replace('​', '')
                    solved = False
                    if article_content[curr_mention['begin']:curr_mention['end']] != curr_mention_text:
                        curr_mention_begin = curr_mention['begin']
                        curr_mention_end = curr_mention['end']
                        offset = 0

                        if not solved:
                            print('--------------------------------')
                            print('ERROR ALIGNMENT: texts don\'t match for {}: "{}" vs "{}", the textual content of '
                                  'the files won\'t be complete '
                                  .format(article_id, article_content[curr_mention['begin']:curr_mention['end']],
                                          curr_mention_text))
                            print('--------------------------------')
                            problematic_articles.add(article_id)
                        else:
                            curr_mention['begin'] = curr_mention_begin - offset
                            curr_mention['end'] = curr_mention_end - offset

            if not should_tokenize:
                annos_json = OrderedDict({'id': annos_only_json['id'],
                                          'content': article_content,
                                          'tags': annos_only_json['tags'],
                                          'mentions': annos_only_json['mentions'],
                                          'concepts': annos_only_json['concepts'],
                                          'relations': annos_only_json['relations'],
                                          'frames': annos_only_json['frames'],
                                          'iptc': annos_only_json['iptc']})
            else:
                tokenized = tokenizer.tokenize(article_content)
                tokens = list()
                begin = list()
                end = list()
                for curr_token in tokenized:
                    tokens.append(curr_token['token'])
                    begin.append(curr_token['offset'])
                    end.append(curr_token['offset'] + curr_token['length'])
                annos_json = OrderedDict({'id': annos_only_json['id'],
                                          'content': article_content,
                                          'tokenization': OrderedDict({'tokens': tokens, 'begin': begin, 'end': end}),
                                          'tags': annos_only_json['tags'],
                                          'mentions': annos_only_json['mentions'],
                                          'concepts': annos_only_json['concepts'],
                                          'relations': annos_only_json['relations'],
                                          'frames': annos_only_json['frames'],
                                          'iptc': annos_only_json['iptc']})

            hash_content = hashlib.sha1(article_content.encode("UTF-8")).hexdigest()

            if hash_content != article_hash:
                print('!!ERROR - hash doesn\'t match for ', article_id)
                problematic_hash_articles.add(article_id)
                attempts += 1

            sleep(.1)
            done = True
        if done:
            articles_done += 1
            # annos_with_content_art_path = os.path.join(annos_path, '{}.json'.format(article_id))
            annos_with_content_art_path = os.path.join(output_path, '{}.json'.format(article_id))
            json.dump(annos_json, open(annos_with_content_art_path, 'wt'), indent=4,
                      ensure_ascii=False)

    print('nr of fetched articles: ', articles_done)
    print('Mention span errors detected in {} articles: '.format(len(problematic_articles)), problematic_articles)
    print('Hash inconsistencies detected in {} articles: '.format(len(problematic_hash_articles)),
          problematic_hash_articles)

    if len(problematic_hash_articles) > 0:
        print('ERROR - A hash inconsistency was detected in one or more articles while fetching from dw.api, '
              'please contact klim.zaporojets@ugent.be to look into this issue.')

    if len(problematic_articles) > 0:
        print('ERROR - An annotation inconsistency was detected in one or more articles while fetching from dw.api, '
              'please contact klim.zaporojets@ugent.be to look into this issue.')