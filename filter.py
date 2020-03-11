#!/usr/bin/env python
# coding: utf-8

import re, string
import os
import time
import jieba
import argparse
from smart_open import open


def contain_city(line, city_list):
        # segment the sentence using jieba
        words = set(' '.join(jieba.cut(line, cut_all=False)).split(' '))
        for city in city_list:
            if city in words:
                return True
        return False

    
def process(root_path, file_list, city_list):
    for i, filename in enumerate(file_list):
        if i % 10 == 0:
            print('Currently the {}th document: '.format(i) + filename)

        # ------------ your website data
        with open(root_path + '/webdata/' + filename, encoding='utf-8') as fin:
            with open(root_path + '/new/' + filename, 'w', encoding='utf-8') as fout:
                for line in fin:
                    l = line
                    if l == '' or l.startswith('\r'):
                        continue
                    # drop alphabetic characters
                    l = re.sub(r'[a-zA-Z]', '', l)
                    # drop digits and punctuations
                    l = re.sub('[%s]' % (string.punctuation + string.digits), '', l)
                    # drop empty line
                    if l == '\r':
                        continue
                    isContain = contain_city(l, city_list)
                    if line.startswith('WARC-Target-URI:') or line.startswith('WARC-Date:'):
                        fout.write(line)
                    elif isContain:
                        fout.write(line + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, help='From 0')
    args = parser.parse_args()

    root_path = '/Users/joy/Desktop'    # ----------------- need to be replaced

    # create directory for segment videos
    try:
        if not os.path.exists("{}/new".format(root_path)):
            os.mkdir("{}/new".format(root_path))
    except OSError:
        raise OSError("Creation of the directory {} failed"
                      .format("{}/new".format(root_path)))

    city_list = []
    with open(root_path + '/China_Cities_Coordinates_CHN_ENG.csv') as f:    # --------- cities
        skip_head = True
        for line in f:
            if skip_head:
                skip_head = False
                continue
            else:
                city_list.append(line.split(',')[0])
    city_list = set(city_list)

    jieba.disable_parallel()

    start = args.id * 560
    end = min((args.id + 1) * 560, 56000)
    print('Start: {}, end: {}'.format(start, end))

    file_list = [f for f in os.listdir(root_path + '/webdata')      # ------------ your website data
                 if f.startswith('part-')][start:end]

    process(root_path, file_list, city_list)
