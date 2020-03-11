#!/usr/bin/env python
# coding: utf-8
#%%
import re
import csv
n=0
text=[]
prefix='WARC-Target-URI: '

with open ('webdata/part-00002-9f030fc2-0a97-4d33-9eb6-0325d8263f3b-c000.txt','r') as lines:
    with open ('fweb.csv') as webs:
        for web in webs:
            web = '.*\.' + web[:-1] + '\..*'
            print(web)
            for num,line in enumerate(lines):
                if not line.startswith(prefix):
                    continue
                if re.search(web,line):
                    text.append(line)
                    n=num
                    print(line)
                if num-n==8:
                    text.append(line)
                    n=0

with open("texts_topwebs.txt", "w+") as f:
    for i in text:
        f.write(i)
