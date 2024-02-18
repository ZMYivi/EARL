import config

import argparse
import json

import nltk
import numpy as np

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

kb = {}

def load_data(opt, fname):
    global kb
    raw = []
    kg_all = []
    with open('%s/%s' % (opt.data_dir, fname), encoding='utf8') as f:
        kb = json.loads(f.readline().strip())
        for idx, line in enumerate(f):
            if idx == 100:
                break
            mtknz_post = nltk.tokenize.MWETokenizer(separator='_')
            mtknz_resp = nltk.tokenize.MWETokenizer(separator='_')
            
            content = json.loads(line.strip())
            context = []
            session = []
            kg = []
            entity2idx = {}
            ents = {}
            for ent in content['kg']:
                mtknz_post.add_mwe(nltk.word_tokenize(ent.lower()))
                goldens = content['kg'][ent]
                kg += [goldens[:]]
                for triple in kb[ent]:
                    if triple not in goldens:
                        kg[-1] += [triple[:]]
                        if len(kg[-1]) == opt.triple_num:
                            break
                for (h, r, t) in kg[-1]:
                    mtknz_resp.add_mwe(nltk.word_tokenize(h.lower()))
                    mtknz_resp.add_mwe(nltk.word_tokenize(t.lower()))
                    
            eidx = 0
            for triples in kg:
                ent = mtknz_post.tokenize(nltk.word_tokenize(triples[0][0].lower()))
                if ent[0] not in ents:
                    ents[ent[0]] = len(ents)
                entity2idx[ent[0]] = -1
                for triple in triples:
                    triple[0] = mtknz_resp.tokenize(nltk.word_tokenize(triple[0].lower()))
                    triple[2] = mtknz_resp.tokenize(nltk.word_tokenize(triple[2].lower()))
                    if len(triple[2]) == 0:
                        triple[2] = ['']
                        continue
                    entity2idx[triple[2][0]] = eidx
                    eidx += 1

            context += [mtknz_resp.tokenize(nltk.word_tokenize(content['dialog'][0][0].lower()))]
            for turn in content['dialog'][1:]:
                resp = mtknz_resp.tokenize(nltk.word_tokenize(turn[0].lower()))[:opt.max_length]
                post = [x for t in context for x in t]
                #if len(resp) > 0 and (not any([ent in resp for ent in entity2idx]) or any([ent in post for ent in ents])):
                if len(resp) > 0:
                    session += [{'post': post, 'response': resp, 'kg': kg, 'entity2idx': entity2idx, 'head_ents': ents}]
                context += [['_EOS'] + mtknz_resp.tokenize(nltk.word_tokenize(turn[0].lower()))]
            if session:
                raw += [session]
    
    data_train, data_valid, data_test_seen, data_test_unseen = raw[:-4 * opt.set_size], raw[-4 * opt.set_size:-2 * opt.set_size], raw[-2 * opt.set_size: -opt.set_size], raw[-opt.set_size:]
    return data_train, data_valid, data_test_seen, data_test_unseen

def build_vocab(opt, path, data):
    print("Creating vocabulary...")
    vocab = {}
    vocab_kg = {}
    for i, session in enumerate(data):
        if i % 1000 == 0:
            print("    processing line %d" % i)
        pair = session[-1]
        for triples in pair['kg']:
            for triple in triples:
                for token in triple[0] + [triple[1]] + triple[2]:
                    if token in vocab_kg:
                        vocab_kg[token] += 1
                    else:
                        vocab_kg[token] = 1
                vocab[triple[1]] = vocab.get(triple[1], 0) + 1
        for token in session[-1]['post'] + [x for turn in session for x in turn['response']]:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
    vocab_kg_list = sorted(vocab_kg, key=vocab_kg.get, reverse=True)
    vocab_list = _START_VOCAB + ['_MASK%d' % x for x in range(16)] + [w for w in sorted(vocab, key=vocab.get, reverse=True) if w[-4:] == '/rel' or '_' not in w] + [ent for ent in vocab_kg_list if ent not in vocab]

    if len(vocab_list) > opt.symbols:
        vocab_list = vocab_list[:opt.symbols]
    else:
        opt.symbols = len(vocab_list)
        
    print("Loading word vectors...")
    vectors = {}
    with open(path, encoding='utf8') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("    processing line %d" % i)
            s = line.strip()
            word = s[:s.find(' ')]
            vector = s[s.find(' ')+1:]
            vectors[word] = vector
    
    embed = []
    for word in vocab_list:
        if word in vectors:
            vector = list(map(float, vectors[word].split()))
        else:
            vector = np.zeros((opt.embed_units), dtype=np.float32)
        embed.append(vector)
    embed = np.array(embed, dtype=np.float32)
            
    return vocab_list, embed

if __name__ == '__main__':
    opt = config.get_options()
    data_train, data_valid, data_test_seen, data_test_unseen = load_data(opt, 'data.json')
    vocab, embed = build_vocab(opt, opt.data_dir + '/vector.txt', data_train)