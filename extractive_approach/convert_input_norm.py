import json
from ds import split_doc_on_words
import os
import argparse
from collections import Counter
    
if os.getcwd().find('converters') >= 0:
    os.chdir("../")

parser = argparse.ArgumentParser(description='Help to parse args')
parser.add_argument('--data_path', type=str,
                    help='Path to the dataset, which will be converted to the spert format, this path must include files which have test and train substringes in their names')
parser.add_argument('--res_path', type=str,
                    help='This is the relative path from ./data/datasets folder. The full path is ./data/datasets/ + res_dir')
parser.add_argument('--filter_negative', action='store_true', help='If there is a need in filtering negative samples')
parser.add_argument('--use_valid', action='store_true', help='If there is a need use valid part')
parser.add_argument("--split_count_tokens", help="", nargs="?", type=int, const=256)
parser.add_argument('--use_normalization', action='store_true', help='If there is a need use normalization')


args = parser.parse_args()

ds_path = args.data_path
res_path = args.res_path
split_count_tokens = args.split_count_tokens

if res_path.find('/data/datasets/') > 0:
    raise ValueError('Error, --res_path must be relative path from ./data/datasets folder')

train_files = [os.path.join(ds_path, file) for file in os.listdir(ds_path) if file.find('train') >= 0]
if len(train_files) != 1:
    raise ValueError(
        "Error, dataset directory must contain only one file with the name, which include the substring 'train'")
train_file = train_files[0]

test_files = [os.path.join(ds_path, file) for file in os.listdir(ds_path) if file.find('test') >= 0]
if not args.use_valid:
    if len(test_files) != 1:
        raise ValueError(
            "Error, dataset directory must contain only one file with the name, which include the substring 'test'")
    else:
        test_file = test_files[0]

valid_files = [os.path.join(ds_path, file) for file in os.listdir(ds_path) if file.find('valid') >= 0]
if args.use_valid:
    if len(valid_files) != 1:
        raise ValueError(
            "Error, dataset directory must contain only one file with the name, which include the substring 'valid'")
    else:
        valid_file = valid_files[0]

if args.filter_negative:
    # what we consider to be negative classes
    negative_types = ['false', 'neg']

def tokenization(text):
    data_df = split_doc_on_words({'text': text}, language='other')
    if data_df is None:
        return None
    toks = []
    for row in data_df.iterrows():
        new_tok = dict.fromkeys(['forma', 'posStart', 'posEnd', 'len'])
        new_tok['posStart'] = row[1]['word_start_in_doc']
        new_tok['posEnd'] = row[1]['word_stop_in_doc']
        new_tok['forma'] = row[1]['word']
        new_tok['len'] = len(new_tok['forma'])
        new_tok['id'] = row[0]
        toks.append(new_tok)
    return toks


with open('./configs/spert_config_example_xlmroberta.conf', 'r') as f:
    spert_config = f.read()

ent_types_set = set()
rel_types_set = set()

if args.use_valid:
    print('use valid as test!')
    test_file = valid_file

print('train file path:\n%s' % (train_file))
print('test file path:\n%s' % (test_file))

for mode, mode_path in [('train', train_file), ('test', test_file)]:
    with open(mode_path) as f:
        data = json.load(f)
    
    if split_count_tokens:
        print(f"Num. docs before splitting in {mode}: {len(data)}")

    lost_ner, lost_re = [], []
    spert_data = []
    for k, uniq_id in enumerate(data):
        
        # This doc run error in spert model
        if "phaedra" in train_file.lower():
            if str(uniq_id["text_id"])=="19889885":
                continue

        # я считаю, что разметка неразрывная, ибо только её и берет spert
        # поэтому от разрывных сущностей я беру только первую часть
        ent_spans = [list(ent['spans'][0].values()) for ent in uniq_id['entities'].values()]
        ent_ids = list(uniq_id['entities'].keys())
        ent_vals = list(uniq_id['entities'].values())

        # также берем лишь один тег
        ent_types = [ent['tag'][0] for ent in uniq_id['entities'].values()]
        text = uniq_id['text']



        new_sample = {}
        toks = tokenization(text)
        if not toks:
            continue
        new_sample['tokens'] = toks

        # fix multilabel ner
        multi_ents = []
        for ind_ent, ent in enumerate(ent_spans[:-1]):
            for ind_cand_ent, cand_ent in enumerate(ent_spans[ind_ent+1:], ind_ent+1):
                if ent==cand_ent:
                    multi_ents.append([ind_ent, ind_cand_ent])
        if multi_ents != []:
            remove_ent_inds = []
            for inds_multi_ents in multi_ents:
                tags_multi_ents = [ent_types[i] for i in inds_multi_ents]
                if "ADR" in tags_multi_ents:
                    remove_ent_inds.extend([i for i in inds_multi_ents if ent_types[i] != "ADR"])
                elif "Medication:MedTypeDrugname" in tags_multi_ents:
                    remove_ent_inds.extend([i for i in inds_multi_ents if ent_types[i] != "Medication:MedTypeDrugname"])
                else:
                    remove_ent_inds.extend(inds_multi_ents[1:])
            remove_ent_inds = sorted(remove_ent_inds, reverse=True)
            lost_ner+=[ent_types[i] for i in remove_ent_inds]
            for remove_ind in remove_ent_inds:
                ent_ids.pop(remove_ind)
                ent_spans.pop(remove_ind)
                ent_types.pop(remove_ind)
            del multi_ents, remove_ent_inds

        # заполняем поле entities в spert формате
        new_sample['entities'] = []
        ent_num = 0
        for ann, ann_id, ann_type, ann_val in zip(ent_spans, ent_ids, ent_types, ent_vals):
            new_ent = dict.fromkeys(['type', 'start', 'end', 'text', 'origin_entity_id'])
            new_ent['id'] = ann_id
            
            # находим токены
            e_start = ann[0]
            e_end = ann[1]
            ent_toks = []

            for tok in new_sample['tokens']:
                cond_1 = (tok['posStart'] >= e_start) & (
                        tok['posEnd'] <= e_end)
                cond_4 = (tok['posStart'] <= e_start) & (
                        tok['posEnd'] >= e_end)
                cond_2 = (tok['posStart'] < e_start) & (
                        tok['posStart'] < e_end) & (
                                 tok['posEnd'] > e_start)
                cond_3 = (tok['posStart'] > e_start) & (
                        tok['posEnd'] > e_end) & (
                                 tok['posStart'] < e_end)

                if (cond_1 | cond_2 | cond_3 | cond_4):
                    ent_toks.append(tok)

            new_ent['type'] = ann_type
            try:
                new_ent['start'] = ent_toks[0]['id']
                new_ent['end'] = ent_toks[-1]['id'] + 1
            except:
                print('In %s file in review %s entity number %s was lost' % (mode_path, k, ent_num))
                ent_num += 1
                continue
            ent_num += 1
            new_ent['text'] = ' '.join([tok['forma'] for tok in ent_toks])
            if args.use_normalization:
                if ann_type in ['ADR', 'Disease:DisTypeIndication']: # если тег показывает, что сущность - АДР или симптом
                    if 'MedDRA' in ann_val:                          # и если есть поле meddra    
                        new_ent['normalization'] = ann_val['MedDRA'] # то в поле с нормализованным понятием будет текст из meddra
                    else:
                        new_ent['normalization'] = ''
            new_sample['entities'].append(new_ent)
            ent_types_set.add(ann_type)
        new_sample['tokens'] = [tok['forma'] for tok in toks]
        new_sample['tok_spans'] = [[tok['posStart'], tok['posEnd']] for tok in toks]
        new_sample['text'] = uniq_id['text']
       
        # заполняем поле с relations
        new_sample['relations'] = []
        for r_num, rel in enumerate(uniq_id['relations']):
            if 'relation_class' in rel:
                if args.filter_negative:
                    if str(rel['relation_class']) == '0':
                        continue
                rel_type = rel['relation_type'] + '_' + str(rel['relation_class'])
            else:
                rel_type = rel['relation_type']
            if args.filter_negative:
                if rel_type in negative_types:
                    continue
            rel_types_set.add(rel_type)
            new_rel = dict.fromkeys(['type', 'head', 'tail'])
            new_rel['type'] = rel_type
            for ent_id, ent in enumerate(new_sample['entities']):
                if str(ent['id']) == str(rel['first_entity']['entity_id']):
                    new_rel['head'] = ent_id
                if str(ent['id']) == str(rel['second_entity']['entity_id']):
                    new_rel['tail'] = ent_id
            if new_rel['head'] is None or new_rel['tail'] is None:
                lost_re.append(rel_type)
                #print('In %s file in review %s relation number %s was lost' % (mode_path, k, r_num))
                continue
            new_sample['relations'].append(new_rel)
        
        # Механизм разрезания длинных документов:
        if split_count_tokens:
            id_sample = uniq_id["text_id"]
            
            # если кол-во токенов меньше чем трешхолд, то не режем документ
            if len(new_sample["tokens"]) <= split_count_tokens:
                new_sample["split_info"] = f"{id_sample}_1_1"
                spert_data.append(new_sample)
            else:
                # TODO: doc_parts = func split_doc(id_sample, new_sample, split_count_tokens)

                n_split = len(new_sample["tokens"])//split_count_tokens
                if len(new_sample["tokens"])%split_count_tokens > 0:
                    n_split+=1

                prev_split_token = -1
                for i in range(1, n_split+1):
                    if i==1:
                        first_split_token = 0
                        first_split_char = 0
                    else:
                        first_split_token = prev_split_token+1
                        first_split_char = new_sample["tok_spans"][prev_split_token][1]
                    if i==n_split:
                        last_split_token = len(new_sample["tokens"])-1
                        last_split_char = len(new_sample["text"])
                    else:
                        last_split_token = prev_split_token+1+split_count_tokens-1

                        for ind_ent, ent in enumerate(new_sample['entities']):
                            if first_split_token <= ent["start"] < last_split_token:
                                if ent["end"]-1 > last_split_token:
                                    last_split_token = ent["end"]-1

                        last_split_char = new_sample["tok_spans"][last_split_token][1]-1

                    #print(k, first_split_token, last_split_token, first_split_char, last_split_char)

                    tok_spans, entities, entities_inds, relations = [], [], [], []
                    for span in new_sample['tok_spans'][first_split_token:last_split_token+1]:
                        tok_spans.append(span.copy())
                        if i != 1:
                            tok_spans[-1][0] -= first_split_char
                            tok_spans[-1][1] -= first_split_char

                    for ind_ent, ent in enumerate(new_sample['entities']):
                        if first_split_token<=ent["start"] and last_split_token>=ent["end"]-1:
                            entities.append(ent.copy())
                            entities_inds.append(ind_ent)
                            if i != 1:
                                entities[-1]["start"] -= first_split_token
                                entities[-1]["end"] -= first_split_token

                    for rel in new_sample['relations']:
                        if rel["head"] in entities_inds and rel["tail"] in entities_inds:
                            relations.append(rel.copy())
                            relations[-1]["head"] = entities_inds.index(rel["head"])
                            relations[-1]["tail"] = entities_inds.index(rel["tail"])

                    new_split_sample = {
                        'split_info': f'{id_sample}_{i}_{n_split}',
                        'text': new_sample["text"][first_split_char:last_split_char+1], 
                        'tokens': new_sample["tokens"][first_split_token:last_split_token+1], 
                        'entities': entities, 
                        'tok_spans': tok_spans, 
                        'relations': relations
                    }

                    spert_data.append(new_split_sample)

                    prev_split_token = last_split_token
        else:
            spert_data.append(new_sample)

    if split_count_tokens:
        print(f"Num. docs after splitting in {mode}: {len(spert_data)}")

    print(f"Lost NER: {len(lost_ner)}")
    for k, v in Counter(lost_ner).items():
        print("\t", k, v)
    print(f"Lost RE: {len(lost_re)}")
    for k, v in Counter(lost_re).items():
        print("\t", k, v)
    
    save_path = os.path.join('./data/datasets/', res_path)
    os.makedirs(save_path, exist_ok=True)
    with open(save_path + '/' + mode + '_spert' + '.json', 'w') as f:
        json.dump(spert_data, f)
    print('%s file in spert format saved in:\n%s' % (mode, os.path.join(save_path, mode + '_spert' + '.json')))

# типы спертовские дампим
types_d = {k: {} for k in ['entities', 'relations']}

for ent_type in ent_types_set:
    types_d['entities'][ent_type] = {'short': ent_type,
                                     'verbose': ent_type}

for rel_type in rel_types_set:
    types_d['relations'][rel_type] = {'short': rel_type,
                                      'verbose': rel_type,
                                      'symmetric': False}

with open(save_path + '/' + 'types' + '.json', 'w') as f:
    json.dump(types_d, f)

# конфиг спертовский дампим
save_path = './configs/'

ds_name = res_path.replace('/', '_')
ds_name = ds_name.strip('_')
ds_name = ds_name.replace('folds', 'fold')

fold_conf = spert_config.replace('data/datasets/RDRS_multicontext/folds_4_classes/1/RDRS_spert_train.json',
                                 os.path.join('data/datasets/', res_path, 'train_spert' + '.json'))
fold_conf = fold_conf.replace('data/datasets/RDRS_multicontext/folds_4_classes/1/RDRS_spert_test.json',
                              os.path.join('data/datasets/', res_path, 'test_spert' + '.json'))
fold_conf = fold_conf.replace('data/datasets/RDRS_multicontext/folds_4_classes/1/RDRS_types.json',
                              os.path.join('data/datasets/', res_path, 'types' + '.json'))
fold_conf = fold_conf.replace('RDRS_multicontext_fold_1', ds_name)

print('Config, which will be used for train:\n%s' % (ds_name + '_' + 'train' '.conf'))
print('DS label:\n%s' % ds_name)

with open(save_path + ds_name + '_' + 'train' '.conf', 'w') as f:
    f.write(fold_conf)
