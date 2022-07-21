import json
from argparse import ArgumentParser
from transformers import BertTokenizer


def map_index(pieces):
    idxs = []
    for i, piece in enumerate(pieces):
        if i == 0:
            idxs.append([0, len(piece)])
        else:
            _, last = idxs[-1]
            idxs.append([last, last + len(piece)])
    return idxs


def convert(input_file, output_file, tokenizer):
    with open(input_file, 'r', encoding='utf-8') as r, \
            open(output_file, 'w', encoding='utf-8') as w:
        for line in r:
            doc = json.loads(line)
            doc_id = doc['doc_key']
            sentences = doc['sentences']
            sent_num = len(sentences)
            entities = doc.get('ner', [[] for _ in range(sent_num)])
            relations = doc.get('relations', [[] for _ in range(sent_num)])
            events = doc.get('events', [[] for _ in range(sent_num)])

            offset = 0
            for i, (sent_tokens, sent_entities, sent_relations, sent_events) in enumerate(zip(
                sentences, entities, relations, events
            )):
                sent_id = '{}-{}'.format(doc_id, i)
                pieces = [tokenizer.tokenize(t) for t in sent_tokens]
                word_lens = [len(p) for p in pieces]
                idx_mapping = map_index(pieces)

                sent_entities_ = []
                sent_entity_map = {}
                for j, (start, end, entity_type) in enumerate(sent_entities):
                    start, end = start - offset, end - offset + 1
                    entity_id = '{}-E{}'.format(sent_id, j)
                    entity = {
                        'id': entity_id,
                        'start': start, 'end': end,
                        'entity_type': entity_type,
                        # Mention types are not included in DyGIE++'s format
                        'mention_type': 'UNK',
                        'text': ' '.join(sent_tokens[start:end])}
                    sent_entities_.append(entity)
                    sent_entity_map[start] = entity

                sent_relations_ = []
                for j, (start1, end1, start2, end2, rel_type) in enumerate(sent_relations):
                    start1, end1 = start1 - offset, end1 - offset
                    start2, end2 = start2 - offset, end2 - offset
                    arg1 = sent_entity_map[start1]
                    arg2 = sent_entity_map[start2]
                    relation_id = '{}-R{}'.format(sent_id, j)
                    rel_type = rel_type.split('.')[0]
                    relation = {
                        'relation_type': rel_type,
                        'id': relation_id,
                        'arguments': [
                            {
                                'entity_id': arg1['id'],
                                'text': arg1['text'],
                                'role': 'Arg-1'
                            },
                            {
                                'entity_id': arg2['id'],
                                'text': arg2['text'],
                                'role': 'Arg-2'
                            },
                        ]
                    }
                    sent_relations_.append(relation)

                sent_events_ = []
                for j, event in enumerate(sent_events):
                    event_id = '{}-EV{}'.format(sent_id, j)
                    if len(event[0]) == 3:
                        trigger_start, trigger_end, event_type = event[0]
                    elif len(event[0]) == 2:
                        trigger_start, event_type = event[0]
                        trigger_end = trigger_start
                    trigger_start, trigger_end = trigger_start - offset, trigger_end - offset + 1
                    event_type = event_type.replace('.', ':')
                    args = event[1:]
                    args_ = []
                    for arg_start, arg_end, role in args:
                        arg_start, arg_end = arg_start - offset, arg_end - offset
                        arg = sent_entity_map[arg_start]
                        args_.append({
                            'entity_id': arg['id'],
                            'text': arg['text'],
                            'role': role
                        })
                    event_obj = {
                        'event_type': event_type,
                        'id': event_id,
                        'trigger': {
                            'start': trigger_start,
                            'end': trigger_end,
                            'text': ' '.join(sent_tokens[trigger_start:trigger_end])
                        },
                        'arguments': args_
                    }
                    sent_events_.append(event_obj)

                sent_ = {
                    'doc_id': doc_id,
                    'sent_id': sent_id,
                    'entity_mentions': sent_entities_,
                    'relation_mentions': sent_relations_,
                    'event_mentions': sent_events_,
                    'tokens': sent_tokens,
                    'pieces': [p for w in pieces for p in w],
                    'token_lens': word_lens,
                    'sentence': ' '.join(sent_tokens)
                }
                w.write(json.dumps(sent_) + '\n')

                offset += len(sent_tokens)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the input file')
    parser.add_argument('-o', '--output', help='Path to the output file')
    args = parser.parse_args()
    
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased',
                                               do_lower_case=False)
    convert(args.input, args.output, bert_tokenizer)
