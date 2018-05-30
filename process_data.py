import os
import csv
import json


def sort_dict_by_val(dict, reverse=True):
    return [(key, dict[key]) for key in sorted(dict, key=dict.get, reverse=reverse)]


def sort_dict_by_key(dict, reverse=False):
    return [(key, dict[key]) for key in sorted(dict.keys(), reverse=reverse)]


def get_lines(filenames):
    lines = []
    if (type(filenames) is str):
        filenames = [filenames]
    for filename in filenames:
        with open(filename,"r") as file:
            lines += [line.split('\n')[0] for line in file]
    return lines


def get_terms(filenames):
    terms = []
    for line in get_lines(filenames):
        terms += line.split()
    return terms


def get_sequences(terms, seq_size):
    sequences = []
    for i in range(len(terms)-seq_size):
        sequences += [" ".join(terms[i:i+seq_size])]
    return sequences

def get_sequences_from_file(filenames , seq_size):
    terms = []
    for line in get_lines(filenames):
        terms += line.split()
    return get_sequences(terms, seq_size)
    

def get_term_frequency(terms):
    term_frequency = {}
    for cmd in terms:
        try:
            term_frequency[cmd]+=1
        except KeyError:
            term_frequency[cmd] = 1
    return term_frequency


def get_vocabulary_from_term_frequency(term_frequency):
    return sorted(term_frequency.keys())


def get_vocabulary(terms):
    return get_vocabulary_from_term_frequency(get_term_frequency(terms))


def write_term_frequency_to_csv(term_frequency, filename):
    splitext = os.path.splitext(filename)
    if(splitext[1] != '.csv'):
        filename = splitext[0]+'.csv'
    sorted_wc = sort_dict_by_val(term_frequency)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['cmd','count'])
        writer.writerows(sorted_wc)


def write_vocabulary_to_csv(vocabulary, filename):
    splitext = os.path.splitext(filename)
    if(splitext[1] != '.csv'):
        filename = splitext[0]+'.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([[cmd] for cmd in vocabulary])


def get_stats(terms):
    info = {}
    info['terms'] = terms
    info['term_frequency'] = get_term_frequency(terms)
    # info['sorted_wc'] = sort_dict_by_val(info['term_frequency'])
    info['vocabulary'] = get_vocabulary_from_term_frequency(info['term_frequency'])
    info['total_wc'] = 0
    for wc in info['term_frequency'].values():
         info['total_wc'] += wc
    info['vocabulary_size'] = len(info['vocabulary'])
    return info


def get_users_stats():
    users = {}
    seg_size = 100
    with open(os.path.abspath(os.path.join('MFDCA-DATA','challengeToFill.csv')), newline='') as tags_csv:
        reader = csv.reader(tags_csv)
        next(reader, None) #skip headers
        for row in reader:
            username = row[0]
            filename = os.path.abspath(os.path.join('MFDCA-DATA','FraudedRawData',username))
            terms = get_terms(filename)
            user = get_stats(terms)
            user['segments'] = []

            for i in range(0, len(terms), seg_size): 
                segment_terms = terms[i:i+seg_size]
                seg_stats = get_stats(segment_terms)
                seg_stats['label'] = row[int(i/seg_size)+1]
                user['segments'].append(seg_stats)
            users[username] = user
    
    return users


def get_segments_from_users(users):
    segments = []
    for i in range(0,len(users.keys())):
        username = 'User'+str(i)
        for i,segment in enumerate(users[username]['segments']):
            segments.append({
                'username': username,
                'segment_number': i,
                'vocabulary': segment['vocabulary'],
                'term_frequency': segment['term_frequency'],
                'terms': segment['terms'],
                'label': segment['label']
            })
    return segments
        

def get_segments():
    seg_size = 100
    segments = []
    with open(os.path.abspath(os.path.join('MFDCA-DATA','challengeToFill.csv')), newline='') as tags_csv:
        reader = csv.reader(tags_csv)
        next(reader, None) #skip headers
        for row in reader:
            username = row[0]
            filename = os.path.abspath(os.path.join('MFDCA-DATA','FraudedRawData',username))
            terms = get_terms(filename)
            for i in range(0, len(terms), seg_size): 
                segment_terms = terms[i:i+seg_size]
                seg_stats = get_stats(segment_terms)
                seg_number = int(i/seg_size)
                seg_stats['username'] = username
                seg_stats['segment_number'] = seg_number    
                seg_stats['label'] = row[seg_number+1]
                segments.append(seg_stats)
    return segments
    
        

def main():
    filenames = [os.path.abspath(os.path.join('MFDCA-DATA','FraudedRawData','User'+str(i))) for i in range(40)]
    # filenames = [os.path.abspath(os.path.join('MFDCA-DATA','dummy','User'+str(i))) for i in range(2)]
    
    users = get_users_stats()
    seg_size = 100
    for username, user in users.items():  
        write_term_frequency_to_csv(user['term_frequency'], os.path.abspath(os.path.join('MFDCA-DATA','CmdCount',username)))
        write_vocabulary_to_csv(user['vocabulary'], os.path.abspath(os.path.join('MFDCA-DATA','Vocabulary',username)))
        for i,seg_stats in enumerate(user['segments']):
            write_term_frequency_to_csv(seg_stats['term_frequency'], os.path.abspath(os.path.join('MFDCA-DATA','CmdCount',username+'_Seg'+str(i))))
            write_vocabulary_to_csv(seg_stats['vocabulary'], os.path.abspath(os.path.join('MFDCA-DATA','Vocabulary',username+'_Seg'+str(i))))

    terms = get_terms(filenames)
    total_stats = get_stats(terms)
    write_term_frequency_to_csv(total_stats['term_frequency'], os.path.abspath(os.path.join('MFDCA-DATA','CmdCount','TotalCmdCount')))
    write_vocabulary_to_csv(total_stats['vocabulary'], os.path.abspath(os.path.join('MFDCA-DATA','Vocabulary','TotalVocabulary')))


    # print('@@cmd Count:')
    # for key, val in sorted_wc:
    #     print(key+': '+str(val))
    # print('\n@@ Total numbr of terms: '+str(total_stats['total_wc']))
    # print('\n@@ terms per user: '+str(total_stats['total_wc']/40))
    # print('\n@@ Vocabulary size: '+str(total_stats['vocabulary_size']))

    common_vocabulary_users = list(
        filter(lambda cmd:
            all([cmd in users[user]['vocabulary'] for user in users.keys()]) , total_stats['vocabulary']))

    # print('\n@@ Common vocabulary users:')
    # print(common_vocabulary_users)
    # print('\n@@ Common vocabulary users size: '+str(len(common_vocabulary_users)))

    total_stats['common_vocabulary_users'] = common_vocabulary_users
    total_stats['common_vocabulary_users_size'] = len(common_vocabulary_users)


    segments = get_segments_from_users(users)

    common_vocabulary_segments = list(
        filter(lambda cmd:
            all([cmd in segment['vocabulary'] for segment in segments]), total_stats['vocabulary']))


    # print('\n@@ Common vocabulary segments:')
    # print(common_vocabulary_segments)
    # print('\n@@ Common vocabulary segments size: '+str(len(common_vocabulary_segments)))

    total_stats['common_vocabulary_segments'] = common_vocabulary_segments
    total_stats['common_vocabulary_segments_size'] = len(common_vocabulary_segments)

    stats = {
        'total_stats': total_stats,
        'users': users
    }

    print('\n@@ Writing stats to stats.json')    
    with open(os.path.abspath(os.path.join('MFDCA-DATA','stats.json')), 'w') as outfile:
        json.dump(stats, outfile)

    print('\n@@ Writing segments to segments.json')        
    with open(os.path.abspath(os.path.join('MFDCA-DATA','segments.json')), 'w') as outfile:
        json.dump(segments, outfile)

    # dataset = {
    #     'features': {
    #         'username': [segment['username'] for segment in segments[0:1500]],
    #         'segment_number': [int(segment['segment_number']) for segment in segments[0:1500]],
    #         'terms': [segment['terms'] for segment in segments[0:1500]]
    #     },
    #     'labels': [int(segment['label']) for segment in segments[0:1500]],
    #     'vocabulary': stats['total_stats']['vocabulary']
    # }

    dataset = {
        'segments': segments[0:1500],
        'vocabulary': stats['total_stats']['vocabulary']
    }

    print('\n@@ Writing dataset to dataset.json')            
    with open(os.path.abspath(os.path.join('MFDCA-DATA','dataset.json')), 'w') as outfile:
        json.dump(dataset, outfile)

    print('\n@@ DONE!')


if __name__ == '__main__':
    main()