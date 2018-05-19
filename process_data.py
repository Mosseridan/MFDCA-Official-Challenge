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


def get_words(filenames):
    words = []
    for line in get_lines(filenames):
        words += line.split()
    return words


def get_word_count(words):
    word_count = {}
    for word in words:
        try:
            word_count[word]+=1
        except KeyError:
            word_count[word] = 1
    return word_count


def get_vocabulary_from_word_count(word_count):
    return sorted(word_count.keys())


def get_vocabulary(words):
    return get_vocabulary_from_word_count(get_word_count(words))


def write_word_count_to_csv(word_count, filename):
    splitext = os.path.splitext(filename)
    if(splitext[1] != '.csv'):
        filename = splitext[0]+'.csv'
    sorted_wc = sort_dict_by_val(word_count)
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
        writer.writerows([[word] for word in vocabulary])

def get_stats(words):
    info = {}
    info['words'] = words
    info['word_count'] = get_word_count(words)
    # info['sorted_wc'] = sort_dict_by_val(info['word_count'])
    info['vocabulary'] = get_vocabulary_from_word_count(info['word_count'])
    info['total_wc'] = 0
    for wc in info['word_count'].values():
         info['total_wc'] += wc
    info['vocabulary_size'] = len(info['vocabulary'])
    return info


filenames = [os.path.abspath(os.path.join('MFDCA-DATA','FraudedRawData','User'+str(i))) for i in range(40)]
# filenames = [os.path.abspath(os.path.join('MFDCA-DATA','dummy','User'+str(i))) for i in range(2)]
users = {}
seg_size = 100

for filename in filenames:
    username = os.path.basename(filename)
    words = get_words(filename)  
    user = get_stats(words)
    user['segments'] = []

    for i in range(0, len(words), seg_size):
        segment_words = words[i:i+seg_size]
        seg_stats = get_stats(segment_words)
        write_word_count_to_csv(seg_stats['word_count'], os.path.abspath(os.path.join('MFDCA-DATA','WordCount',username+'_Seg'+str(i/seg_size))))
        write_vocabulary_to_csv(seg_stats['vocabulary'], os.path.abspath(os.path.join('MFDCA-DATA','Vocabulary',username+'_Seg'+str(i/seg_size))))
        user['segments'].append(seg_stats)

    users[username] = user
    write_word_count_to_csv(user['word_count'], os.path.abspath(os.path.join('MFDCA-DATA','WordCount',username)))
    write_vocabulary_to_csv(user['vocabulary'], os.path.abspath(os.path.join('MFDCA-DATA','Vocabulary',username)))
    print('\n@@ '+username+' Total numbr of commands: '+str(user['total_wc']))
    print('\n@@ '+username+' Vocabulary size: '+str(user['vocabulary_size']))

    
    
        
        


words = get_words(filenames)
total_stats = get_stats(words)
write_word_count_to_csv(total_stats['word_count'], os.path.abspath(os.path.join('MFDCA-DATA','WordCount','TotalWordCount')))
write_vocabulary_to_csv(total_stats['vocabulary'], os.path.abspath(os.path.join('MFDCA-DATA','Vocabulary','TotalVocabulary')))


print('@@Word Count:')
# for key, val in sorted_wc:
#     print(key+': '+str(val))
print('\n@@ Total numbr of commands: '+str(total_stats['total_wc']))
print('\n@@ Commands per user: '+str(total_stats['total_wc']/40))
print('\n@@ Vocabulary size: '+str(total_stats['vocabulary_size']))

common_vocabulary_users = list(
    filter(lambda word:
        all([word in users[user]['vocabulary'] for user in users.keys()]) , total_stats['vocabulary']))

print('\n@@ Common vocabulary users:')
print(common_vocabulary_users)
print('\n@@ Common vocabulary users size: '+str(len(common_vocabulary_users)))

total_stats['common_vocabulary_users'] = common_vocabulary_users
total_stats['common_vocabulary_users_size'] = len(common_vocabulary_users)


segments = []
for username in users.keys():
    for segment in users[username]['segments']:
        segments.append(segment['vocabulary'])

common_vocabulary_segments = list(
    filter(lambda word:
        all([word in segment for segment in segments]), total_stats['vocabulary']))


print('\n@@ Common vocabulary segments:')
print(common_vocabulary_segments)
print('\n@@ Common vocabulary segments size: '+str(len(common_vocabulary_segments)))

total_stats['common_vocabulary_segments'] = common_vocabulary_segments
total_stats['common_vocabulary_segments_size'] = len(common_vocabulary_segments)

out_data = {
    'total_stats': total_stats,
    'users': users
}

with open(os.path.abspath(os.path.join('MFDCA-DATA','stats.json')), 'w') as outfile:
    json.dump(out_data, outfile)

print('\n@@ DONE!')