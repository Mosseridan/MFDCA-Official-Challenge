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


def get_commands(filenames):
    commands = []
    for line in get_lines(filenames):
        commands += line.split()
    return commands


def get_cmd_count(commands):
    cmd_count = {}
    for cmd in commands:
        try:
            cmd_count[cmd]+=1
        except KeyError:
            cmd_count[cmd] = 1
    return cmd_count


def get_vocabulary_from_cmd_count(cmd_count):
    return sorted(cmd_count.keys())


def get_vocabulary(commands):
    return get_vocabulary_from_cmd_count(get_cmd_count(commands))


def write_cmd_count_to_csv(cmd_count, filename):
    splitext = os.path.splitext(filename)
    if(splitext[1] != '.csv'):
        filename = splitext[0]+'.csv'
    sorted_wc = sort_dict_by_val(cmd_count)
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


def get_stats(commands):
    info = {}
    info['commands'] = commands
    info['cmd_count'] = get_cmd_count(commands)
    # info['sorted_wc'] = sort_dict_by_val(info['cmd_count'])
    info['vocabulary'] = get_vocabulary_from_cmd_count(info['cmd_count'])
    info['total_wc'] = 0
    for wc in info['cmd_count'].values():
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
            commands = get_commands(filename)
            user = get_stats(commands)
            user['segments'] = []

            for i in range(0, len(commands), seg_size): 
                segment_commands = commands[i:i+seg_size]
                seg_stats = get_stats(segment_commands)
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
                'cmd_count': segment['cmd_count'],
                'commands': segment['commands'],
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
            commands = get_commands(filename)
            for i in range(0, len(commands), seg_size): 
                segment_commands = commands[i:i+seg_size]
                seg_stats = get_stats(segment_commands)
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
        write_cmd_count_to_csv(user['cmd_count'], os.path.abspath(os.path.join('MFDCA-DATA','CmdCount',username)))
        write_vocabulary_to_csv(user['vocabulary'], os.path.abspath(os.path.join('MFDCA-DATA','Vocabulary',username)))
        for i,seg_stats in enumerate(user['segments']):
            write_cmd_count_to_csv(seg_stats['cmd_count'], os.path.abspath(os.path.join('MFDCA-DATA','CmdCount',username+'_Seg'+str(i))))
            write_vocabulary_to_csv(seg_stats['vocabulary'], os.path.abspath(os.path.join('MFDCA-DATA','Vocabulary',username+'_Seg'+str(i))))

    commands = get_commands(filenames)
    total_stats = get_stats(commands)
    write_cmd_count_to_csv(total_stats['cmd_count'], os.path.abspath(os.path.join('MFDCA-DATA','CmdCount','TotalCmdCount')))
    write_vocabulary_to_csv(total_stats['vocabulary'], os.path.abspath(os.path.join('MFDCA-DATA','Vocabulary','TotalVocabulary')))


    # print('@@cmd Count:')
    # for key, val in sorted_wc:
    #     print(key+': '+str(val))
    # print('\n@@ Total numbr of commands: '+str(total_stats['total_wc']))
    # print('\n@@ Commands per user: '+str(total_stats['total_wc']/40))
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
    #         'segment_number': [float(segment['segment_number']) for segment in segments[0:1500]],
    #         'commands': [segment['commands'] for segment in segments[0:1500]]
    #     },
    #     'labels': [float(segment['label']) for segment in segments[0:1500]],
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