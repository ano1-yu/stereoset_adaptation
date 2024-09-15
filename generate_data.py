import pandas as pd
import re
import os
import json
from tqdm import tqdm

CURRENT_DIR = os.getcwd()

def safe_replace(text, old, new):
    """
    Safely replaces words in the text by ensuring that only whole words are replaced.
    
    Parameters:
    - text: The string where replacements are to be made.
    - old: The word to be replaced.
    - new: The word to replace with.
    
    Returns:
    - Modified string with replacements made.
    """
    pattern = r'\b' + re.escape(old) + r'\b'
    return re.sub(pattern, new, text)

def read_pairs(demo):
    """
    Reads replacement word pairs from a specified file based on the demographic.
    
    Parameters:
    - demo: The demographic group to read replacement pairs for (e.g., 'gender', 'race').
    
    Returns:
    - A tuple of word pairs to be used for replacements.
    """
    pairs = []
    with open(CURRENT_DIR + '/replace_words/replace_words_' + demo + '.txt', 'r', encoding='utf-8') as file:
        for line in file:
            items = line.strip().split(', ')
            if len(items) == 2:
                pairs.append((items[0], items[1]))
    return tuple(pairs)

def resturcture_intersentence_data(demographic, df):
    """
    Restructures intersentence data by replacing target words based on demographic-specific pairs.
    
    Parameters:
    - demographic: The specific demographic for which the data is being processed.
    - df: DataFrame containing the original intersentence data.
    
    Returns:
    - A new DataFrame with the replaced and restructured data.
    """
    # Pre-process the dataset
    df['Context'] = df['Context'].str.lower()
    df['Stereotype Sentence'] = df['Stereotype Sentence'].str.lower()

    pairs = read_pairs(demographic)

    new_df = pd.DataFrame(columns=['initial_demo', 'replaced_demo', 'context', 'replaced_context', 'sentence', 'replaced_sentence'])
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f'Restructuring intersentence data for {demographic}'):
        initial_demo = []
        replaced_demo = []
        c = row['Context']
        s = row['Stereotype Sentence']
        target = row['Target'].lower()
        new_df.at[idx, 'context'] = c
        new_df.at[idx, 'sentence'] = s
        for p in pairs:
            if p[0] == target:
                c = c.replace(*p)
            if p[1] in c and p[0] in row['Context'] and p[0] == target:
                initial_demo.append(p[0])
                replaced_demo.append(p[1])
            new_df.at[idx, 'replaced_context'] = c

        if demographic == 'gender':
            male_pronoun_pairs = (('he', 'she'), ('his', 'her'), ('himself', 'herself'))
            female_pronoun_pairs = (('she', 'he'), ('her', 'his'), ('herself', 'himself'))
            if target in ('gentlemen', 'grandfather', 'himself', 'male', 'schoolboy'):
                for p in male_pronoun_pairs:
                    s = safe_replace(s, *p)
                    c = safe_replace(c, *p)
            else:
                for p in female_pronoun_pairs:
                    s = safe_replace(s, *p)
                    c = safe_replace(c, *p)
            new_df.at[idx, 'replaced_context'] = c
            new_df.at[idx, 'replaced_sentence'] = s
        else:
            new_df.at[idx, 'replaced_sentence'] = s
        new_df.at[idx, 'initial_demo'] = initial_demo
        new_df.at[idx, 'replaced_demo'] = replaced_demo

    return new_df

def resturcture_intrasentence_data(demographic, df):
    df['Stereotype Sentence'] = df['Stereotype Sentence'].str.lower()
    pairs = read_pairs(demographic)
    new_df = pd.DataFrame(columns=['initial_demo', 'replaced_demo', 'sentence', 'replaced_sentence'])

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f'Restructuring intrasentence data for {demographic}'):
        initial_demo = []
        replaced_demo = []
        s = row['Stereotype Sentence']
        target = row['Target'].lower()
        new_df.at[idx, 'sentence'] = s
        for p in pairs:
            if p[0] == target:
                s = s.replace(*p)
            if p[1] in s and p[0] in row['Stereotype Sentence'] and p[0] == target:
                initial_demo.append(p[0])
                replaced_demo.append(p[1])
            new_df.at[idx, 'replaced_sentence'] = s
        new_df.at[idx, 'initial_demo'] = initial_demo
        new_df.at[idx, 'replaced_demo'] = replaced_demo

    return new_df

if __name__ == '__main__':
    json_data = {
        'version': '1.0-dev',
        'data':{
            'intersentence':{
                'race': {},
                'religion': {},
                'gender': {},
                'profession': {
                    'profession1':{},
                    'profession2':{},
                    'profession3':{},
                    'profession4':{},
                    'profession5':{}
                }
            },
            'intrasentence':{
                'race': {},
                'religion': {},
                'gender': {},
                'profession': {
                    'profession1':{},
                    'profession2':{},
                    'profession3':{},
                    'profession4':{},
                    'profession5':{}
                }
            }
        }
    }

    demos = ['race', 'religion', 'gender', 'profession']

    # Process intersentence and intrasentence data for each demographic
    for demo in demos:
        if demo == 'profession':
            for i in tqdm(range(1, 6), desc=f'Processing profession data'):
                intersentence_data_path = CURRENT_DIR + f'/data/intersentence/intersentence_{demo}_data.csv'
                intrasentence_data_path = CURRENT_DIR + f'/data/intrasentence/intrasentence_{demo}_data.csv'
                intersentence_df = pd.read_csv(intersentence_data_path)
                intrasentence_df = pd.read_csv(intrasentence_data_path)
                new_intersentence_df = resturcture_intersentence_data(f"{demo}{i}", intersentence_df)
                new_intrasentence_df = resturcture_intrasentence_data(f"{demo}{i}", intrasentence_df)
                new_intersentence_df.to_csv(CURRENT_DIR + f'/new_intersentence_{demo}{i}.csv', index=False)
                new_intrasentence_df.to_csv(CURRENT_DIR + f'/new_intrasentence_{demo}{i}.csv', index=False)
        else:
            for data_type in ['intersentence', 'intrasentence']:
                data_path = CURRENT_DIR + f'/data/{data_type}/{data_type}_{demo}_data.csv'
                df = pd.read_csv(data_path)
                if data_type == 'intersentence':
                    new_df = resturcture_intersentence_data(demo, df)
                    new_df.to_csv(CURRENT_DIR + f'/new_intersentence_{demo}.csv', index=False)
                else:
                    new_df = resturcture_intrasentence_data(demo, df)
                    new_df.to_csv(CURRENT_DIR + f'/new_intrasentence_{demo}.csv', index=False)

    # Convert restructured CSV files to JSON format
    for demo in demos:
        if demo == 'profession':
            for i in tqdm(range(1, 6), desc=f'Converting profession CSV to JSON'):
                intersentence_data_path = CURRENT_DIR + f'/new_intersentence_{demo}{i}.csv'
                intrasentence_data_path = CURRENT_DIR + f'/new_intrasentence_{demo}{i}.csv'
                df = pd.read_csv(intersentence_data_path)
                os.remove(intersentence_data_path)  # Remove CSV after reading
                df_list = df.to_dict(orient='records')
                json_data['data']['intersentence'][demo][f'{demo}{i}'] = df_list

                df = pd.read_csv(intrasentence_data_path)
                os.remove(intrasentence_data_path)  # Remove CSV after reading
                df_list = df.to_dict(orient='records')
                json_data['data']['intrasentence'][demo][f'{demo}{i}'] = df_list
        else:
            for data_type in ['intersentence', 'intrasentence']:
                data_path = CURRENT_DIR + f'/new_{data_type}_{demo}.csv'
                df = pd.read_csv(data_path)
                os.remove(data_path)  # Remove CSV after reading
                df_list = df.to_dict(orient='records')
                json_data['data'][data_type][demo] = df_list

    # Save the final JSON file
    with open('new_data.json', 'w') as json_file:
        json.dump(json_data, json_file)
