import os
import pandas as pd


def merge_raw_data():
    '''
    合并原始数据，抽取
    :return:
    '''

    raw_data_dir_path_4_emb = './raw_data/Embedding_'
    raw_data_dir_path_4_cop = './raw_data/Copilot_'
    raw_data_dir_path_4_agn = './raw_data/Agent_'

    output_data_dir_path_4_emb = './processed_data/emb_processed_data.csv'
    output_data_dir_path_4_cop = './processed_data/copilot_processed_data.csv'
    output_data_dir_path_4_agn = './processed_data/agent_processed_data.csv'

    extract_columns = ['Article Title', 'Author Keywords', 'Abstract', 'Publication Year']

    merged_data_emb, merged_data_cop, merged_data_agn = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for file_emb in os.listdir(raw_data_dir_path_4_emb):
        file_path = os.path.join(raw_data_dir_path_4_emb, file_emb)
        try:
            data_emb = pd.read_excel(file_path, usecols=extract_columns, sheet_name=0, header=0)
            data_emb['resource_data'] = file_emb
            merged_data_emb = pd.concat([merged_data_emb, data_emb])
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    for file_cop in os.listdir(raw_data_dir_path_4_cop):
        file_path = os.path.join(raw_data_dir_path_4_cop, file_cop)
        try:
            data_cop = pd.read_excel(file_path, usecols=extract_columns, sheet_name=0, header=0)
            data_cop['resource_data'] = file_cop
            merged_data_cop = pd.concat([merged_data_cop, data_cop])
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    for file_agn in os.listdir(raw_data_dir_path_4_agn):
        file_path = os.path.join(raw_data_dir_path_4_agn, file_agn)
        try:
            data_agn = pd.read_excel(file_path, usecols=extract_columns, sheet_name=0, header=0)
            data_agn['resource_data'] = file_agn
            merged_data_agn = pd.concat([merged_data_agn, data_agn])
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    merged_data_emb.to_csv(output_data_dir_path_4_emb, encoding='utf-8')
    merged_data_cop.to_csv(output_data_dir_path_4_cop, encoding='utf-8')
    merged_data_agn.to_csv(output_data_dir_path_4_agn, encoding='utf-8')

def process_merged_data():
    # merged_data_emb_path = './processed_data/emb_processed_data.csv'
    # merged_data_cop_path = './processed_data/copilot_processed_data.csv'
    # merged_data_agn_path = './processed_data/agent_processed_data.csv'

    mered_data_path_list = ['./processed_data/emb_processed_data.csv', './processed_data/copilot_processed_data.csv',
                            './processed_data/agent_processed_data.csv']
    output_data_path_list = ['./processed_data/emb_dedup_.csv', './processed_data/cop_dedup_.csv',
                             './processed_data/agn_dedup_.csv']
    for i in range(3):
        merged_data = pd.read_csv(mered_data_path_list[i], encoding='utf-8')
        merged_data['Article Title'] = merged_data['Article Title'].astype(str)
        merged_data['Author Keywords'] = merged_data['Author Keywords'].astype(str)
        merged_data['Abstract'] = merged_data['Abstract'].astype(str)

        print(f"Raw data have{len(merged_data)} items")
        deduplicated_results = merged_data.drop_duplicates(subset=['Article Title', 'Author Keywords', 'Abstract'],
                                                           keep='first')
        print(f"the deduplicated data have {len(deduplicated_results)} items")

        merged_dedup_data = pd.DataFrame()
        merged_dedup_data['pub_years'] = deduplicated_results['Publication Year']
        merged_dedup_data['3_in_1'] = deduplicated_results['Article Title'] + ' ' + deduplicated_results[
            'Author Keywords'] + ' ' + deduplicated_results['Abstract']
        merged_dedup_data.to_csv(output_data_path_list[i], encoding='utf8')
    pass


import csv
import re

def remove_think_content(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 编译正则表达式模式（支持多行匹配）
        pattern = re.compile(r'<think>.*?</think>', re.DOTALL)

        for row in reader:
            # 假设内容在第一个字段（可根据需要调整索引）
            cleaned = pattern.sub('', row[0]).strip()
            writer.writerow([cleaned])



if __name__ == '__main__':
    input_file_list = ['./processed_data/deepseek-r1_7b_ext_emb.csv', './processed_data/deepseek-r1_7b_ext_cop.csv',
                 './processed_data/deepseek-r1_7b_ext_agn.csv']
    output_file_list = ['./processed_data/p_deepseek-r1_7b_ext_emb.csv', './processed_data/p_deepseek-r1_7b_ext_cop.csv',
                 './processed_data/p_deepseek-r1_7b_ext_agn.csv']

    for i in range(len(input_file_list)):
        remove_think_content(input_file_list[i], output_file_list[i])
        pass


    # merge_raw_data()
    # process_merged_data()