import csv
import ollama
import pandas as pd
from openpyxl.styles.builtins import total
from tqdm import tqdm
from transformers.trainer_utils import total_processes_number

client = ollama.Client(host='http://localhost:11434')
def query_ollama(prompt, model='llama3.2:latest', **options):
    """
    查询Ollama模型的函数
    模型选项
    gemma3:4b
    deepseek-r1:7b

    参数:
        prompt: 用户输入的提示
        model: 要使用的模型名称
        options: 推理参数，如temperature, top_p等

    返回:
        模型的完整响应
    """
    # 设置默认选项
    default_options = {
        'max_new_tokens':512,
        'temperature': 0.6,
        'top_p': 0.9,
        # 'num_predict': 256,
    }
    # 更新用户提供的选项
    default_options.update(options)

    response = client.chat(  #ollama
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        options=default_options
    )

    return response['message']['content']

def summary_abs():

    prompt_1 = ('Extract the central argument from the paper abstract, ensuring the distilled statement precisely encapsulates the essence of the abstract and is succinctly comprehensive, thereby facilitating its efficacious deployment in topic clustering analysis. You only need to provide the extracted central argument and nothing else.'
               'Paper Abstract:')
    prompt_2 = 'Central Argument:'
    abs_file_list = ['./processed_data/emb_dedup_.csv', './processed_data/cop_dedup_.csv',
                             './processed_data/agn_dedup_.csv']
    extract_argument_result = ['./processed_data/llama3_2_ext_emb.csv', './processed_data/llama3_2_ext_cop.csv',
                               './processed_data/llama3_2_ext_agn.csv']
    extract_argument_result_gemma = ['./processed_data/gemma3_4b_ext_emb.csv', './processed_data/gemma3_4b_ext_cop.csv',
                               './processed_data/gemma3_4b_ext_agn.csv']
    extract_argument_result_dpr = ['./processed_data/deepseek-r1_7b_ext_emb.csv', './processed_data/deepseek-r1_7b_ext_cop.csv',
                 './processed_data/deepseek-r1_7b_ext_agn.csv']

    for i in range(3):
        # 第一版是使用lamma3;第3版使用deep seekR1；第2版使用gemma3:4b
        data_set = pd.read_csv(abs_file_list[i], encoding='utf8')
        for items in tqdm(data_set['3_in_1'], desc="Processing", bar_format="{l_bar}{bar:20}{r_bar}"):
            total_prompt = prompt_1 + str(items) + prompt_2
            ext_result  = query_ollama(total_prompt,model='deepseek-r1:7b', temperature=0.6, num_predict=500)

            with open(extract_argument_result_dpr[i], 'a', newline='', encoding='utf8') as f:
                writer = csv.writer(f)
                writer.writerow([ext_result])
            pass
    pass

def summary_topics(topic_docs_list, key_words_list, model='llama3.2:latest'):
    prompt_1 = 'Please provide a concise topic label that accurately encapsulates its core content for the following topic, which encompasses the following documents:\n'
    prompt_2 = 'The Key words central to this topic are:\n'
    prompt_3 = ('The length of the provided topic label should between 6-10 words, and must supply ONLY the topic label, '
                'excluding any additional content.\n Topic label:')
    total_prompt = prompt_1 + topic_docs_list + prompt_2 + key_words_list + prompt_3

    summary_topics_result = query_ollama(total_prompt, mdoel=model ,temperature=0.6, num_predict=500)

    return summary_topics_result



# 使用示例
if __name__ == "__main__":
    # prompt = ('Extract the central argument from the paper abstract, '
    #           'ensuring the distilled statement precisely encapsulates the essence of the abstract and is succinctly comprehensive, '
    #           'thereby facilitating its efficacious deployment in topic clustering analysis. You only need to provide the extracted central argument and nothing else.'
    #           'Paper Abstract:AI-Assisted Programming Tasks Using Code Embeddings and Transformers AI-assisted programming; code embeddings; transformers This review article provides an in-depth analysis of the growing field of AI-assisted programming tasks, specifically focusing on the use of code embeddings and transformers. With the increasing complexity and scale of software development, traditional programming methods are becoming more time-consuming and error-prone. As a result, researchers have turned to the application of artificial intelligence to assist with various programming tasks, including code completion, bug detection, and code summarization. The utilization of artificial intelligence for programming tasks has garnered significant attention in recent times, with numerous approaches adopting code embeddings or transformer technologies as their foundation. While these technologies are popular in this field today, a rigorous discussion, analysis, and comparison of their abilities to cover AI-assisted programming tasks is still lacking. This article discusses the role of code embeddings and transformers in enhancing the performance of AI-assisted programming tasks, highlighting their capabilities, limitations, and future potential in an attempt to outline a future roadmap for these specific technologies.'
    #           'Central Argument:')
    # result = query_ollama(prompt, temperature=0.5, num_predict=500)
    # print("模型回复:")
    # print(result)
    # with open('./1_.csv', 'a', newline='', encoding='utf8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([result])

    summary_abs()

