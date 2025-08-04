import csv
import logging
import pandas as pd
import torch
import umap
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from extract_ import summary_topics

def bert_topic_():
    """
    BERTOPIC 模型的训练与保存
    :return:
    """
    data_list = ['./processed_data/llama3_2_ext_emb.csv', './processed_data/llama3_2_ext_cop.csv',
                               './processed_data/llama3_2_ext_agn.csv']
    data_list_gemma = ['./processed_data/gemma3_4b_ext_emb.csv', './processed_data/gemma3_4b_ext_cop.csv',
                               './processed_data/gemma3_4b_ext_agn.csv']
    data_list_dpr = ['./processed_data/deepseek-r1_7b_ext_emb.csv', './processed_data/deepseek-r1_7b_ext_cop.csv',
                 './processed_data/deepseek-r1_7b_ext_agn.csv']

    for i in range(3):
        data = pd.read_csv(data_list_dpr[i], encoding='utf-8')
        text = data['text'].tolist()
        print(f"{data_list_dpr[i]}数据总量:{len(text)}")
        logging.basicConfig(level=logging.INFO)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device: ", device)

        embedding_model = SentenceTransformer('./resource_/all-MiniLM-L6-v2', device=device)
        umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
        hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words='english', ngram_range=(1, 3))
        c_tf_idf_model = ClassTfidfTransformer()

        text_tqdm = tqdm(text, desc='Processing Documents')
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=c_tf_idf_model,
            min_topic_size=50,
            nr_topics='auto',
            top_n_words=15,
            n_gram_range=(1, 3),
            calculate_probabilities=True,
        )

        print("训练主题模型中...")
        topics, probs = topic_model.fit_transform(list(text_tqdm))

        # 获取文档信息
        document_info = topic_model.get_document_info(text)
        print(document_info)
        print(f"提取的主题: {topics}")

        # 保存模型
        topic_model.save('./result_/models_/DPR_Bertopic_model_local_full_text_'+str(i))
        print("模型已保存")

        # 打印主题频率
        topic_freq = topic_model.get_topic_freq()
        print("主题频率:", topic_freq)

        # 提取主题并保存到 CSV 文件
        topics = topic_model.get_topics()
        # topic_save_path = './result/BERTopics_extract.csv'
        topic_save_path = './result_/data_result_/DPR_'+str(i)+'BERTopics_extract_topics.csv'
        with open(topic_save_path, mode='w', encoding='utf8') as f:
            writer = csv.writer(f)
            writer.writerow(['Topic', 'Word', 'Weight'])
            for topic_num, words in topics.items():
                for word, weight in words:
                    writer.writerow([topic_num, word, weight])
        print(f"主题已保存至: {topic_save_path}")
        pass

    pass

def load_local_models_():
    # 这里要换路径
    model_emb = BERTopic.load('./result_/models_/DPR_Bertopic_model_local_full_text_0')
    model_cop = BERTopic.load('./result_/models_/DPR_Bertopic_model_local_full_text_1')
    model_agn = BERTopic.load('./result_/models_/DPR_Bertopic_model_local_full_text_2')
    models_list = [model_emb, model_cop, model_agn]
    models_list_name = ['model_emb', 'model_cop', 'model_agn']

    for model in models_list:
        representative_docs = model.get_representative_docs()
        print(representative_docs) #2,3,9
        for topic_id, docs in representative_docs.items():
            if topic_id == -1:
                continue
            print(docs)
            topic_words = [word for word, weight in model.get_topic(topic_id)]
            full_docs_ = ''
            full_words = ''
            for doc in docs:
                # 每个模型对应的每个主题对应的文档
                full_docs_ = str(doc) + '\n'
                pass
            for t_word in topic_words:
                full_words = str(t_word) + '\n'
                pass
            sum_top = summary_topics(full_docs_, full_words, model='deepseek-r1:7b')
            with open('./result_/data_result_/dpr_sum_topics_'+models_list_name[models_list.index(model)]+'_'+str(topic_id)+'_.csv', 'a', newline='', encoding='utf8') as f:
                writer = csv.writer(f)
                writer.writerow([topic_id, sum_top])
            pass
        pass
    pass

def basic_draw():
    # model_emb = BERTopic.load('./result_/models_/Bertopic_model_local_full_text_0')
    # model_cop = BERTopic.load('./result_/models_/Bertopic_model_local_full_text_1')
    # model_agn = BERTopic.load('./result_/models_/Bertopic_model_local_full_text_2')

    # model_emb = BERTopic.load('./result_/models_/Gemma_Bertopic_model_local_full_text_0')
    # model_cop = BERTopic.load('./result_/models_/Gemma_Bertopic_model_local_full_text_1')
    # model_agn = BERTopic.load('./result_/models_/Gemma_Bertopic_model_local_full_text_2')

    model_emb = BERTopic.load('./result_/models_/DPR_Bertopic_model_local_full_text_0')
    model_cop = BERTopic.load('./result_/models_/DPR_Bertopic_model_local_full_text_1')
    model_agn = BERTopic.load('./result_/models_/DPR_Bertopic_model_local_full_text_2')

    models_list = [model_emb, model_cop, model_agn]
    models_list_name = ['model_emb', 'model_cop', 'model_agn']

    for model in models_list:
        # 层次聚类
        topic_hier = model.visualize_hierarchy()
        topic_hier.write_html('./result_/pic_result_/dpr_topic_hiera_'+models_list_name[models_list.index(model)]+'.html')
        # 热力图
        topic_hot_map = model.visualize_heatmap()
        topic_hot_map.write_html('./result_/pic_result_/dpr_topic_heatmap_'+models_list_name[models_list.index(model)]+'.html')
        # 条形图
        topic_bar = model.visualize_barchart()
        topic_bar.write_html('./result_/pic_result_/dpr_topic_bar_'+models_list_name[models_list.index(model)]+'.html')
        # term_rank
        topic_term_rank = model.visualize_term_rank()
        topic_term_rank.write_html('./result_/pic_result_/dpr_topic_trem_rank_'+models_list_name[models_list.index(model)]+'.html')
        # # topics
        # topic_topics = model.visualize_topics()
        # topic_topics.write_html('./result_/pic_result_/topics_'+models_list_name[models_list.index(model)]+'.html')

        pass

    pass
# 使用
# representative_docs = get_custom_representative_docs(topic_model, docs=your_documents, n_docs=5)
if __name__ == '__main__':
    bert_topic_()
    load_local_models_()
    basic_draw()
    pass