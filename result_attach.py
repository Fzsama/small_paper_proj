import pandas as pd
from bertopic import BERTopic

def use_local_model():
    model_emb = BERTopic.load('./result_/models_/Bertopic_model_local_full_text_0')
    model_cop = BERTopic.load('./result_/models_/Bertopic_model_local_full_text_1')
    model_agn = BERTopic.load('./result_/models_/Bertopic_model_local_full_text_2')
    models_list = [model_emb, model_cop, model_agn]
    models_list_name = ['model_emb_', 'model_cop_', 'model_agn_']
    hci_model_list = ['embedding', 'copilot', 'agent']
    topic_doc_map = []
    for model in models_list:
        pointer = models_list.index(model)
        represent_docs = model.get_representative_docs()
        print(represent_docs)
        for topic_id, docs in represent_docs.items():
            if topic_id == -1:
                continue
            for doc in docs:
                doc_index = find_doc_index(doc, hci_model_list[pointer])
                full_doc = find_full_text(doc_index, hci_model_list[pointer])

                topic_doc_map.append({
                    "topic_id":topic_id,
                    "content":doc,
                    "doc_index":doc_index,
                    "full_doc":full_doc
                })
        df_topics = pd.DataFrame(topic_doc_map)
        topic_doc_map *= 0
        df_topics.to_csv('./result_/data_result_/topics_doc_llama_'+ models_list_name[pointer] + hci_model_list[pointer]+'_.csv')
    pass

def find_doc_index(doc, hci_model):
    doc_index = 0
    if hci_model == 'embedding':
        docs_core = pd.read_csv('./processed_data/llama3_2_ext_emb.csv')
        doc_index = docs_core.loc[docs_core['text'] == doc,:].index[0]
        pass
    elif hci_model == 'copilot':
        docs_core = pd.read_csv('./processed_data/llama3_2_ext_cop.csv')
        doc_index = docs_core.loc[docs_core['text'] == doc, :].index[0]
        pass
    else:
        docs_core = pd.read_csv('./processed_data/llama3_2_ext_agn.csv')
        doc_index = docs_core.loc[docs_core['text'] == doc, :].index[0]
        pass
    return doc_index

def find_full_text(doc_index, hci_model):
    if hci_model == 'embedding':
        full_doc = pd.read_csv('./processed_data/emb_dedup_.csv')
        return full_doc['3_in_1'][doc_index]
    elif hci_model == 'copilot':
        full_doc = pd.read_csv('./processed_data/cop_dedup_.csv')
        return full_doc['3_in_1'][doc_index]
    else:
        full_doc = pd.read_csv('./processed_data/agn_dedup_.csv')
        return full_doc['3_in_1'][doc_index]




if __name__ == '__main__':
    use_local_model()