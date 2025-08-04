import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from pyvis.network import Network
import community as community_louvain
from collections import defaultdict
import os

# 配置设置
MODEL_PATHS_GEMMA = {
    "embedding": "./result_/models_/Gemma_Bertopic_model_local_full_text_0",
    "copilot": "./result_/models_/Gemma_Bertopic_model_local_full_text_1",
    "agent": "./result_/models_/Gemma_Bertopic_model_local_full_text_2"
}
MODEL_PATHS_DPR = {
    "embedding": "./result_/models_/DPR_Bertopic_model_local_full_text_0",
    "copilot": "./result_/models_/DPR_Bertopic_model_local_full_text_1",
    "agent": "./result_/models_/DPR_Bertopic_model_local_full_text_2"
}

MODEL_PATHS= {
    "embedding": "./result_/models_/Bertopic_model_local_full_text_0",
    "copilot": "./result_/models_/Bertopic_model_local_full_text_1",
    "agent": "./result_/models_/Bertopic_model_local_full_text_2"
}
OUTPUT_DIR = "topic_network_analysis"
THRESHOLD = 0.1  # 主题相似度阈值
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_models():
    """加载保存的BERTopic模型"""
    models = {}
    for mode, path in MODEL_PATHS.items():
        models[mode] = BERTopic.load(path)
        print(f"Loaded {mode} model with {len(models[mode].get_topic_info())} topics")
    return models


def extract_topic_data(models):
    """提取主题信息"""
    topic_data = {}
    for mode, model in models.items():
        # 获取主题信息
        topic_info = model.get_topic_info()
        topic_embeddings = model.topic_embeddings_

        # 获取主题关键词
        topics = {}
        for topic_id in topic_info['Topic'].unique():
            if topic_id != -1:  # 排除噪声主题
                keywords = [word for word, _ in model.get_topic(topic_id)]
                topics[topic_id] = {
                    "keywords": keywords,
                    "size": topic_info[topic_info['Topic'] == topic_id]['Count'].values[0],
                    "embedding": topic_embeddings[topic_id]
                }

        topic_data[mode] = {
            "topics": topics,
            "embeddings": topic_embeddings,
            "topic_info": topic_info
        }
    return topic_data


def build_network(topic_data):
    """构建主题网络"""
    # 创建完整网络图
    full_network = nx.Graph()
    mode_networks = {}

    # 首先构建每个模式内部的网络
    for mode, data in topic_data.items():
        G = nx.Graph()
        topics = data["topics"]
        embeddings = data["embeddings"]

        # 添加节点
        for topic_id, topic_info in topics.items():
            G.add_node(f"{mode}_{topic_id}",
                       label=f"{mode.capitalize()} T-{topic_id}",
                       keywords=", ".join(topic_info["keywords"][:15]),
                       size=topic_info["size"],
                       group=mode)

        # 添加边（基于主题相似性）
        topic_ids = list(topics.keys())
        similarity_matrix = cosine_similarity(embeddings[topic_ids])

        for i, topic_id_i in enumerate(topic_ids):
            for j, topic_id_j in enumerate(topic_ids[i + 1:], start=i + 1):
                sim_score = similarity_matrix[i][j]
                if sim_score > THRESHOLD:
                    G.add_edge(f"{mode}_{topic_id_i}",
                               f"{mode}_{topic_id_j}",
                               weight=sim_score,
                               title=f"Similarity: {sim_score:.2f}")

        mode_networks[mode] = G
        full_network = nx.compose(full_network, G)

    # 添加跨模式连接
    all_modes = list(topic_data.keys())
    for i, mode1 in enumerate(all_modes):
        for mode2 in all_modes[i + 1:]:
            topics1 = topic_data[mode1]["topics"]
            topics2 = topic_data[mode2]["topics"]
            embeddings1 = topic_data[mode1]["embeddings"]
            embeddings2 = topic_data[mode2]["embeddings"]

            for topic_id1, topic_info1 in topics1.items():
                for topic_id2, topic_info2 in topics2.items():
                    # 计算跨模式主题相似度
                    emb1 = embeddings1[topic_id1].reshape(1, -1)
                    emb2 = embeddings2[topic_id2].reshape(1, -1)
                    sim_score = cosine_similarity(emb1, emb2)[0][0]

                    if sim_score > THRESHOLD:
                        full_network.add_edge(
                            f"{mode1}_{topic_id1}",
                            f"{mode2}_{topic_id2}",
                            weight=sim_score,
                            title=f"Cross-Mode Similarity: {sim_score:.2f}",
                            color="purple",
                            dashes=True
                        )

    return full_network, mode_networks


def analyze_network(network):
    """分析网络结构"""
    # 计算中心性指标
    degree_centrality = nx.degree_centrality(network)
    betweenness_centrality = nx.betweenness_centrality(network)
    closeness_centrality = nx.closeness_centrality(network)

    # 识别枢纽主题
    hub_topics = sorted(
        [(node, centrality) for node, centrality in betweenness_centrality.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]

    # 社区检测
    partition = community_louvain.best_partition(network)

    # 分析社区特征
    community_keywords = defaultdict(list)
    for node, comm_id in partition.items():
        keywords = network.nodes[node].get('keywords', '').split(", ")
        community_keywords[comm_id].extend(keywords)

    # 提取每个社区的关键词
    top_community_keywords = {}
    for comm_id, keywords in community_keywords.items():
        vectorizer = CountVectorizer()
        word_matrix = vectorizer.fit_transform([" ".join(keywords)])
        word_counts = zip(vectorizer.get_feature_names_out(), np.array(word_matrix.sum(axis=0))[0])
        top_words = sorted(word_counts, key=lambda x: x[1], reverse=True)[:10]
        top_community_keywords[comm_id] = [word for word, count in top_words]

    return {
        "degree_centrality": degree_centrality,
        "betweenness_centrality": betweenness_centrality,
        "closeness_centrality": closeness_centrality,
        "hub_topics": hub_topics,
        "partition": partition,
        "community_keywords": top_community_keywords
    }


def visualize_network(network, analysis_results, mode="full"):
    """可视化网络"""
    # 设置节点属性
    partition = analysis_results["partition"]
    betweenness = analysis_results["betweenness_centrality"]

    # 计算节点大小比例
    max_size = max([data['size'] for node, data in network.nodes(data=True)])
    min_size = min([data['size'] for node, data in network.nodes(data=True)])

    for node, data in network.nodes(data=True):
        # 基于社区设置颜色
        data["group"] = partition.get(node, 0)

        # 基于中心性设置大小
        centrality = betweenness.get(node, 0)
        size_scale = 10 + 90 * centrality  # 大小基于中介中心性
        data["size"] = size_scale

        # 添加标题
        data["title"] = f"""
        <b>{data['label']}</b><br>
        Keywords: {data['keywords']}<br>
        Degree Centrality: {analysis_results['degree_centrality'].get(node, 0):.3f}<br>
        Betweenness Centrality: {analysis_results['betweenness_centrality'].get(node, 0):.3f}<br>
        Closeness Centrality: {analysis_results['closeness_centrality'].get(node, 0):.3f}
        """

    # 创建pyvis网络
    nt = Network(height="750px", width="100%", bgcolor="#dddddd", font_color="black")

    # 从networkx导入数据
    nt.from_nx(network)

    # 配置可视化选项
    nt.set_options("""
    var options = {
      "nodes": {
        "borderWidth": 1,
        "borderWidthSelected": 2,
        "font": {
          "size": 16,
          "face": "arial"
        }
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": {
          "type": "continuous"
        }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -5000,
          "centralGravity": 0.1,
          "springLength": 150,
          "springConstant": 0.01,
          "damping": 0.09
        },
        "minVelocity": 0.75
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200
      }
    }
    """)

    # 保存可视化
    output_path = os.path.join(OUTPUT_DIR, f"Gemma_{mode}_topic_network.html")
    # nt.show(f"{mode}_topic_network")
    nt.save_graph(output_path)
    print(f"Saved {mode} network visualization to {output_path}")

    return output_path


def generate_community_report(analysis_results):
    """生成社区分析报告"""
    report = "# 主题社区分析报告\n\n"

    # 社区关键词分析
    report += "## 社区关键词特征\n"
    for comm_id, keywords in analysis_results["community_keywords"].items():
        report += f"### 社区 {comm_id}\n"
        report += f"**核心关键词**: {', '.join(keywords[:5])}\n\n"

    # 枢纽主题分析
    report += "## 关键枢纽主题\n"
    report += "| 主题 | 中介中心性 | 连接社区 |\n"
    report += "|------|------------|----------|\n"
    for node, centrality in analysis_results["hub_topics"]:
        comm_id = analysis_results["partition"].get(node, -1)
        report += f"| {node} | {centrality:.25f} | {comm_id} |\n"

    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, "Gemma_community_analysis_report.md")
    with open(report_path, "w", encoding='utf8') as f:
        f.write(report)
    print(f"Saved community analysis report to {report_path}")

    return report_path


def plot_community_keywords(analysis_results):
    """绘制社区关键词词云"""
    from wordcloud import WordCloud

    for comm_id, keywords in analysis_results["community_keywords"].items():
        text = " ".join(keywords)
        wordcloud = WordCloud(width=800, height=400,
                              background_color='white').generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"Community {comm_id} Keywords", fontsize=15)
        plt.axis("off")

        # 保存图片
        img_path = os.path.join(OUTPUT_DIR, f"community_{comm_id}_keywords.png")
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()
        print(f"Saved word cloud for community {comm_id} to {img_path}")


def main():
    # 1. 加载模型
    print("Loading BERTopic models...")
    models = load_models()

    # 2. 提取主题数据
    print("\nExtracting topic data...")
    topic_data = extract_topic_data(models)

    # 3. 构建主题网络
    print("\nBuilding topic networks...")
    full_network, mode_networks = build_network(topic_data)

    # 4. 分析完整网络
    print("\nAnalyzing full network...")
    full_analysis = analyze_network(full_network)

    # 5. 可视化完整网络
    print("\nVisualizing full network...")
    full_viz_path = visualize_network(full_network, full_analysis, "full")

    # 6. 生成分析报告
    print("\nGenerating analysis report...")
    report_path = generate_community_report(full_analysis)

    # 7. 绘制关键词词云
    print("\nCreating keyword visualizations...")
    plot_community_keywords(full_analysis)

    # 8. 分析各模式内部网络
    print("\nAnalyzing mode-specific networks...")
    mode_analysis = {}
    for mode, network in mode_networks.items():
        print(f"  Analyzing {mode} network...")
        analysis = analyze_network(network)
        mode_analysis[mode] = analysis

        # 可视化模式内部网络
        viz_path = visualize_network(network, analysis, mode)

    print("\nAnalysis completed successfully!")
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()