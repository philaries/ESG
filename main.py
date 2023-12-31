import os.path

import numpy as np
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pathlib import Path

# load stop words
stop_words = set()
with open("stop_words.txt", 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.replace("\n", "")
        stop_words.add(line)

def Chinese_words_segmentation(mytexts: str, _stop_words: set()):
    '''
    中文分词
    :param mytext:
    :return:
    '''
    import jieba
    jieba.initialize()
    mytexts_seged_with_stop_words = jieba.cut(mytexts, cut_all=True)
    mytexts_seged_without_stop_words = []
    for phrase in mytexts_seged_with_stop_words:
        # remove the useless ones
        if phrase not in _stop_words and len(phrase) > 1 and (not str.isdigit(phrase) ) and str.isprintable(phrase):
            mytexts_seged_without_stop_words.append(phrase)
    return " ".join(mytexts_seged_without_stop_words)


def extract_topic_words_from_text(_mytext: str):

    global stop_words
    text_seged = Chinese_words_segmentation(_mytext, stop_words)

    n_features = 1000 # 主题词数目
    tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                    max_features=n_features,
                                    stop_words='english')

    tf = tf_vectorizer.fit_transform([text_seged])
    n_top_words = 100
    n_topics = 2  # ESG relevant and others
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=100,
                                    learning_method='batch',
                                    random_state=100)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names_out()


    for topic_index, topic in enumerate(lda.components_):
        return [tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]


def get_KL(topic_words1, topic_words2):
    '''
    return the entropy of two set of topic words based on the ESG_words
    :param topic_words1:
    :param topic_words2:
    :return:
    '''
    import scipy
    ESG_words = set()

    # load the ESG words
    with open("ESG_words.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            ESG_words.add(line)
    counts1 = [0] * 2
    counts2 = [0] * 2
    for topic_word1 in topic_words1:
        if topic_word1 in ESG_words:
            counts1[0] += 1
        else:
            counts1[1] += 1

    for topic_word2 in topic_words2:
        if topic_word2 in ESG_words:
            counts2[0] += 1
        else:
            counts2[1] += 1
    counts1 = counts1 / np.sum(counts1)
    counts2 = counts2 / np.sum(counts2)

    return scipy.stats.entropy(counts1, counts2)

def get_my_text(dir_path, text_file_name):
    mytext = ""
    # load text materials
    if not os.path.exists(f"{dir_path}/{text_file_name}.txt"):
        with pdfplumber.open(f"{dir_path}/{text_file_name}.pdf") as pdf:
            for page in pdf.pages:
                text = page.extract_text()  # 提取文本
                txt_file = open(f"{dir_path}/{text_file_name}.txt", mode='a', encoding='utf-8')
                txt_file.write(text)
                mytext += text
    else:
        with open(f'{dir_path}/{text_file_name}.txt') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace("\n", "")
                mytext += line
    return mytext


def extract_topic_words(dir_path, text_file_name, output_dir="topic_words"):
    # load from the cache
    if os.path.exists(f"{output_dir}/{text_file_name}.txt"):
        with open(f"{output_dir}/{text_file_name}.txt") as f:
            lines = f.readlines()
            topic_words = lines[0].split()
            return topic_words

    mytext = get_my_text(dir_path, text_file_name)
    # load text materials
    if not os.path.exists(f"{dir_path}/{text_file_name}.txt"):
        with pdfplumber.open(f"{dir_path}/{text_file_name}.pdf") as pdf:
            for page in pdf.pages:
                text = page.extract_text()  # 提取文本
                txt_file = open(f"{dir_path}/{text_file_name}.txt", mode='a', encoding='utf-8')
                txt_file.write(text)
                mytext += text
    else:
        with open(f'{dir_path}/{text_file_name}.txt') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace("\n", "")
                mytext += line

    # extract the topic words
    topic_words = extract_topic_words_from_text(mytext)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(f"{output_dir}/{text_file_name}.txt", 'w') as f:
        for topic_word in topic_words:
            f.write(topic_word + " ")

    return topic_words


def get_topic_words_frequency(dir_path, text_file_name, topic_words):
    '''
    return the frequency of the topic words in the text
    :param dir_path:
    :param text_file_name:
    :param topic_words:
    :return:
    '''
    mytext = get_my_text(dir_path, text_file_name)
    frequencies = {}
    for topic_word in topic_words:
        frequencies[topic_word] = mytext.count(topic_word)
    return frequencies

def output_frequency():
    ref_path = "ESG reports"
    ref_file_name1 = "2023阿里巴巴ESG报告"
    ref_file_name2 = "2022腾讯ESG报告"

    frequency1 = get_topic_words_frequency(ref_path, ref_file_name1, extract_topic_words(ref_path, ref_file_name1))
    frequency2 = get_topic_words_frequency(ref_path, ref_file_name2, extract_topic_words(ref_path, ref_file_name2))
    frequency = {}
    for key in frequency1.keys():
        if key in frequency2.keys():
            frequency[key] = frequency1[key] + frequency2[key]
        else:
            frequency[key] = frequency1[key]
    for key in frequency2.keys():
        if key not in frequency.keys():
            frequency[key] = frequency2[key]
    frequency = sorted(frequency.items(), key=lambda item: item[1], reverse=True)
    # save the frequency
    with open("frequency.txt", 'w') as f:
        for key in frequency:
            f.write(f'{key[0]} {key[1]} \n')

    new_Text = ""
    for key in frequency:
        for count in range(key[1]):
            new_Text += key[0] + " "
    # word cloud to show the frequency
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    wc = WordCloud(
        font_path="/System/Library/Fonts/PingFang.ttc",  # macos: # windows: C:\Windows\Fonts\msyh.ttc
        background_color='white',  # 设置背景颜色
        width=1000,  # 设置宽度
        height=800,  # 设置高度
        margin=2,  # 设置边缘
        collocations=False
    ).generate(new_Text)
    plt.imshow(wc)
    plt.axis("off")
    plt.savefig("frequency.png")
    plt.show()

if __name__ == '__main__':


    ref_path = "ESG reports"
    ref_file_name1 = "2023阿里巴巴ESG报告"
    ref_file_name2 = "2022腾讯ESG报告"
    material_path = "materials"


    # all materials ready to be analyzed
    material_names = set()
    for material_name in os.listdir(material_path):
        material_names.add(Path(material_name).stem)

    ref_topic_words = extract_topic_words(ref_path, ref_file_name1)
    # ref_topic_words = extract_topic_words(ref_path, ref_file_name2)

    failed_material_names = []
    KLs = []
    # # load the KL before
    f = open(f"{ref_file_name1}_KL.txt", 'w')
    for material_name in material_names:
        print(f"processing {material_name}")
        try:
            material_topic_words = extract_topic_words(material_path, material_name)
            KL = get_KL(material_topic_words, ref_topic_words)
            KLs.append([material_name.split('_')[0], KL])
        except:
            failed_material_names.append(material_name)
            continue
    KLs.sort(key=lambda KL : KL[1])
    for KL in KLs:
        f.write(f'{KL[0]} {KL[1]} \n')

    f.close()
    print(failed_material_names)