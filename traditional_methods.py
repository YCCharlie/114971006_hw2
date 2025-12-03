# traditional_methods.py - Part A 的所有實作

import math
import jieba
from collections import Counter
import numpy as np
# 這裡需要將 TCSP 和 read_stopwords_list 替換為實際的停用詞載入邏輯
# 假設您將停用詞列表作為參數傳入，或者在主程式中初始化

# --- A-1: TF-IDF 核心函數 ---
def calculate_tf(word_dict, total_words):
    """計算詞頻 (Term Frequency)"""
    # [貼上您的 calculate_tf 函數]
    pass

def calculate_idf(documents, word):
    """計算逆文件頻率 (Inverse Document Frequency)"""
    # [貼上您的 calculate_idf 函數]
    pass

def calculate_tfidf_vector(doc_tokens, all_docs):
    """計算單個文檔的 TF-IDF 向量"""
    # [貼上您的 calculate_tfidf_vector 函數]
    pass

def manual_cosine_similarity(vec1, vec2):
    """計算兩個向量的餘弦相似度"""
    # [貼上您的 manual_cosine_similarity 函數]
    pass

# --- A-2: 規則分類器 ---
class RuleBasedSentimentClassifier:
    def __init__(self, stop_words):
        # [貼上您的 __init__ 內容]
        self.stop_words = stop_words # 必須接收停用詞
        pass

    def classify(self, text):
        # [貼上您的 classify 內容]
        pass

class TopicClassifier:
    def __init__(self):
        # [貼上您的 __init__ 內容]
        pass

    def classify(self, text):
        # [貼上您的 classify 內容]
        pass

# --- A-3: 統計式摘要 ---
class StatisticalSummarizer:
    def __init__(self, stop_words):
        # [貼上您的 __init__ 內容]
        self.stop_words = stop_words

    def sentence_score(self, sentence, word_freq, sentence_index, total_sentences):
        # [貼上您的 sentence_score 內容]
        pass

    def summarize(self, text, ratio=0.3):
        # [貼上您的 summarize 內容]
        pass
