import math
import jieba
import numpy as np
from collections import Counter

# --- 共用資料與設定 ---
# 手動定義停用詞，確保獨立執行時不依賴外部檔案
STOP_WORDS = set([
    '的','了','在','是','我','有','和','就','不','人','都','也','很','到','說','要',
    ',','，','。','.','！','!','？','?','：',':','「','」','（','）','(',' )'
])

# [cite_start]測試資料 (Part A-1) [cite: 55-61]
documents = [
    "人工智慧正在改變世界,機器學習是其核心技術",
    "深度學習推動了人工智慧的發展,特別是在圖像識別領域",
    "今天天氣很好,適合出去運動",
    "機器學習和深度學習都是人工智慧的重要分支",
    "運動有益健康,每天都應該保持運動習慣"
]

# [cite_start]測試文本 (Part A-2) [cite: 101-105]
test_texts = [
    "這家餐廳的牛肉麵真的太好吃了,湯頭濃郁,麵條Q彈,下次一定再來!",
    "最新的AI技術突破讓人驚艷,深度學習模型的表現越來越好",
    "這部電影劇情空洞,演技糟糕,完全是浪費時間",
    "每天慢跑5公里,配合適當的重訓,體能進步很多"
]

# [cite_start]測試文章 (Part A-3) [cite: 142-146]
article = """
人工智慧(AI)的發展正在深刻改變我們的生活方式。從早上起床時的智慧鬧鐘,
到通勤時的路線規劃,再到工作中的各種輔助工具,AI無處不在。
在醫療領域,AI協助醫生進行疾病診斷,提高了診斷的準確率和效率。透過分析
大量的醫療影像和病歷資料,AI能夠發現人眼容易忽略的細節,為患者提供更好
的治療方案。
教育方面,AI個人化學習系統能夠根據每個學生的學習進度和特點,提供客製化
的教學內容。這種因材施教的方式,讓學習變得更加高效和有趣。
然而,AI的快速發展也帶來了一些挑戰。首先是就業問題,許多傳統工作可能會
被AI取代。其次是隱私和安全問題,AI系統需要大量數據來訓練,如何保護個人
隱私成為重要議題。最後是倫理問題,AI的決策過程往往缺乏透明度,可能會產
生偏見或歧視。
面對這些挑戰,我們需要在推動AI發展的同時,建立相應的法律法規和倫理準則。
只有這樣,才能確保AI技術真正為人類福祉服務,創造一個更美好的未來。
"""

# --- 工具函數 ---
def get_tokens(text):
    return [t for t in jieba.cut(text) if t not in STOP_WORDS and len(t.strip()) > 0]

# --- A-1: TF-IDF 核心函數 ---
def calculate_tf(word_dict, total_words):
    """計算詞頻 (Term Frequency)"""
    tf_dict = {}
    if total_words == 0: return tf_dict
    for word, count in word_dict.items():
        tf_dict[word] = count / total_words
    return tf_dict

def calculate_idf(documents_tokens, word):
    """計算逆文件頻率 (Inverse Document Frequency)"""
    N = len(documents_tokens)
    doc_count = sum(1 for doc in documents_tokens if word in doc)
    # 使用平滑化公式 log(N / (df + 1)) + 1 防止除以零
    return math.log(N / (doc_count + 1)) + 1

def calculate_tfidf_vector(doc_tokens, all_docs_tokens):
    """計算單個文檔的 TF-IDF 向量"""
    all_words = set(word for doc in all_docs_tokens for word in doc)
    word_counts = Counter(doc_tokens)
    total_words = len(doc_tokens)
    tf_values = calculate_tf(word_counts, total_words)

    tfidf_vector = {}
    for word in all_words:
        idf = calculate_idf(all_docs_tokens, word)
        tf = tf_values.get(word, 0)
        tfidf_vector[word] = tf * idf
    
    # 轉為 numpy array，並確保詞彙順序固定
    sorted_words = sorted(list(all_words))
    vector_array = np.array([tfidf_vector[word] for word in sorted_words])
    return vector_array

def manual_cosine_similarity(vec1, vec2):
    """計算餘弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

# --- A-2: 規則分類器 ---
class RuleBasedSentimentClassifier:
    def __init__(self):
        self.positive_words = set(['好','棒','優秀','喜歡','推薦','滿意','開心','值得','精彩','完美','好吃','濃郁','Q彈','驚艷','進步'])
        self.negative_words = set(['差','糟','失望','討厭','不推薦','浪費','無聊','爛','糟糕','差勁','空洞'])
        self.negation_words = set(['不','沒','無','非','別'])
    
    def classify(self, text):
        tokens = list(jieba.cut(text)) # 不移除停用詞以免移除否定詞
        score = 0
        for i, token in enumerate(tokens):
            if token in STOP_WORDS and token not in self.negation_words: continue
            
            # 檢查前一個詞是否為否定詞
            is_negated = (i > 0 and tokens[i-1] in self.negation_words)
            
            if token in self.positive_words:
                score += (-1 if is_negated else 1)
            elif token in self.negative_words:
                score += (1 if is_negated else -1)
                
        if score > 0: return "正面"
        elif score < 0: return "負面"
        return "中性"

class TopicClassifier:
    def __init__(self):
        self.topic_keywords = {
            '科技': ['AI','人工智慧','電腦','軟體','程式','演算法','深度學習','機器學習','技術','模型'],
            '運動': ['運動','健身','跑步','游泳','球類','比賽','慢跑','重訓','體能'],
            '美食': ['吃','食物','餐廳','美味','料理','烹飪','牛肉麵','湯頭','麵條','好吃'],
            '旅遊': ['旅行','景點','飯店','機票','觀光','度假']
        }
    
    def classify(self, text):
        tokens = list(jieba.cut(text))
        scores = {topic: 0 for topic in self.topic_keywords}
        for token in tokens:
            for topic, keywords in self.topic_keywords.items():
                if token in keywords:
                    scores[topic] += 1
        
        max_score = max(scores.values())
        if max_score == 0: return "其他"
        return max(scores, key=scores.get)

# --- A-3: 統計式摘要 ---
class StatisticalSummarizer:
    def __init__(self):
        pass
    
    def summarize(self, text, ratio=0.3):
        # 簡單分句 (以句號、問號、驚嘆號分割，這裡簡化處理)
        raw_sentences = text.replace('\n', '').split('。')
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if not sentences: return ""

        # 計算全域詞頻
        all_tokens = []
        for s in sentences:
            all_tokens.extend(get_tokens(s))
        word_freq = Counter(all_tokens)
        
        sentence_scores = {}
        for i, sent in enumerate(sentences):
            score = 0
            sent_tokens = get_tokens(sent)
            if not sent_tokens: continue
            
            # 1. 詞頻加權
            for t in sent_tokens:
                score += word_freq[t]
            
            # 2. 位置加權 (首尾句重要)
            if i == 0 or i == len(sentences) - 1:
                score *= 1.2
            
            # 3. 長度懲罰 (過短忽略，過長扣分) -- 這裡使用簡化邏輯
            if len(sent) < 5: score = 0
            
            sentence_scores[sent] = score
            
        # 選出最高分句子
        count = max(1, int(len(sentences) * ratio))
        sorted_sent = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        top_sentences = set([s[0] for s in sorted_sent[:count]])
        
        # 依原文順序重組
        summary = [s for s in sentences if s in top_sentences]
        return "。".join(summary) + "。"

# --- 獨立執行檢查 ---
if __name__ == "__main__":
    print("=== Part A: 傳統方法 自我檢查 ===")
    
    # A-1
    print("\n[A-1 TF-IDF]")
    docs_tokens = [get_tokens(d) for d in documents]
    v1 = calculate_tfidf_vector(docs_tokens[0], docs_tokens)
    v4 = calculate_tfidf_vector(docs_tokens[3], docs_tokens)
    sim = manual_cosine_similarity(v1, v4)
    print(f"Doc1 vs Doc4 相似度: {sim:.4f}")
    
    # A-2
    print("\n[A-2 分類]")
    sent_cls = RuleBasedSentimentClassifier()
    topic_cls = TopicClassifier()
    for t in test_texts:
        print(f"'{t[:10]}...' -> {sent_cls.classify(t)} / {topic_cls.classify(t)}")
        
    # A-3
    print("\n[A-3 摘要]")
    summ = StatisticalSummarizer()
    res = summ.summarize(article, ratio=0.3)
    print(f"摘要結果 ({len(res)}字): {res[:50]}...")
