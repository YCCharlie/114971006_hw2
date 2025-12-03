import time
import pandas as pd
import os
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import userdata

# 關鍵步驟：匯入前兩個獨立檔案作為模組
# 如果這裡報錯，代表您沒有先執行 %%writefile traditional_methods.py 和 modern_methods.py
import traditional_methods as tm
import modern_methods as mm

# --- 初始化設定 ---
def init_client():
    """初始化 OpenAI Client，優先讀取環境變數，其次讀取 Colab Secrets"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = userdata.get('OPENAI_API_KEY')
        except:
            pass
    
    if api_key:
        return openai.OpenAI(api_key=api_key)
    else:
        print("⚠️ [警告] 未偵測到 API Key。Part B (AI) 功能將無法執行，僅顯示傳統方法結果。")
        return None

def run_comparison():
    client = init_client()
    
    print("\n" + "="*80)
    print(" 📊 Part C: 傳統 vs 現代 NLP 方法 比較報告")
    print("="*80)

    # ==========================================
    # 任務 1: 相似度計算比較 [cite: 194]
    # ==========================================
    print("\n🔹 [比較 1] 文本相似度計算 (Similarity)")
    print("-" * 60)
    
    # 測試樣本: D1 (人工智慧) vs D4 (機器學習...人工智慧)
    doc_a = tm.documents[0]
    doc_b = tm.documents[3]
    
    # --- 方法 A: 傳統手動 TF-IDF ---
    start_time = time.time()
    # 重新取得分詞列表
    tokens_list = [tm.get_tokens(d) for d in tm.documents]
    vec_a = tm.calculate_tfidf_vector(tokens_list[0], tokens_list)
    vec_b = tm.calculate_tfidf_vector(tokens_list[3], tokens_list)
    score_manual = tm.manual_cosine_similarity(vec_a, vec_b)
    time_manual = time.time() - start_time

    # --- 方法 B: Scikit-learn (基準) ---
    vectorizer = TfidfVectorizer(tokenizer=tm.get_tokens, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform(tm.documents)
    matrix_sim = cosine_similarity(tfidf_matrix)
    score_sklearn = matrix_sim[0][3]

    # --- 方法 C: 現代 AI (GPT-4o) ---
    start_time_ai = time.time()
    score_ai = mm.ai_similarity(doc_a, doc_b, client)
    time_ai = time.time() - start_time_ai

    # --- 輸出結果 ---
    print(f"1. 傳統 TF-IDF (Manual):  {score_manual:.4f} (耗時: {time_manual:.5f}s)")
    print(f"2. 傳統 TF-IDF (Sklearn): {score_sklearn:.4f}")
    print(f"3. 現代 AI (GPT-4o):      {score_ai}/100   (耗時: {time_ai:.5f}s)")


    # ==========================================
    # 任務 2: 文本分類比較 [cite: 194]
    # ==========================================
    print("\n\n🔹 [比較 2] 文本分類 (Classification)")
    print("-" * 60)

    rule_sentiment = tm.RuleBasedSentimentClassifier()
    rule_topic = tm.TopicClassifier()

    results_data = []

    for text in tm.test_texts:
        # 傳統
        t0 = time.time()
        trad_sent = rule_sentiment.classify(text)
        trad_topic = rule_topic.classify(text)
        t_trad = time.time() - t0

        # AI
        t0 = time.time()
        ai_res = mm.ai_classify(text, client)
        t_ai = time.time() - t0

        results_data.append({
            "Text": text[:8] + "...",
            "Trad_Sent": trad_sent,
            "AI_Sent": ai_res.get('sentiment'),
            "Trad_Topic": trad_topic,
            "AI_Topic": ai_res.get('topic'),
            "T_Time": f"{t_trad:.4f}s",
            "A_Time": f"{t_ai:.4f}s"
        })

    df = pd.DataFrame(results_data)
    # 顯示表格
    print(df.to_string(index=False))


    # ==========================================
    # 任務 3: 自動摘要比較 [cite: 194]
    # ==========================================
    print("\n\n🔹 [比較 3] 自動摘要 (Summarization)")
    print("-" * 60)
    
    article_src = tm.article
    print(f"原文長度: {len(article_src)} 字")

    # 傳統 (統計式)
    t0 = time.time()
    summ_trad = tm.StatisticalSummarizer().summarize(article_src, ratio=0.3)
    t_trad = time.time() - t0

    # AI (生成式)
    t0 = time.time()
    summ_ai = mm.ai_summarize(article_src, max_length=150, client=client)
    t_ai = time.time() - t0

    print(f"\n[傳統摘要] (耗時 {t_trad:.4f}s):")
    print(f"\"{summ_trad[:60]}...\"")
    
    print(f"\n[AI 摘要] (耗時 {t_ai:.4f}s):")
    print(f"\"{summ_ai[:60]}...\"")

    print("\n" + "="*80)
    print(" ✅ Part C 執行完成")
    print("="*80)

if __name__ == "__main__":
    run_comparison()
