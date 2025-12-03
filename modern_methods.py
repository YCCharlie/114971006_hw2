import json
import os
import openai

# --- B-1: 語意相似度 ---
def ai_similarity(text1, text2, client):
    """B-1: 使用 AI 計算語意相似度 (0-100)"""
    if not client: return 0
    prompt = f"""
    請評估以下兩段文字的語意相似度。
    文字1: {text1}
    文字2: {text2}
    請只回答一個 0-100 的整數，代表相似度百分比，不要有任何其他文字。
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        content = response.choices[0].message.content.strip()
        # 清理非數字字符
        digits = ''.join(filter(str.isdigit, content))
        return int(digits) if digits else 0
    except Exception as e:
        print(f"Similarity Error: {e}")
        return 0

# --- B-2: 文本分類 (含快取機制) ---
_CLASSIFY_CACHE = {}

def ai_classify(text, client, use_cache=True):
    """B-2: 使用 AI 進行多維度分類"""
    if use_cache and text in _CLASSIFY_CACHE:
        return _CLASSIFY_CACHE[text]

    if not client: return {"sentiment": "N/A", "topic": "N/A"}
    
    prompt = f"""
    請分析以下文本："{text}"
    請以 JSON 格式返回：
    {{
        "sentiment": "正面" 或 "負面" 或 "中性",
        "topic": "科技" 或 "運動" 或 "美食" 或 "旅遊" 或 "其他",
        "confidence": 0.0 到 1.0 的數字
    }}
    只返回 JSON。
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        if use_cache:
            _CLASSIFY_CACHE[text] = result
        return result
    except Exception as e:
        print(f"Classify Error: {e}")
        return {"sentiment": "Error", "topic": "Error"}

# --- B-3: 自動摘要 ---
def ai_summarize(text, max_length, client):
    """B-3: AI 自動摘要"""
    if not client: return "Client not initialized"
    prompt = f"""
    請將以下文章摘要，長度控制在 {max_length} 字以內，保留關鍵資訊。
    文章: {text}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_length * 2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summarize Error: {e}"

# --- 獨立執行檢查 ---
if __name__ == "__main__":
    print("=== Part B: 現代方法 獨立檢查 ===")
    
    # 嘗試從環境變數讀取 Key
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key:
        client = openai.OpenAI(api_key=api_key)
        print("✅ Client initialized.")
        
        # B-1 測試
        t1 = "機器學習是AI的核心"
        t2 = "深度學習推動了AI發展"
        print(f"\n[B-1 相似度] '{t1}' vs '{t2}'")
        score = ai_similarity(t1, t2, client)
        print(f"得分: {score}")
        
        # B-2 測試
        print(f"\n[B-2 分類] '{t1}'")
        cls_res = ai_classify(t1, client)
        print(f"結果: {cls_res}")

        # B-3 測試 (新增)
        print(f"\n[B-3 摘要]")
        article_text = "人工智慧(AI)的發展正在深刻改變我們的生活方式。從早上起床時的智慧鬧鐘到工作中的各種輔助工具,AI無處不在。在醫療領域,AI協助醫生進行疾病診斷,提高了診斷的準確率和效率。"
        summary_res = ai_summarize(article_text, max_length=50, client=client)
        print(f"原文 ({len(article_text)}字) -> 摘要結果: {summary_res}")

    else:
        print("⚠️ 無 API Key，跳過實際 API 呼叫測試。僅驗證函數定義無誤。")
