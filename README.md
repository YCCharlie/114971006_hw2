# 國立政治大學 - 作業二：傳統 NLP vs 現代 AI 文本處理方法比較

### 項目摘要
[cite_start]本項目實作了 TF-IDF、規則分類等傳統 NLP 技術，並與 GPT-4o 模型進行相同任務的實作與量化比較 [cite: 207, 217, 218][cite_start]。目標是比較兩種技術在速度、成本和語義理解上的核心差異 [cite: 219]。

### 執行環境與操作說明 (Colab 專用)

本程式碼已作為 Colab Notebook (.ipynb) 檔案提交，助教無需在本地環境安裝大量套件。

#### 1. 必要依賴
本 Notebook 內部已包含所有依賴的安裝指令：
- **基礎套件:** `scikit-learn`, `jieba`, `numpy`, `TCSP`
- **AI 依賴:** `openai`
（執行時，請確保所有的 `!pip install` 指令區塊已運行。）

#### 2. API Key 設定 (關鍵步驟)
[cite_start]為了安全起見，OpenAI API Key 必須透過 Colab Secrets 傳入 [cite: 536]。

1.  開啟此 Colab Notebook。
2.  點擊左側欄的「🔑 **Secrets** (密鑰)」圖標。
3.  新增一個密鑰，名稱必須設定為：`OPENAI_API_KEY`。
4.  將你的 OpenAI API Key 貼入值欄位。

#### 3. 運行步驟
請依照 Notebook 中的程式碼區塊順序執行：
1.  執行**環境設置與 Client 初始化**區塊 (檢查 API Key 連線)。
2.  依序執行 **Part A-1, A-2, A-3** 的實作區塊 (傳統方法計算)。
3.  依序執行 **Part B-1, B-2, B-3** 的實作區塊 (現代 AI 呼叫)。
4.  執行 **Part C-1 量化比較運行**區塊 (獲取最終準確率和時間數據)。

---

### 作業成果展示

#### 1. [額外加分] 效能優化 (快取機制)
[cite_start]程式碼實作了快取機制，以避免重複呼叫 API 造成費用浪費 [cite: 465]。
- **實測效果:** 第二次呼叫狀態為 `Cache Hit`，時間顯著低於第一次 `API Call` 時間，證明快取實作成功。

#### 2. [額外加分] 詞雲圖視覺化
成功對 Part A-3 的文章進行視覺化分析，展示了文章高頻詞彙的分佈。
![詞雲圖](請將詞雲圖圖片檔案上傳並在此處填寫路徑，例如：`word_cloud.png`)

---
**提交文件:**
- `學號_hw2.ipynb` (Colab Notebook 程式碼)
- `report.md` 或 PDF 報告 (Part C-2 質性分析和所有截圖)
