# 114971006_hw2
NCCU 114971006 GenAI Assignment 2


輸出結果如下：
Requirement already satisfied: jieba in /usr/local/lib/python3.12/dist-packages (0.42.1)
Requirement already satisfied: scikit-learn in /usr/local/lib/python3.12/dist-packages (1.6.1)
Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (2.0.2)
Requirement already satisfied: TCSP in /usr/local/lib/python3.12/dist-packages (0.0.9)
Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (1.16.3)
Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (1.5.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (3.6.0)
Requirement already satisfied: openai in /usr/local/lib/python3.12/dist-packages (1.109.1)
Requirement already satisfied: anyio<5,>=3.5.0 in /usr/local/lib/python3.12/dist-packages (from openai) (4.11.0)
Requirement already satisfied: distro<2,>=1.7.0 in /usr/local/lib/python3.12/dist-packages (from openai) (1.9.0)
Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.12/dist-packages (from openai) (0.28.1)
Requirement already satisfied: jiter<1,>=0.4.0 in /usr/local/lib/python3.12/dist-packages (from openai) (0.12.0)
Requirement already satisfied: pydantic<3,>=1.9.0 in /usr/local/lib/python3.12/dist-packages (from openai) (2.11.10)
Requirement already satisfied: sniffio in /usr/local/lib/python3.12/dist-packages (from openai) (1.3.1)
Requirement already satisfied: tqdm>4 in /usr/local/lib/python3.12/dist-packages (from openai) (4.67.1)
Requirement already satisfied: typing-extensions<5,>=4.11 in /usr/local/lib/python3.12/dist-packages (from openai) (4.15.0)
Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.12/dist-packages (from anyio<5,>=3.5.0->openai) (3.11)
Requirement already satisfied: certifi in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->openai) (2025.10.5)
Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->openai) (1.0.9)
Requirement already satisfied: h11>=0.16 in /usr/local/lib/python3.12/dist-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.16.0)
Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.12/dist-packages (from pydantic<3,>=1.9.0->openai) (0.7.0)
Requirement already satisfied: pydantic-core==2.33.2 in /usr/local/lib/python3.12/dist-packages (from pydantic<3,>=1.9.0->openai) (2.33.2)
Requirement already satisfied: typing-inspection>=0.4.0 in /usr/local/lib/python3.12/dist-packages (from pydantic<3,>=1.9.0->openai) (0.4.2)
✅ 成功讀取 OPENAI_API_KEY。
✅ OpenAI 客戶端初始化完成。

================================================================================
                       【Part A-1: TF-IDF 文本相似度計算】
================================================================================

1. 手動計算 TF-IDF 相似度 (Doc 1 vs Doc 4):
   相似度得分: 0.1274

2. Scikit-learn TF-IDF 相似度矩陣 (5x5):
[[1.         0.18270139 0.04170053 0.2944224  0.03759802]
 [0.18270139 1.         0.03808984 0.34052565 0.03434255]
 [0.04170053 0.03808984 1.         0.         0.20950992]
 [0.2944224  0.34052565 0.         1.         0.07719388]
 [0.03759802 0.03434255 0.20950992 0.07719388 1.        ]]

================================================================================
                     【Part A-2: 基於規則的文本分類】
================================================================================

1. 情感分類器:
   - 文本: '這家餐廳的牛肉麵真的太好吃了,...' -> 情感: 正面
   - 文本: '最新的AI技術突破讓人驚艷,深...' -> 情感: 正面
   - 文本: '這部電影劇情空洞,演技糟糕,完...' -> 情感: 負面
   - 文本: '每天慢跑5公里,配合適當的重訓...' -> 情感: 正面

2. 主題分類器:
   - 文本: '這家餐廳的牛肉麵真的太好吃了,...' -> 主題: 美食
   - 文本: '最新的AI技術突破讓人驚艷,深...' -> 主題: 科技
   - 文本: '這部電影劇情空洞,演技糟糕,完...' -> 主題: 其他/無法分類
   - 文本: '每天慢跑5公里,配合適當的重訓...' -> 主題: 運動

================================================================================
                     【Part A-3: 統計式自動摘要】
================================================================================

原文長度: 411 字
摘要內容 (Ratio=30%, 116 字):
人工智慧(AI)的發展正在深刻改變我們的生活方式。透過分析
大量的醫療影像和病歷資料,AI能夠發現人眼容易忽略的細節,為患者提供更好
的治療方案。教育方面,AI個人化學習系統能夠根據每個學生的學習進度和特點,提供客製化
的教學內容。

================================================================================
                       【Part B-1: 語意相似度計算】
================================================================================
語意相似度計算 (GPT-4o, Text A vs Text B): 78 / 100
處理時間: 1.1534 秒

================================================================================
                       【Part B-2: AI 文本分類】
================================================================================

AI 文本分類 (GPT-4o, Text C):
{
    "sentiment": "負面",
    "topic": "娛樂",
    "confidence": 0.95
}
處理時間: 0.7240 秒

================================================================================
                       【Part B-3: AI 自動摘要】
================================================================================

--- 輸出詳情 ---

[ 摘要內容 ]
--------------------------------------------------------------------------------
人工智慧(AI)深刻改變生活方式，從智慧鬧鐘、通勤規劃到工作輔助均可見其應用。在醫療領域，它提高了診斷準確性和效率，能分析大量影像和資料提供更佳治療方
案。在教育方面，AI個人化學習系統可依學生進度提供客製教學，提升學習效率與趣味。然而，AI發展也帶來就業、隱私和倫理挑戰，威脅工作機會並涉及數據安全和
決策透明問題。因此，推動AI的同時需建立法律和倫理準則，確保AI促進人類福祉。
--------------------------------------------------------------------------------
處理時間: 5.6020 秒

================================================================================
                     【Part C-1: 量化比較運行】
================================================================================

傳統方法總處理時間 (4個文本): 0.001212 秒
AI 方法總處理時間 (4個文本): 4.328504 秒

情感分類傳統準確率: 0.50
情感分類 AI 準確率: 0.75

================================================================================
                       【額外加分 1: 詞雲視覺化】
================================================================================
/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 25991 (\N{CJK UNIFIED IDEOGRAPH-6587}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)
/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 31456 (\N{CJK UNIFIED IDEOGRAPH-7AE0}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)
/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 35422 (\N{CJK UNIFIED IDEOGRAPH-8A5E}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)
/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 24409 (\N{CJK UNIFIED IDEOGRAPH-5F59}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)
/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 38971 (\N{CJK UNIFIED IDEOGRAPH-983B}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)
/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 29575 (\N{CJK UNIFIED IDEOGRAPH-7387}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)
/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 38642 (\N{CJK UNIFIED IDEOGRAPH-96F2}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)
/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 22294 (\N{CJK UNIFIED IDEOGRAPH-5716}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)


================================================================================
                       【額外加分 2: 效能優化 (快取)】
================================================================================

文本: '最新的AI技術突破讓...'
第一次呼叫: 狀態=API Call, 時間=0.8664 秒
第二次呼叫: 狀態=Cache Hit, 時間=0.0001 秒

結論: Cache Hit 的時間應該遠小於第一次 API Call 的時間。
