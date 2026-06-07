# 貢獻指南

本文件說明本實驗室的 Git commit 撰寫規範，所有成員提交程式碼前請務必遵守。

---

## Commit 訊息格式

```
<類型>(<範圍>): <簡短描述>

<詳細說明（選填）>
```

### 規則

- 第一行（標題）**不超過 72 個字元**
- 標題使用**現在式**動詞（新增、修正、重構，而非新增了、修正了）
- 標題說明「**為什麼**改」，而非只說「改了什麼」（程式碼本身就能說明做了什麼）
- 詳細說明與標題之間空一行
- 每個 commit 只做**一件事**；若改了多個不相關的事，請拆成多個 commit

---

## 類型（Type）

| 類型 | 使用時機 |
|------|---------|
| `feat` | 新增功能或模組 |
| `fix` | 修復錯誤或非預期行為 |
| `refactor` | 重構程式碼（不改變外部行為） |
| `docs` | 新增或更新文件（README、CLAUDE.md 等） |
| `test` | 新增或修改測試 |
| `chore` | 雜項維護（更新 .gitignore、設定檔、依賴套件等） |
| `perf` | 效能改善 |
| `style` | 排版、命名等不影響邏輯的格式調整 |

---

## 範圍（Scope）（選填）

範圍表示這次改動影響的模組或功能區塊，使用小寫英文，例如：

| 範圍 | 對應模組 |
|------|---------|
| `model` | `model.py` — TimeLLMWithACE 架構 |
| `ace` | `ace.py` — ACE_Playbook 動態提示詞 |
| `data` | `data_utils.py`、`dataset.py` — 資料前處理 |
| `train` | `train.py` — 訓練流程 |
| `infer` | `inference.py` — 推論與填答 |
| `eval` | `evaluate.py` — 評估指標 |
| `config` | 設定檔、`.gitignore`、`CLAUDE.md` 等 |

---

## 詳細範例

### 新增功能

```
feat(ace): 加入滾動視窗策略，避免提示詞重複

ACE_Playbook 原本只保留最新一條策略，
改為保留最近 3 條以提升上下文多樣性，
降低模型對單一策略過度依賴的風險。
```

### 修復錯誤

```
fix(infer): 修正缺失值以訓練集均值填補而非零值

原本對 disp_x / disp_z 缺失位置填入 0，
導致靠近序列起始點的預測偏差過大。
改為使用訓練集的 train_mean 作為填補基準值。
```

### 重構

```
refactor(data): 將日期解析邏輯抽離為獨立函式

原本 prepare_training_data 與 build_env_dict 都有
各自的 regex 日期解析，造成重複維護的問題。
統一由 extract_date_key() 處理，方便日後修改格式。
```

### 文件更新

```
docs(CLAUDE.md): 補充 TFT 模型執行參數說明

新增 --max_encoder_length、--hidden_size 等旗標的
預設值與建議調整範圍，方便新成員快速上手。
```

### 雜項維護

```
chore(config): 將原始資料壓縮檔加入 .gitignore

TRAIN 0-5.zip 與 TEST 0-5.zip 檔案過大，
不應納入版本控制，改由 README 提供 Google Drive 下載連結。
```

### 效能改善

```
perf(model): 以 8-bit 量化載入 LLaMA-2 減少顯存佔用

在 Colab T4（16GB）環境下，fp16 完整載入會 OOM，
改用 BitsAndBytesConfig load_in_8bit=True 可降至約 9GB。
```

---

## 常見錯誤示範（請避免）

```
# 太模糊，看不出改了什麼
git commit -m "update"
git commit -m "fix bug"
git commit -m "修改"

# 說「做了什麼」而非「為什麼」（程式碼已能說明做了什麼）
git commit -m "feat: 在 model.py 第 23 行新增一個 Linear layer"

# 一次 commit 做太多不相關的事
git commit -m "fix inference + update README + 修改 gitignore"
```

---

## 快速對照表

| 情境 | 建議格式 |
|------|---------|
| 新增一個推論模組 | `feat(infer): 新增批次推論功能以加速測試集填答` |
| 修正訓練時 loss 為 NaN | `fix(train): 對 disp_diff 正規化加入 eps 避免除以零` |
| 更新模型說明文件 | `docs(model): 補充 patch_embedding 維度計算說明` |
| 升級 PyTorch 版本 | `chore(config): 將 PyTorch 升級至 2.3 以支援 CUDA 12.1` |
| 移除 Jupyter Notebook 改為 .py | `refactor: 將車床.ipynb 拆解為獨立模組化 Python 腳本` |
