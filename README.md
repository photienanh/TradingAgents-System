# TradingAgents — AI Trading Platform

> Nền tảng giao dịch đa tác nhân AI, tích hợp phân tích kỹ thuật, tâm lý, tin tức, tài chính doanh nghiệp và alpha factor định lượng cho sàn HOSE.

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Tính năng chính](#tính-năng-chính)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
  - [Pipeline phân tích (TradingAgents)](#pipeline-phân-tích-tradingagents)
  - [Pipeline Alpha-GPT](#pipeline-alpha-gpt)
  - [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
  - [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
  - [Cài đặt dependencies](#cài-đặt-dependencies)
  - [Cấu hình biến môi trường](#cấu-hình-biến-môi-trường)
  - [Khởi tạo dữ liệu](#khởi-tạo-dữ-liệu)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
  - [Chạy Web App](#chạy-web-app)
  - [Chạy Alpha-GPT Pipeline (CLI)](#chạy-alpha-gpt-pipeline-cli)
  - [API Reference](#api-reference)
- [Cấu hình nâng cao](#cấu-hình-nâng-cao)

---

## Tổng quan

**TradingAgents** là framework multi-agent LLM cho phân tích và ra quyết định giao dịch cổ phiếu Việt Nam. Hệ thống gồm hai thành phần chính:

1. **TradingAgents Pipeline** — Mạng lưới các AI agent chuyên biệt (Market, Social, News, Fundamentals, Alpha) phân tích song song rồi tranh luận qua nhiều vòng để ra quyết định BUY/SELL/HOLD.

2. **Alpha-GPT Pipeline** — Hệ thống tự động sinh, tối ưu và kiểm định alpha factor theo phương pháp định lượng (LangGraph + Genetic Programming), lưu trữ kết quả vào thư viện alpha và tích hợp tín hiệu vào pipeline phân tích.

---

## Tính năng chính

### Phân tích đa tác nhân
- **Market Analyst** — Phân tích kỹ thuật: OHLCV, SMA/EMA, MACD, RSI, Bollinger Bands, ATR, VWMA, bối cảnh VN30.
- **Social Analyst** — Thu thập và phân tích tâm lý từ diễn đàn F247 và Google News theo mã.
- **News Analyst** — Tổng hợp tin tức doanh nghiệp (CafeF, vnstock VCI) và vĩ mô (Vietstock RSS, OpenAI web search).
- **Fundamentals Analyst** — Bảng cân đối kế toán, kết quả kinh doanh, lưu chuyển tiền tệ, chỉ số định giá (P/E, P/B, ROE, ...) từ vnstock/yfinance.
- **Alpha Analyst** — Tín hiệu định lượng từ alpha library, báo cáo IC, Sharpe, Return.

### Debate & Decision
- Vòng tranh luận Bull vs Bear Researcher (có thể cấu hình 1–5 vòng).
- Risk Management: 3 góc nhìn Risky / Safe / Neutral trước khi Portfolio Manager ra quyết định cuối.
- Memory system (ChromaDB + OpenAI embeddings) giúp các agent học từ lịch sử giao dịch.
- Hỗ trợ **Trading Horizon**: Swing Trading (2–7 ngày) hoặc Long-term Investing.

### Alpha-GPT
- Tự động sinh trading hypothesis và công thức alpha factor bằng LLM.
- Tối ưu formula qua Genetic Programming (crossover, mutation).
- Backtesting cross-sectional IC, Sharpe, annualized return trên universe.
- RAG từ thư viện alpha (FAISS + sentence-transformers) để tái sử dụng kinh nghiệm.
- Daily runner: tự động cập nhật dữ liệu thị trường và tính toán tín hiệu hàng ngày.

### Web Interface
- Dashboard real-time với sidebar, topbar và 4 section: Home, New Analysis, Sessions, Alpha.
- Bảng giá thị trường kiểu iBoard (VN30/HNX30/HOSE/HNX/UPCOM).
- Streaming log pipeline Alpha-GPT.
- Lưu trữ session phân tích vào SQLite, có thể xem lại và xoá.

### Multi-vendor Data
- Hỗ trợ nhiều nguồn dữ liệu với cơ chế fallback tự động: **vnstock**, **yfinance**, **CafeF**, **Vietstock**, **F247**, **OpenAI web search**.

---

## Kiến trúc hệ thống

### Pipeline phân tích (TradingAgents)

```
START
  │
  ├─► Market Analyst ──(tools)──► Msg Clear
  ├─► Social Analyst ──(tools)──► Msg Clear
  ├─► News Analyst ───(tools)──► Msg Clear
  ├─► Fundamentals Analyst ──(tools)──► Msg Clear
  └─► Alpha Analyst (không dùng tools)
        │
        ▼
  Bull Researcher ◄──────────────────────┐
        │                                │
        ▼                                │
  Bear Researcher ──────────────────────►┘
        │ (sau max_debate_rounds)
        ▼
  Research Manager (Investment Plan)
        │
        ▼
  Trader (Trading Plan)
        │
        ├─► Risky Analyst ◄─────────────────┐
        ├─► Safe Analyst                    │
        └─► Neutral Analyst ───────────────►┘
               │ (sau max_risk_discuss_rounds)
               ▼
          Portfolio Manager
               │
              END (Final Trade Decision)
```

### Pipeline Alpha-GPT

```
START
  │
  ▼
data_injector (load CSV market data)
  │
  ▼
hypothesis_generator (LLM + RAG từ FAISS KB)
  │
  ▼
alpha_generator (LLM sinh formula seeds)
  │
  ▼
gp_enhancement (Genetic Programming tối ưu IC)
  │
  ▼
backtest (cross-sectional IC / Sharpe / Return)
  │
  ▼
analyst (LLM review, xác định alpha yếu)
  │
  ▼
persist (SQLite + alpha_library.json)
  │
  ├─► [should_continue] ──► hypothesis_generator (vòng tiếp)
  └─► END
```

### Cấu trúc thư mục

```
.
├── alpha/                          # Alpha-GPT pipeline
│   ├── agents/                     # Các agent của Alpha-GPT
│   │   ├── alpha_generator_agent.py
│   │   ├── analyst_agent.py
│   │   ├── backtest_agent.py
│   │   ├── gp_agent.py
│   │   ├── hypothesis_agent.py
│   │   └── persist_agent.py
│   ├── knowledge/                  # RAG knowledge base
│   │   ├── alpha_kb_data.py        # 38 alphas từ Kakushadze (2016)
│   │   ├── build_kb.py             # Script build FAISS index
│   │   └── retriever.py            # FAISS retrieval
│   ├── prompts/                    # Prompt templates
│   ├── alpha_operators.py          # Toán tử alpha
│   ├── backtester.py               # Cross-sectional backtest engine
│   ├── config.py                   # Hyperparameters
│   ├── daily_runner.py             # Daily data & signal updater
│   ├── data_loader.py              # CSV loader + technical indicators
│   ├── gp_search.py                # Genetic Programming
│   ├── graph.py                    # LangGraph workflow
│   ├── manager.py                  # Alpha manager (threading)
│   ├── run.py                      # CLI entry point
│   ├── state.py                    # LangGraph state
│   └── validators.py               # Formula validation
│
├── tradingagents/                  # TradingAgents pipeline
│   ├── agents/                     # Các agent
│   │   ├── analysts/               # Market, Social, News, Fundamentals, Alpha
│   │   ├── researchers/            # Bull, Bear
│   │   ├── managers/               # Research Manager, Risk Manager
│   │   ├── risk_mgmt/              # Risky, Safe, Neutral debators
│   │   ├── trader/                 # Trader
│   │   └── utils/                  # Tools, memory, state definitions
│   ├── dataflows/                  # Data vendors
│   │   ├── cafef_news.py
│   │   ├── vnstock_finance.py
│   │   ├── y_finance.py
│   │   ├── social_media.py         # F247 scraper + Google News RSS
│   │   ├── vietstock_news.py
│   │   ├── openai.py               # Web search fallback
│   │   ├── interface.py            # Multi-vendor routing với fallback
│   │   └── config.py
│   ├── graph/                      # LangGraph orchestration
│   │   ├── trading_graph.py
│   │   ├── setup.py
│   │   ├── propagation.py
│   │   ├── conditional_logic.py
│   │   └── signal_processing.py
│   └── default_config.py
│
├── app/                            # FastAPI Web Application
│   ├── routes/
│   │   ├── alpha.py                # Alpha API endpoints
│   │   └── market.py               # Market data endpoints
│   ├── services/
│   │   ├── analysis_runner.py      # Background task runner
│   │   ├── progress_tracker.py     # Real-time progress tracking
│   │   ├── session_manager.py      # Thread-safe session store
│   │   └── session_serialization.py
│   ├── storage/
│   │   └── session_store.py        # SQLite session persistence
│   ├── static/                     # CSS + JS frontend
│   │   ├── css/
│   │   └── js/
│   ├── templates/
│   │   ├── index.html              # Main app
│   │   └── market_new.html         # Market iBoard
│   └── main.py                     # FastAPI entrypoint
│
├── data/                           # Data directory (tự tạo khi chạy)
│   ├── market_data/                # CSV files theo mã (ticker.csv)
│   ├── knowledge_base/             # FAISS index
│   ├── alpha_library.json          # Thư viện alpha tích lũy
│   ├── signals/                    # Daily signal snapshots
│   └── alphagpt.db                 # SQLite DB
│
└── .env                            # Biến môi trường
```

---

## Cài đặt

### Yêu cầu hệ thống

- Python **3.10+**
- RAM tối thiểu 4GB (8GB khuyến nghị khi chạy FAISS + GP)
- Kết nối internet để crawl dữ liệu thị trường

### Cài đặt dependencies

```bash
# Clone repository
git clone https://github.com/photienanh/TradingAgents-System
cd TradingAgents-System

# Tạo virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# hoặc: venv\Scripts\activate     # Windows

# Cài đặt packages
pip install -r requirements.txt
```

### Cấu hình biến môi trường

Tạo file `.env` tại root của project:

```env
# OpenAI API Key (bắt buộc)
OPENAI_API_KEY=sk-...

# Groq API Key (tuỳ chọn, dùng cho fallback model)
GROQ_API_KEY=gsk_...

# vnstock API Key (bắt buộc nếu dùng vnstock làm data vendor)
VNSTOCK_API_KEY=vnstock_...

# Đường dẫn thư mục dữ liệu thị trường (mặc định: ./data/market_data)
ALPHAGPT_DATA_DIR=./data/market_data

# Đường dẫn alpha library (mặc định: ./data/alpha_library.json)
ALPHA_LIBRARY_PATH=./data/alpha_library.json

# Đường dẫn SQLite database (mặc định: ./data/alphagpt.db)
ALPHAGPT_DB=./data/alphagpt.db

# Giờ bắt đầu daily update (mặc định: 9h sáng)
ALPHAGPT_DAILY_START_HOUR=9
```

### Khởi tạo dữ liệu

**Bước 1 — Chuẩn bị dữ liệu thị trường:**

Đặt file CSV lịch sử giá vào thư mục `data/market_data/`. Mỗi file tương ứng một mã cổ phiếu, tên file là `<TICKER>.csv`.

Định dạng file CSV:

```csv
time,ticker,open,high,low,close,volume,industry
2024-01-02,HPG,27500,28000,27200,27800,15000000,Thép
2024-01-03,HPG,27800,28500,27600,28200,18000000,Thép
...
```

**Bước 2 — Build FAISS knowledge base** (chỉ cần chạy một lần):

```bash
python -m alpha.knowledge.build_kb
```

Lệnh này tạo FAISS index từ 38 alpha Kakushadze (2016) tại `data/knowledge_base/`.

---

## Hướng dẫn sử dụng

### Chạy Web App

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Truy cập `http://localhost:8000` để mở giao diện web.

**Các trang chính:**

| URL | Mô tả |
|-----|-------|
| `/` | Dashboard chính (Home, Analysis, Sessions, Alpha) |
| `/market` | Bảng giá kiểu iBoard real-time |

**Hướng dẫn phân tích cổ phiếu:**

1. Vào mục **New Analysis** trên sidebar.
2. Nhập mã cổ phiếu (ví dụ: `HPG`, `VNM`, `MBB`).
3. Chọn **Trading Horizon**: Swing Trading hoặc Long-term Investing.
4. Chọn các analyst trong team (Market, Social, News, Fundamentals, Alpha).
5. Cấu hình Research Depth và LLM model.
6. Nhấn **Start Analysis** và theo dõi tiến trình.
7. Xem kết quả ở các tab: Phân tích hiện tại / Báo cáo tổng hợp / Quyết định.

### Chạy Alpha-GPT Pipeline (CLI)

**Tự động sinh trading idea:**

```bash
python -m alpha.run
```

**Chỉ định trading idea hoặc số vòng lặp:**

```bash
python -m alpha.run \
  --idea "RSI phân kỳ với giá trong xu hướng ngắn hạn" \
  --iterations 5
```

**Kết quả được lưu tại:**
- `data/alpha_library.json` — Thư viện alpha tích lũy qua các lần chạy.
- `data/alphagpt.db` — SQLite với toàn bộ lịch sử hypothesis, alpha và backtest.

**Chạy Alpha-GPT qua Web UI:**

1. Vào mục **Alpha** trên sidebar.
2. Chọn tab **Alpha Generator**.
3. Nhập trading idea (hoặc để trống để tự sinh).
4. Chọn số iteration rồi nhấn **Start Generation**.
5. Xem log streaming trực tiếp trong terminal panel.
6. Kết quả xuất hiện ở tab **Library** sau khi hoàn tất.

### API Reference

**Phân tích cổ phiếu:**

```http
POST /api/analyze
Content-Type: application/json

{
  "ticker": "HPG",
  "trading_horizon": "short",
  "analysts": ["market", "social", "news", "fundamentals", "alpha"],
  "research_depth": 1,
  "deep_think_llm": "gpt-4o-mini",
  "quick_think_llm": "gpt-4o-mini"
}
```

## Cấu hình nâng cao

### Thay đổi LLM provider

Chỉnh sửa `tradingagents/default_config.py`:

```python
DEFAULT_CONFIG = {
    "deep_think_llm":  "gpt-4o",           # Model cho phân tích sâu
    "quick_think_llm": "gpt-4o-mini",       # Model cho tác vụ nhanh
    "fallback_llm":    "openai/gpt-oss-120b", # Model fallback
    "backend_url":     "https://api.openai.com/v1",
    ...
}
```

Hoặc truyền trực tiếp khi gọi API:

```json
{
  "deep_think_llm": "gpt-4o",
  "quick_think_llm": "gpt-4o-mini"
}
```

### Thay đổi nguồn dữ liệu

Trong `tradingagents/default_config.py`, mỗi loại dữ liệu có thể dùng vendor khác nhau:

```python
"data_vendors": {
    "core_stock_apis":        "vnstock",   # hoặc "yfinance"
    "technical_indicators":   "vnstock",
    "fundamental_data":       "yfinance",  # hoặc "vnstock", "openai"
    "news_data":              "cafef",     # hoặc "vnstock", "openai"
    "social_data":            "f247",
    "global_data":            "vietstock", # hoặc "openai"
},
"tool_vendors": {
    "get_news": "cafef, vnstock",          # Dùng nhiều vendor, nối bằng dấu phẩy
},
```

### Tham số Alpha-GPT

Chỉnh sửa `alpha/config.py`:

```python
@dataclass
class PipelineConfig:
    ic_signal_threshold:  float = 0.02   # IC_OOS tối thiểu để alpha đạt chuẩn
    sharpe_min_threshold: float = 1.0    # Sharpe ratio tối thiểu
    return_min_threshold: float = 0.0    # Annualized return tối thiểu
    test_ratio:           float = 0.3    # Tỷ lệ out-of-sample
    min_sota:             int   = 3      # Số alpha SOTA tối thiểu để dừng
    max_sota:             int   = 5      # Số alpha SOTA tối đa lưu giữ
    corr_threshold:       float = 0.55   # Ngưỡng tương quan để dedup alpha
    gp_iterations:        int   = 15     # Số vòng GP per seed alpha
    population_size:      int   = 6      # Kích thước population trong GP
    rag_top_k:            int   = 5      # Số alpha retrieve từ FAISS KB
```