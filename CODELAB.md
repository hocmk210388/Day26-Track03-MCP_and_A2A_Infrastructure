# Codelab: Xây Dựng Hệ Thống Multi-Agent với A2A Protocol

**Thời gian:** 2 giờ  
**Ngôn ngữ:** Python 3.11+  
**Công nghệ:** LangGraph, LangChain, A2A SDK

## Mục Tiêu Học Tập

Sau khi hoàn thành codelab này, bạn sẽ:
- Hiểu cách LLM hoạt động từ cơ bản đến nâng cao
- Biết cách tích hợp tools và RAG vào LLM
- Xây dựng được single agent với ReAct pattern
- Tạo multi-agent system với LangGraph
- Triển khai distributed agents với A2A protocol

## Chuẩn Bị

### Yêu Cầu Hệ Thống
- Python 3.11 trở lên
- [uv](https://docs.astral.sh/uv/) package manager
- API key từ [OpenRouter](https://openrouter.ai)

### Cài Đặt

```bash
# Clone repository
git clone <repo-url>
cd legal_multiagent

# Cài đặt dependencies
uv sync

# Cấu hình environment
cp .env.example .env
# Sửa file .env, thêm OPENROUTER_API_KEY của bạn
```

---

## Phần 1: Direct LLM Calling (20 phút)

### Lý Thuyết

LLM (Large Language Model) ở dạng cơ bản nhất là một API nhận input text và trả về output text. Không có memory, không có tools, chỉ dựa vào training data.

**Ưu điểm:**
- Đơn giản, dễ implement
- Phản hồi nhanh

**Nhược điểm:**
- Không có kiến thức real-time
- Không thể tra cứu database
- Không có context giữa các lần gọi

### Thực Hành

**Bước 1:** Chạy demo Stage 1

```bash
uv run python stages/stage_1_direct_llm/main.py
```

**Bước 2:** Đọc và hiểu code

Mở file `stages/stage_1_direct_llm/main.py` và trả lời:

1. LLM được khởi tạo như thế nào? (Tìm hàm `get_llm()`)
2. Message được gửi đến LLM có cấu trúc gì?
3. Tại sao cần có `SystemMessage` và `HumanMessage`?

**Đáp án**

1. Trong `main.py` gọi `llm = get_llm()`. Hàm `get_llm()` nằm ở `common/llm.py`: trả về `ChatOpenAI` trỏ OpenRouter (`openai_api_base`), model từ `OPENROUTER_MODEL`, API key từ `OPENROUTER_API_KEY`, có `temperature` và `max_tokens` (giới hạn credit).
2. Một **list** các message LangChain theo thứ tự: `SystemMessage(content=...)` rồi `HumanMessage(content=QUESTION)`. Truyền vào `await llm.ainvoke(messages)`.
3. `SystemMessage` đặt **vai trò và quy tắc** (ví dụ “bạn là chuyên gia pháp lý”, giới hạn độ dài). `HumanMessage` là **nội dung người dùng** (câu hỏi). Tách hai loại giúp model phân biệt chỉ dẫn hành vi và nhiệm vụ cần trả lời.

**Bài Tập 1.1:** Thay đổi câu hỏi

Sửa biến `QUESTION` thành câu hỏi pháp lý khác (tiếng Việt hoặc tiếng Anh) và chạy lại.

**Bài Tập 1.2:** Thêm temperature control

Thêm parameter `temperature=0.3` vào hàm `get_llm()` trong `common/llm.py` để làm output ổn định hơn.

---

## Phần 2: LLM + RAG & Tools (30 phút)

### Lý Thuyết

**RAG (Retrieval-Augmented Generation):** Cho phép LLM tra cứu knowledge base trước khi trả lời.

**Tools:** Các function mà LLM có thể gọi để thực hiện tác vụ cụ thể (tính toán, query database, gọi API).

**Function Calling Flow:**
1. LLM nhận câu hỏi + danh sách tools
2. LLM quyết định gọi tool nào (hoặc không gọi)
3. Tool được execute, trả về kết quả
4. LLM nhận kết quả và tạo câu trả lời cuối cùng

### Thực Hành

**Bước 1:** Chạy demo Stage 2

```bash
uv run python stages/stage_2_rag_tools/main.py
```

**Bước 2:** Phân tích code

Mở `stages/stage_2_rag_tools/main.py` và tìm:

1. Hàm `@tool` decorator được dùng ở đâu?
2. `LEGAL_KNOWLEDGE` được cấu trúc như thế nào?
3. LLM được bind với tools ra sao? (Tìm `.bind_tools()`)

**Đáp án**

1. Trên các hàm `search_legal_database`, `calculate_damages`, `check_statute_of_limitations` (import `tool` từ `langchain_core.tools`).
2. `LEGAL_KNOWLEDGE` là **list các dict**, mỗi phần tử có `id`, `keywords` (list từ khóa), `text` (đoạn văn bản kiến thức). Tool search so khớp từ khóa trong query với `keywords`.
3. `llm_with_tools = llm.bind_tools(TOOLS)` — gắn danh sách tool vào model để API completion có thể trả về `tool_calls`.

**Bài Tập 2.1:** Thêm knowledge base entry

Thêm một entry mới vào `LEGAL_KNOWLEDGE` về luật lao động:

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
}
```

**Bài Tập 2.2:** Tạo tool mới

Tạo một tool `@tool` mới tên `check_statute_of_limitations` nhận vào `case_type` (string) và trả về thời hiệu khởi kiện:

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.
    
    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

Thêm tool này vào danh sách tools và test.

---

## Phần 3: Single Agent với ReAct (25 phút)

### Lý Thuyết

**ReAct Pattern:** Reasoning + Acting

Agent tự động lặp lại chu trình:
1. **Think:** Suy nghĩ cần làm gì
2. **Act:** Gọi tool
3. **Observe:** Nhận kết quả
4. Lặp lại cho đến khi có câu trả lời cuối cùng

LangGraph cung cấp `create_react_agent` để tự động hóa pattern này.

### Thực Hành

**Bước 1:** Chạy demo Stage 3

```bash
uv run python stages/stage_3_single_agent/main.py
```

**Bước 2:** Quan sát output

Chú ý cách agent tự động:
- Quyết định tool nào cần gọi
- Gọi nhiều tools liên tiếp
- Tổng hợp kết quả

**Bước 3:** Đọc code

Mở `stages/stage_3_single_agent/main.py`:

1. Tìm `create_react_agent()` — đây là magic function
2. So sánh với Stage 2: không còn manual tool loop
3. Xem `agent_executor.invoke()` — chỉ cần gọi một lần

**Đáp án**

1. `from langgraph.prebuilt import create_react_agent`, gọi với `model=llm`, `tools=TOOLS`, `prompt=SYSTEM_PROMPT` (trong repo có thêm `debug=True` để log luồng nội bộ).
2. Stage 2 tự viết vòng lặp: gọi LLM → nếu có `tool_calls` thì execute tool → thêm `ToolMessage` → gọi LLM lại. Stage 3 LangGraph gói chu trình ReAct; không cần tự cập nhật `messages` theo từng bước như Stage 2.
3. Trong file demo hiện tại **không** có biến tên `agent_executor`; thay bằng `graph = create_react_agent(...)` rồi **`graph.astream(...)`** để in từng bước. Về ý niệm giống CODELAB: **một lần đưa input vào graph** là đủ (có thể dùng `graph.ainvoke(inputs)` nếu chỉ cần kết quả cuối).

**Bài Tập 3.1:** Thêm tool tra cứu án lệ

```python
@tool
def search_case_law(keywords: str) -> str:
    """Tìm kiếm án lệ theo từ khóa.
    
    Args:
        keywords: Từ khóa tìm kiếm
    """
    cases = {
        "breach": "Hadley v. Baxendale (1854) - Consequential damages",
        "negligence": "Donoghue v. Stevenson (1932) - Duty of care",
        "contract": "Carlill v. Carbolic Smoke Ball Co (1893) - Unilateral contract",
    }
    for key, case in cases.items():
        if key in keywords.lower():
            return case
    return "Không tìm thấy án lệ phù hợp"
```

Thêm vào tools list và test với câu hỏi về breach of contract.

**Bài Tập 3.2:** Debug agent reasoning

Trong tài liệu gốc gợi ý `verbose=True`. Trên **LangGraph 1.x**, `create_react_agent` không có `verbose`; hãy dùng **`debug=True`** để xem log chi tiết luồng agent (đã áp dụng trong `stages/stage_3_single_agent/main.py`).

---

## Phần 4: Multi-Agent In-Process (30 phút)

### Lý Thuyết

**Multi-Agent System:** Nhiều agents chuyên môn hóa cùng làm việc.

**Ưu điểm:**
- Mỗi agent tập trung vào domain riêng
- Có thể chạy song song (parallel execution)
- Dễ maintain và mở rộng

**LangGraph StateGraph:**
- Định nghĩa state (dữ liệu chia sẻ giữa các nodes)
- Tạo nodes (các bước xử lý)
- Định nghĩa edges (luồng điều khiển)

**Send API:** Cho phép dispatch nhiều tasks song song.

### Thực Hành

**Bước 1:** Chạy demo Stage 4

```bash
uv run python stages/stage_4_milti_agent/main.py
```

**Bước 2:** Phân tích kiến trúc

Mở `stages/stage_4_milti_agent/main.py`:

1. Tìm `class State(TypedDict)` — đây là shared state
2. Tìm các agent functions: `law_agent`, `tax_agent`, `compliance_agent`
3. Tìm `Send()` API — dispatch parallel tasks
4. Xem `graph.add_node()` và `graph.add_edge()`

**Đáp án**

1. Trong code là **`LegalState(TypedDict)`** (cùng vai trò “shared state” như `State` trong đề): gồm `question`, `law_analysis`, `needs_tax`, `needs_compliance`, `tax_result`, `compliance_result`, `privacy_analysis`, `final_answer`; một số field dùng `Annotated[..., _last_wins]` để merge khi chạy song song.
2. Tên hàm/node trong repo: **`analyze_law`** (luật tổng), **`call_tax_specialist`**, **`call_compliance_specialist`**, **`call_privacy_agent`** — tương đương law / tax / compliance / privacy trong đề bài.
3. **`from langgraph.types import Send`**. Trong `route_to_specialists`, trả về `list[Send]` — nhiều `Send("tên_node", state)` → các nhánh có thể chạy **song song**, sau đó hội tụ `aggregate`.
4. `StateGraph(LegalState)` rồi `graph.add_node(...)`, `graph.add_edge(...)` cho luồng tuyến tính; `graph.add_conditional_edges("check_routing", route_to_specialists, [...])` để rẽ nhánh theo từ khóa / fallback.

**Bước 3:** Vẽ graph

```python
# Thêm vào cuối file main.py
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

**Bài Tập 4.1:** Thêm agent mới

Tạo `privacy_agent` chuyên về GDPR và privacy law:

```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về luật bảo vệ dữ liệu cá nhân."""
    llm = get_llm()
    
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.
    
Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Hãy phân tích các vấn đề về privacy và GDPR (nếu có).
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

Thêm node này vào graph và kết nối với `aggregate_results`.

**Bài Tập 4.2:** Implement conditional routing

Sửa `check_routing` để chỉ gọi privacy_agent khi câu hỏi có từ khóa "data", "privacy", "gdpr":

```python
def check_routing(state: State) -> list[Send]:
    question_lower = state["question"].lower()
    tasks = []
    
    if any(kw in question_lower for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))
    
    if any(kw in question_lower for kw in ["compliance", "sec", "regulation"]):
        tasks.append(Send("compliance_agent", state))
    
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
        tasks.append(Send("privacy_agent", state))
    
    return tasks if tasks else [Send("aggregate_results", state)]
```

---

## Phần 5: Distributed A2A System (15 phút)

### Lý Thuyết

**A2A (Agent-to-Agent) Protocol:** Chuẩn giao tiếp giữa các agents qua HTTP.

**Khác biệt với Stage 4:**
- Mỗi agent là một service độc lập
- Giao tiếp qua HTTP thay vì in-process
- Dynamic discovery qua Registry
- Có thể scale từng agent riêng biệt

**Kiến trúc:**
```
Registry (10000) ← agents register on startup
    ↓
Customer Agent (10100) → Law Agent (10101)
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
            Tax Agent (10102)   Compliance Agent (10103)
```

### Thực Hành

**Bước 1:** Khởi động toàn bộ hệ thống

```bash
./start_all.sh
```

Trên **Windows (PowerShell)**, dùng script tương đương trong repo: `.\start_all.ps1` (hoặc chạy `bash ./start_all.sh` nếu có Git Bash).

Chờ ~10 giây để tất cả services khởi động.

**Bước 2:** Test hệ thống

```bash
uv run python test_client.py
```

Hoặc: `python test_client.py` (khi đã kích hoạt `.venv` / đúng interpreter).

**Lưu ý:** Phải gọi bằng `python ...`; gõ riêng `test_client.py` trong PowerShell thường **không** chạy script.

Mở 5 terminal tabs và xem logs của từng service:
- Registry: port 10000
- Customer Agent: port 10100
- Law Agent: port 10101
- Tax Agent: port 10102
- Compliance Agent: port 10103

**Bài Tập 5.1:** Trace request flow

Trong logs, tìm `trace_id` và theo dõi request đi qua các agents. Vẽ sequence diagram.

**Bài Tập 5.2:** Test dynamic discovery

1. Dừng Tax Agent (Ctrl+C)
2. Chạy lại `test_client.py`
3. Quan sát lỗi và cách hệ thống xử lý

**Bài Tập 5.3:** Modify agent behavior

Sửa `tax_agent/graph.py`, thay đổi system prompt để agent trả lời ngắn gọn hơn. Restart tax agent và test lại.

**Gợi ý trả lời / quan sát (5.1–5.2):**

- **5.1 Trace:** Trong log các service, tìm cùng một **`trace_id`** (hoặc id tương đương repo gắn vào request) để thấy luồng Customer → Law → Tax/Compliance. Vẽ sequence diagram: Client → Customer Agent → Law Agent → các specialist → tổng hợp trả lời.
- **5.2 Dynamic discovery:** Khi tắt Tax Agent, registry không còn endpoint hợp lệ cho task thuế — `test_client` có thể báo lỗi kết nối / task failed; quan sát message và phần hệ thống vẫn trả lời được những gì (fallback tùy triển khai).

---

## Phần 6: Tổng Kết & Mở Rộng (10 phút)

### So Sánh 5 Stages

| Stage | Pattern | Use Case | Complexity |
|---|---|---|---|
| 1 | Direct LLM | Câu hỏi đơn giản, không cần tools | ⭐ |
| 2 | LLM + Tools | Cần tra cứu data hoặc tính toán | ⭐⭐ |
| 3 | ReAct Agent | Tự động orchestration, multi-step | ⭐⭐⭐ |
| 4 | Multi-Agent | Nhiều domains, parallel processing | ⭐⭐⭐⭐ |
| 5 | Distributed A2A | Production, scalable, fault-tolerant | ⭐⭐⭐⭐⭐ |

### Câu Hỏi Ôn Tập

1. Khi nào nên dùng single agent thay vì multi-agent?
2. Ưu điểm của A2A protocol so với gRPC hoặc REST thông thường?
3. Làm thế nào để prevent infinite delegation loops trong A2A?
4. Tại sao cần Registry service? Có thể hardcode URLs không?

**Đáp án tham khảo:**

1. **Single agent** khi bài toán thuộc **một domain**, luồng **tuần tự** đủ, muốn **đơn giản / ít LLM call song song / latency thấp**. **Multi-agent** khi **tách domain rõ**, cần **song song**, **prompt hoặc chính sách khác nhau** từng chuyên gia, hoặc triển khai/tách team theo service.
2. A2A chuẩn hóa **agent card**, **task / delegation**, **discovery** (registry) cho hệ nhiều agent. gRPC/REST “thuần” linh hoạt nhưng phải **tự thiết kế** toàn bộ contract và semantics ủy quyền.
3. Giới hạn **độ sâu / số hop** delegation, **timeout**, danh sách **agent được phép gọi**, phát hiện **chu trình**, dùng **trace_id** + idempotency, và policy **dừng khi đủ thông tin** ở orchestrator.
4. Registry hỗ trợ **dynamic discovery** khi agent đổi host/port, scale, phiên bản. **Hardcode URL** vẫn được cho **dev / topology nhỏ cố định**, nhưng kém linh hoạt khi môi trường thay đổi.

### Bài Tập Nâng Cao (Tự Học)

**Challenge 1:** Thêm memory/conversation history

Implement conversation memory để agent nhớ các câu hỏi trước đó.

**Challenge 2:** Add authentication

Thêm API key authentication cho các A2A endpoints.

**Challenge 3:** Implement retry logic

Khi một agent fail, tự động retry với exponential backoff.

**Challenge 4:** Monitoring & Observability

Tích hợp LangSmith hoặc Prometheus để monitor agent performance.

---

## Tài Liệu Tham Khảo

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [A2A Protocol Spec](https://github.com/google/A2A)
- [OpenRouter API](https://openrouter.ai/docs)
- Architecture diagrams: `docs/*.svg`

## Hỗ Trợ

Nếu gặp vấn đề:
1. Check `.env` file có đúng API key không
2. Đảm bảo tất cả ports (10000-10103) không bị chiếm
3. Xem logs trong terminal để debug
4. Đọc error messages cẩn thận — thường có hint rõ ràng

---

**Chúc các bạn học tốt! 🚀**
