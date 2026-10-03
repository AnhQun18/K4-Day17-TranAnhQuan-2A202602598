# Memory Systems implementation

Các module đã có triển khai hoạt động:

- `config.py`: đọc `.env`, cấu hình sáu provider, đường dẫn và ngưỡng compact.
- `model_provider.py`: tạo LangChain chat model, gọi API, chuẩn hóa content và usage metadata; lỗi live được trả ra rõ ràng.
- `memory_store.py`: lưu `User.md`, trích entities tiếng Việt theo quy tắc thận trọng, thay fact cũ bằng correction mới; summary tích lũy giới hạn 1.600 ký tự.
- `agent_baseline.py`: chỉ giữ lịch sử trong thread; offline và live.
- `agent_advanced.py`: thread memory, persistent profile và compact memory; offline và live.
- `benchmark.py`: standard và stress suites với state tách riêng, mặc định offline; `--live` gọi API.
- `chat.py`: demo tương tác, `/new` đổi thread, `/quit` thoát, `--message` gọi một lượt.
- `test_agents.py`: kiểm chứng memory, correction/nhiễu, isolation, compact nhiều lần và nhánh live với model giả.

Cài dependencies từ `requirements.txt`; xem README gốc để chạy trên PowerShell.
Live dùng LangChain model adapter trực tiếp; không sử dụng LangGraph hoặc tools agent.
Fact extraction và summarization vẫn chạy bằng quy tắc ở cả hai chế độ, không gọi thêm LLM.
