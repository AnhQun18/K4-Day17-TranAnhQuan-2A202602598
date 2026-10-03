# Báo cáo: Memory Systems for AI Agent

## Kết quả tái lập offline

Chạy từ root repo: `python src/benchmark.py`. Dataset và rubric không chỉnh sửa.
Mỗi suite và mỗi agent dùng thư mục state tạm riêng; recall được hỏi ở thread mới.
Expected strings chỉ dùng để chấm sau khi agent trả lời, không đưa vào memory hay prompt.

| Bộ dữ liệu | Agent | Token phản hồi | Prompt tokens | Recall | Quality | Memory tăng (bytes) | Compactions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | Baseline | 819 | 10606 | 0.00 | 0.00 | 0 | 0 |
| Standard | Advanced | 828 | 16725 | 1.00 | 1.00 | 309 | 0 |
| Stress | Baseline | 191 | 21244 | 0.00 | 0.00 | 0 | 0 |
| Stress | Advanced | 204 | 10707 | 1.00 | 1.00 | 207 | 8 |

Token offline được ước lượng bằng `len(text.strip()) // 4`, không phải usage API.
Recall là trung bình điểm 0 / 0.5 / 1 theo chuỗi kỳ vọng; không phải xác suất nhớ đúng
trên câu hỏi bất kỳ. Quality = coverage × brevity, dùng cùng chuỗi kỳ vọng,
ngân sách ký tự `max(160, 2 × tổng độ dài expected + 80)`; không phải LLM judge.
Hai điểm 1.00 chứng minh bộ quy tắc xử lý được dataset này, không chứng minh tổng quát hóa.

## Ba lớp memory và so sánh công bằng

Baseline giữ toàn bộ messages theo thread; chỉ trích lại facts từ user messages của
thread đang hỏi. Thread mới không có facts, nên trả lời chưa biết. Baseline và Advanced
sử dụng cùng bộ trích và hàm trả lời offline; khác biệt nằm ở ngữ cảnh sẵn có.
Nhờ vậy điểm Baseline không còn đến từ việc echo câu hỏi có chứa đáp án.

Advanced có short-term messages, profile `User.md` theo user và summary theo thread.
Profile tồn tại sau khi khởi tạo lại agent; summary chỉ tồn tại trong process.
Profile gồm name/location/profession/interest/drink/style/food/pet khi người dùng khai báo.
Những nội dung khác vẫn ở messages/summary, không tự động ghi toàn bộ vào profile.

## Trade-off token và compact

Standard chưa kích hoạt compact. Advanced tăng prompt tokens khoảng **57.7%** so với
Baseline vì mỗi lượt thêm profile vào lịch sử. Token phản hồi cũng tăng nhẹ vì câu trả lời
recall có facts thay vì báo chưa biết. Persistent memory có lợi về recall nhưng có chi phí.

Stress compact **8 lần**, giảm prompt tokens khoảng **49.6%**. Baseline kéo lại toàn bộ
messages ở mỗi lượt, khiến tổng lượng context tăng gần bậc hai khi độ dài lượt tương đương.
Advanced giữ messages gần nhất và summary giới hạn **1.600 ký tự**. Summary trước được
hợp nhất với phần vừa compact; các declared facts được cập nhật và giữ trước các snippets.
Ngưỡng trigger tính cả summary và messages. Nếu một tin gần nhất rất dài, context vẫn có
thể vượt ngưỡng: ngưỡng trigger không phải hard cap cho toàn bộ prompt.

Summary là phép nén mất thông tin: topic snippets chỉ giữ một phần mỗi user message;
những snippets cũ bị loại khi hết budget. Cơ chế này giữ facts đã nhận diện qua nhiều lần
compact, nhưng không bảo đảm giữ mọi chi tiết hoặc abstraction của đoạn news. Benchmark
recall hiện chỉ hỏi persistent facts; nó **không kiểm chứng khả năng hiểu lại bốn tin news**.
Không có LLM summarization call nên không có token API phụ cho việc compact.

## Bonus: entity extraction và conflict handling

Bộ trích nhận các khai báo tiếng Việt và ghi theo field. Correction mới thay dòng cũ cùng
key, không lưu đồng thời nghề/nơi ở cũ và mới. Ví dụ `giờ mình đang ở Huế chứ không còn ở
Đà Nẵng` cập nhật location thành Huế; `giờ chuyển sang MLOps engineer` thay nghề cũ.

Bỏ qua câu hỏi, giá trị nghi vấn như `gì`, điều kiện `nếu`, câu đùa và clause phủ định;
không suy ra nơi ở từ `đi họp tại Hà Nội`. Các test dùng tên/món ăn khác dataset, kiểm tra
profile không bị ghi đè từ yêu cầu `Bạn nhớ giúp mình là ...` hoặc câu hỏi gián tiếp.
Đây là bộ lọc theo quy tắc, không phải confidence score đã hiệu chuẩn hay memory decay.

Một field chỉ có một giá trị hiện tại nên correction không làm tăng số dòng vô hạn.
Giới hạn còn lại: giá trị interest có thể dài, thiếu provenance/thời điểm, rule có thể bỏ sót
paraphrase hoặc lưu sai câu nhiều nghĩa. Nên review `User.md`, thêm provenance và giới hạn
độ dài field khi triển khai thực tế. Không khẳng định đã giải quyết mọi trường hợp phủ định.

## Kiểm chứng và chế độ live

Tests kiểm tra read/write/edit, cross-session và restart recall, baseline within-thread,
correction, nhiễu, user isolation, summary nhiều lần/bounded, prompt reduction, integration
với dataset standard, và nhánh live qua model giả (history, profile, usage, lỗi thiếu key/API).
Các checks này chạy offline và không tiêu thụ API key.

`python src/chat.py --live --user quan` gọi model LangChain thật; `--message` gọi một lượt.
`python src/benchmark.py --live` chạy cùng dataset với API thật. Cả hai agent dùng cùng
provider/model và chỉ dẫn nền; Advanced bổ sung profile và summary. Live lấy token từ
usage metadata nếu có, thiếu metadata thì ước lượng. Quality live vẫn là heuristic,
`judge_model` chỉ là cấu hình dành cho mở rộng và chưa được gọi.

Chưa chạy benchmark bằng dịch vụ LLM thật trong lần hoàn thiện này. Test model giả
xác minh luồng code, không xác minh key, quyền truy cập model, quota hay mạng của provider.
Live trả lỗi trực tiếp, không fallback offline. Fact extraction và summary ở live vẫn theo
quy tắc để giữ phép so sánh dễ kiểm chứng. Không sử dụng LangGraph/tool agent.
