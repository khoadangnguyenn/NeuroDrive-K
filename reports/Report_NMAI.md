# Báo cáo Bài tập lớn: Hệ thống Tự hành NeuroDrive-K
**Môn học**: Nhập môn Trí tuệ nhân tạo (NMAI)

---

## 1. Giới thiệu (Introduction)
- **Mục tiêu**: Xây dựng hệ thống ra quyết định cho xe tự hành dựa trên dữ liệu cảm biến và động học.
- **Phạm vi**: Xử lý dữ liệu, dự báo hành vi, đánh giá rủi ro và lập quy hoạch quỹ đạo.

## 2. Phân tích dữ liệu (EDA)
- **Tập dữ liệu**: Autonomous Driving Sensor-Kinematics Dataset.
- **Đặc trưng chính**: 
  - `ttc` (Time-to-Collision): Thời gian tới khi va chạm.
  - `speed`, `acceleration`: Trạng thái động học của xe.
  - `risk_probability`: Xác suất rủi ro do cảm biến đo được.
- **Trực quan hóa**: (Chèn các biểu đồ phân phối và tương quan từ Notebook).

## 3. Quy trình xử lý (Pipeline)

### 3.1. Trích xuất đặc trưng (Feature Engineering)
Chúng tôi áp dụng các công thức vật lý để tạo ra các đặc trưng mới:
- **Kinetic Danger**: Năng lượng động học kết hợp với khoảng cách an toàn.
- **Centrifugal Risk**: Rủi ro khi vào cua dựa trên gia tốc hướng tâm.
- **Braking Urgency**: Độ khẩn cấp khi phanh dựa trên TTC.

### 3.2. Nhận dạng hành vi (Perception ML)
Sử dụng mô hình Ensemble (Random Forest & Gradient Boosting) để phân loại hành vi xe:
- **Input**: Các đặc trưng đã chuẩn hóa.
- **Output**: Nhãn hành vi (Overtake, Lane Change, Yield, etc.).

### 3.3. Mô hình rủi ro Bayesian (Bayesian Modeling)
Cập nhật xác suất rủi ro của môi trường dựa trên các yếu tố không chắc chắn (thời tiết, bề mặt đường) bằng phương pháp Log-Odds Update.

### 3.4. Quy hoạch quỹ đạo (Path Planning)
Sử dụng thuật toán **A* Search** trên bản đồ lưới (Grid Map) 120x80:
- **Hàm chi phí**: Kết hợp chi phí rủi ro từ Perception và Bayesian.
- **Heuristic**: Khoảng cách Manhattan có trọng số ưu tiên làn đường.

## 4. Thí nghiệm và Kết quả (Experiments & Results)
- **Độ chính xác ML**: Đạt ~XX% trên tập test.
- **Phân tích SHAP**: (Giải thích các yếu tố ảnh hưởng nhất đến quyết định của AI).
- **Kết quả Simulation**: Xe có khả năng tránh vật cản và chọn làn đường an toàn trong các kịch bản phức tạp.

## 5. Kết luận (Conclusion)
Hệ thống đã chứng minh tính hiệu quả trong việc kết hợp giữa AI cổ điển (Search, Bayesian, Rules) và AI hiện đại (Machine Learning).

---
**Phụ lục**:
- Link GitHub: [Link của bạn]
- Link Google Colab: [Link của bạn]
