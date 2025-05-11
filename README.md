# Unknown Detection System

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

Hệ thống phát hiện người lạ xâm nhập sử dụng mạng neural tích chập (CNN) với công nghệ nhận diện khuôn mặt tiên tiến.

## 📝 Thông tin

| Chi tiết    | Giá trị                                                                                        |
| ----------- | ---------------------------------------------------------------------------------------------- |
| **Tác giả** | Giáp Văn Tài                                                                                   |
| **MSSV**    | 63.CNTT-CLC                                                                                    |
| **Đơn vị**  | Trường Đại học Nha Trang                                                                       |
| **Đề tài**  | Đồ án tốt nghiệp - Xây dựng hệ thống phát hiện người lạ xâm nhập sử dụng mạng neural tích chập |

## 📋 Mục lục

* [Giới thiệu](#giới-thiệu)
* [Tính năng chính](#tính-năng-chính)
* [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
* [Hướng dẫn cài đặt](#hướng-dẫn-cài-đặt)
* [Chạy ứng dụng](#chạy-ứng-dụng)
* [Cấu trúc thư mục](#cấu-trúc-thư-mục)

---

## 🔍 Giới thiệu

Hệ thống phát hiện người lạ xâm nhập sử dụng công nghệ nhận diện khuôn mặt kết hợp với mạng neural tích chập (CNN). Dự án này cung cấp giải pháp bảo mật thông minh với khả năng phát hiện, nhận diện người quen, phát hiện người lạ và gửi thông báo khi có xâm nhập không mong muốn.

---

## ✨ Tính năng chính

* **Phát hiện khuôn mặt:** Sử dụng mô hình SCRFD để phát hiện khuôn mặt trong video stream.
* **Chống giả mạo khuôn mặt:** Phát hiện khuôn mặt giả (ảnh, video) với hệ thống anti-spoofing.
* **Phát hiện khẩu trang:** Nhận diện người đeo khẩu trang và điều chỉnh ngưỡng nhận diện phù hợp.
* **Nhận diện khuôn mặt:** Sử dụng mô hình ArcFace để nhận diện danh tính.
* **Theo dõi khuôn mặt:** Theo dõi chuyển động của khuôn mặt qua các frame.
* **Thông báo xâm nhập:** Gửi thông báo tức thì qua email khi phát hiện người lạ.
* **Giao diện Web:** Giao diện trực quan hiển thị video stream và danh sách người được nhận diện.
* **Theo dõi danh tính:** Hệ thống voting để ổn định kết quả nhận diện.
* **Tự động dọn dẹp:** Xóa ảnh cũ và dữ liệu tạm để tiết kiệm bộ nhớ.

---

## 💻 Yêu cầu hệ thống

* **Python:** Phiên bản 3.10
* **CUDA:** Khuyến khích nếu dùng GPU để tăng tốc (có thể chạy trên CPU nếu không có GPU).
* **Miniconda hoặc Anaconda:** Để quản lý môi trường ảo.
* **Webcam hoặc Camera IP:** Để thu nhận hình ảnh.
* **Kết nối Internet:** Để gửi thông báo email.

---

## 🚀 Hướng dẫn cài đặt

### 1️⃣ **Cài đặt Miniconda hoặc Anaconda**

* Tải Miniconda tại: [Miniconda Download](https://docs.conda.io/en/latest/miniconda.html)
* Cài đặt theo hướng dẫn trên trang chủ.

### 2️⃣ **Tạo môi trường ảo với Conda**

```bash
conda create -n unknown_detection_system python=3.10
```

### 3️⃣ **Kích hoạt môi trường**

```bash
conda activate unknown_detection_system
```

### 4️⃣ **Cài đặt các thư viện**

Chạy script install.bat để tự động cài đặt các dependencies:

```bash
install.bat
```

---

## 🏃 **Chạy ứng dụng**

### 1️⃣ **Khởi động server**

Sử dụng uvicorn để chạy ứng dụng:

```bash
uvicorn app:app --reload
```

### 2️⃣ **Truy cập giao diện web**

Mở trình duyệt và truy cập: [127.0.0.1:8000)

---
