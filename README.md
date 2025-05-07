---
title: UnknownDetectionSystem
emoji: 🐠
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
---

# Hệ Thống Phát Hiện Người Lạ Xâm Nhập

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

Hệ thống phát hiện người lạ xâm nhập sử dụng mạng neural tích chập (CNN) với công nghệ nhận diện khuôn mặt tiên tiến.

## 📝 Thông tin

| Chi tiết | Giá trị |
|---------|--------|
| **Tác giả** | Giáp Văn Tài |
| **MSSV** | 63.CNTT-CLC |
| **Đơn vị** | Trường Đại học Nha Trang |
| **Đề tài** | Đồ án tốt nghiệp - Xây dựng hệ thống phát hiện người lạ xâm nhập sử dụng mạng neural tích chập |

## 📋 Mục lục

- [Hệ Thống Phát Hiện Người Lạ Xâm Nhập](#hệ-thống-phát-hiện-người-lạ-xâm-nhập)
  - [📝 Thông tin](#-thông-tin)
  - [📋 Mục lục](#-mục-lục)
  - [🔍 Giới thiệu](#-giới-thiệu)
  - [✨ Tính năng chính](#-tính-năng-chính)
  - [💻 Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
  - [🚀 Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt)
    - [Cài đặt với Docker](#cài-đặt-với-docker)
- [Tạo image Docker](#tạo-image-docker)
- [Chạy container](#chạy-container)

## 🔍 Giới thiệu

Hệ thống phát hiện người lạ xâm nhập sử dụng công nghệ nhận diện khuôn mặt kết hợp với mạng neural tích chập (CNN). Dự án này cung cấp giải pháp bảo mật thông minh với khả năng phát hiện, nhận diện người quen, phát hiện người lạ và gửi thông báo khi có xâm nhập không mong muốn.

## ✨ Tính năng chính

- **Phát hiện khuôn mặt:** Sử dụng mô hình SCRFD để phát hiện khuôn mặt trong video stream
- **Chống giả mạo khuôn mặt:** Phát hiện khuôn mặt giả (ảnh, video) với hệ thống anti-spoofing
- **Phát hiện khẩu trang:** Nhận diện người đeo khẩu trang và điều chỉnh ngưỡng nhận diện phù hợp
- **Nhận diện khuôn mặt:** Sử dụng mô hình ArcFace để nhận diện danh tính
- **Theo dõi khuôn mặt:** Theo dõi chuyển động của khuôn mặt qua các frame
- **Thông báo xâm nhập:** Gửi thông báo tức thì qua email khi phát hiện người lạ
- **Giao diện Web:** Giao diện trực quan hiển thị video stream và danh sách người được nhận diện
- **Theo dõi danh tính:** Hệ thống voting để ổn định kết quả nhận diện
- **Tự động dọn dẹp:** Xóa ảnh cũ và dữ liệu tạm để tiết kiệm bộ nhớ

## 💻 Yêu cầu hệ thống

- Python 3.10
- CUDA hỗ trợ (để tăng tốc) hoặc CPU
- Webcam hoặc camera IP
- Kết nối internet (cho thông báo email)

## 🚀 Hướng dẫn cài đặt

### Cài đặt với Docker

```bash
# Tạo image Docker
docker build -t unknown-detection-system .

# Chạy container
docker run -p 7860:7860 unknown-detection-system