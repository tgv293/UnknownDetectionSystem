TGV293 - Hệ thống an ninh nhận diện người lạ

Giới thiệu

Hệ thống này được thiết kế để nhận diện người lạ xâm nhập, hoạt động ngay cả khi đối tượng đeo khẩu trang. Hệ thống sử dụng:

Face Detection: RetinaFace

Mask Detection: MobileNetV2 (đã train với 3 class: incorrect_mask, with_mask, without_mask)
masked_face_recognition\models\mask_detector.h5

Face Recognition: masked_face_recognition\models\InceptionResNetV1_ArcFace.pt