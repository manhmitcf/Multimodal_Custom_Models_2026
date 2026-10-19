# ==============================================================================
# HƯỚNG DẪN CHẠY 8 CẤU HÌNH KÊNH VIDEO (MARIMO ALL-IN-ONE ONE-SHOT SCRIPTS)
# Thư mục: run_presets/
# ==============================================================================

Mỗi file trong thư mục này là một file kịch bản HOÀN CHỈNH TỪ A ĐẾN Z (All-In-One):
Tự động làm mọi việc từ:
1. Đọc HF_TOKEN và đăng nhập Hugging Face.
2. Tải Dataset qua Git LFS.
3. Sắp xếp lại thư mục dataset (mv audio/audio, mv video/video).
4. Clone repo / Pull code mới nhất nhánh ablations_final/video_input_channels.
5. Cài đặt thư viện requirements.txt.
6. Chạy kiểm toán nhanh (--dry-run).
7. Khởi chạy huấn luyện nền nohup cho đúng cấu hình đó và tự động mở log theo dõi!

KHI BẬT MỘT MÁY CHỦ MARIMO MỚI:
Bạn chỉ cần mở đúng file preset bạn muốn chạy trên máy chủ đó, nhấn Ctrl+A, Ctrl+C và dán thẳng vào terminal Linux!

DANH SÁCH 8 FILE TRONG THƯ MỤC NÀY:
1. run_C3_rgb.txt               -> Chạy C3_rgb [0, 1, 2] (~2.613M params)
2. run_C4_rgb_mag.txt           -> Chạy C4_rgb_mag [0, 1, 2, 5] (~2.613M params)
3. run_C4_rgb_vorticity.txt     -> Chạy C4_rgb_vorticity [0, 1, 2, 6] (~2.613M params)
4. run_C5_rgb_flow.txt          -> Chạy C5_rgb_flow [0, 1, 2, 3, 4] (~2.614M params)
5. run_C5_rgb_scalars.txt       -> Chạy C5_rgb_scalars [0, 1, 2, 5, 6] (~2.614M params)
6. run_C6_rgb_flow_mag.txt      -> Chạy C6_rgb_flow_mag [0, 1, 2, 3, 4, 5] (~2.615M params)
7. run_C6_rgb_flow_vorticity.txt-> Chạy C6_rgb_flow_vorticity [0, 1, 2, 3, 4, 6] (~2.615M params)
8. run_C7_full.txt              -> Chạy C7_full [0..6] (~2.616M params)

LƯU Ý QUAN TRỌNG:
- Khi nhấn Ctrl+C lúc đang xem log, bạn chỉ thoát giao diện xem log, tiến trình huấn luyện vẫn chạy nền ổn định trên GPU!
- Muốn xem lại log bất cứ lúc nào: tail -n +1 -f main_<TÊN_PRESET>.log
- Khi huấn luyện xong 400 epochs, hệ thống tự động nén toàn bộ kết quả và upload lên repo Hugging Face:
  manhmitcf/Results_final_video_input_channels
- Sau khi upload xong, bạn có thể an tâm tắt máy chủ Marimo mà không lo mất mát dữ liệu!
