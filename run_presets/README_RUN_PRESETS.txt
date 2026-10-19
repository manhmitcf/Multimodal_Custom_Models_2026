# ==============================================================================
# HƯỚNG DẪN CHẠY 8 CẤU HÌNH KÊNH VIDEO BẰNG FILE SCRIPT SẴN (GIAI ĐOẠN 1)
# Thư mục: run_presets/
# ==============================================================================

Mỗi file trong thư mục này chứa đầy đủ câu lệnh để chạy duy nhất 1 cấu hình.
Bạn KHÔNG CẦN mở file train_config.json để sửa tay nữa! Chỉ cần mở file tương ứng,
copy toàn bộ nội dung và dán vào terminal của máy chủ GPU / Marimo.

DANH SÁCH 8 FILE SẴN CÓ:
1. run_C3_rgb.txt               -> Chạy preset 1: RGB [0, 1, 2] (~2.613M params)
2. run_C4_rgb_mag.txt           -> Chạy preset 2: RGB + |V| [0, 1, 2, 5] (~2.613M params)
3. run_C4_rgb_vorticity.txt     -> Chạy preset 3: RGB + omega [0, 1, 2, 6] (~2.613M params)
4. run_C5_rgb_flow.txt          -> Chạy preset 4: RGB + u + v [0, 1, 2, 3, 4] (~2.614M params)
5. run_C5_rgb_scalars.txt       -> Chạy preset 5: RGB + |V| + omega [0, 1, 2, 5, 6] (~2.614M params)
6. run_C6_rgb_flow_mag.txt      -> Chạy preset 6: RGB + u + v + |V| [0, 1, 2, 3, 4, 5] (~2.615M params)
7. run_C6_rgb_flow_vorticity.txt-> Chạy preset 7: RGB + u + v + omega [0, 1, 2, 3, 4, 6] (~2.615M params)
8. run_C7_full.txt              -> Chạy preset 8: Full 7 Kênh Động Học [0..6] (~2.616M params)

------------------------------------------------------------------------------
NẾU BẮT ĐẦU TRÊN MÁY CHỦ MỚI (LÀM 1 LẦN DUY NHẤT):
------------------------------------------------------------------------------
# 1. Tải và giải nén Dataset
cd /marimo && nohup git clone https://huggingface.co/datasets/manhmitcf/Fish_Feeding_Intensity_Dataset > /marimo/clone.log 2>&1 &
tail -n +1 -f /marimo/clone.log

# 2. Sắp xếp lại thư mục dataset
mv /marimo/Fish_Feeding_Intensity_Dataset/audio/audio/* /marimo/Fish_Feeding_Intensity_Dataset/audio/
mv /marimo/Fish_Feeding_Intensity_Dataset/video/video/* /marimo/Fish_Feeding_Intensity_Dataset/video/
rm -rf /marimo/Fish_Feeding_Intensity_Dataset/audio/audio
rm -rf /marimo/Fish_Feeding_Intensity_Dataset/video/video
find /marimo/Fish_Feeding_Intensity_Dataset -type f -iname "*.mp4" | wc -l
find /marimo/Fish_Feeding_Intensity_Dataset -type f -iname "*.wav" | wc -l

# 3. Clone hoặc Pull mã nguồn nhánh ablation
if [ ! -d "/marimo/Capstone_2026_Fish_Feeding_Intensity" ]; then
    git clone https://github.com/manhmitcf/Multimodal_Custom_Models_2026.git /marimo/Capstone_2026_Fish_Feeding_Intensity
fi
cd /marimo/Capstone_2026_Fish_Feeding_Intensity
git fetch origin
git checkout ablations_final/video_input_channels
git pull origin ablations_final/video_input_channels

cd /marimo/Capstone_2026_Fish_Feeding_Intensity/U_FFIA27K_multimodal
pip install --upgrade pip
pip install -r requirements.txt

------------------------------------------------------------------------------
CÁCH CHẠY BẤT KỲ PRESET NÀO:
------------------------------------------------------------------------------
Mở file `run_presets/run_<TÊN_PRESET>.txt` và dán thẳng vào terminal, hoặc chạy:
cd /marimo/Capstone_2026_Fish_Feeding_Intensity/U_FFIA27K_multimodal
nohup python main.py --preset <TÊN_PRESET> --temporal-mode sum > main_<TÊN_PRESET>.log 2>&1 &
tail -n +1 -f main_<TÊN_PRESET>.log
