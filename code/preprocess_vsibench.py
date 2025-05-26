import pandas as pd
from decord import VideoReader, cpu

vsi_mcq_path = '/root/autodl-tmp/VSI-Bench/test-00000-of-00001.parquet'
df = pd.read_parquet(vsi_mcq_path)

def get_video_duration_decord(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = total_frames / fps
    return duration

# 记录要保留的索引
keep_indices = []

for index, row in df.iterrows():
    video_info = row.to_dict()
    dataset = video_info['dataset']
    scene_name = video_info['scene_name']
    video_path = f'/root/autodl-tmp/VSI-Bench/{dataset}/{scene_name}.mp4'
    duration = get_video_duration_decord(video_path)
    if duration <= 100:
        keep_indices.append(index)

# 根据保留索引筛选 DataFrame
filtered_df = df.loc[keep_indices]

# 保存回 parquet（可以覆盖原文件或另存）
filtered_df.to_parquet('/root/autodl-tmp/VSI-Bench/test_filtered.parquet')

print(f"✅ Done! Kept {len(filtered_df)} rows (<= 100s).")