from tqdm import tqdm
import pandas as pd
from decord import VideoReader, cpu

question_types = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]
question_type_dict = {}

for question_type in question_types:
    question_type_dict[question_type] = 0

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
val_keep_indices = []

for index, row in tqdm(df.iterrows()):
    video_info = row.to_dict()
    dataset = video_info['dataset']
    scene_name = video_info['scene_name']
    question_type = video_info['question_type']
    if len(question_types) == 0:
        break
    if question_type in question_types:
        val_keep_indices.append(index)
        question_type_dict[question_type] += 1
        if question_type_dict[question_type] == 3:
            question_types.remove(question_type)
    video_path = f'/root/autodl-tmp/VSI-Bench/{dataset}/{scene_name}.mp4'
    duration = get_video_duration_decord(video_path)
    if duration <= 100:
        keep_indices.append(index)

# 根据保留索引筛选 DataFrame
filtered_df = df.loc[keep_indices]
val_filtered_df = df.loc[val_keep_indices]

print(len(filtered_df))
print(len(val_filtered_df))

# 保存回 parquet（可以覆盖原文件或另存）
filtered_df.to_parquet('/root/autodl-tmp/VSI-Bench/vsibench_test.parquet')
val_filtered_df.to_parquet('/root/autodl-tmp/VSI-Bench/vsibench_val.parquet')

print(f"✅ Done! Kept {len(filtered_df)} rows (<= 100s).")