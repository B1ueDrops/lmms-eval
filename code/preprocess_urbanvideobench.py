from tqdm import tqdm
import pandas as pd
from decord import VideoReader, cpu

question_categories = [
    "Progress Evaluation",
    "Counterfactual",
    "Landmark Position",
    "Action Generation",
    "Object Recall",
    "Causal",
    "Start/End Position",
    "Cognitive Map",
    "High-level Planning",
    "Scene Recall",
    "Duration",
    "Goal Detection",
    "Association Reasoning",
    "Trajectory Captioning",
    "Sequence Recall",
    "Proximity"
]

question_category_dict = {}

for question_category in question_categories:
    question_category_dict[question_category] = 0

vsi_mcq_path = '/root/autodl-tmp/UrbanVideo-Bench/MCQ.parquet'
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
    video_id = video_info['video_id']
    video_path = f'/root/autodl-tmp/UrbanVideo-Bench/videos/{video_id}'
    question_category = video_info['question_category']
    if len(question_categories) == 0:
        break
    if question_category in question_categories:
        val_keep_indices.append(index)
        question_category_dict[question_category] += 1
        if question_category_dict[question_category] == 3:
            question_categories.remove(question_category)

    duration = get_video_duration_decord(video_path)
    if duration <= 100:
        keep_indices.append(index)

# 根据保留索引筛选 DataFrame
filtered_df = df.loc[keep_indices]
val_filtered_df = df.loc[val_keep_indices]

print(len(filtered_df))
print(len(val_filtered_df))

# 保存回 parquet（可以覆盖原文件或另存）
filtered_df.to_parquet('/root/autodl-tmp/UrbanVideo-Bench/urbanvideobench_test.parquet')
val_filtered_df.to_parquet('/root/autodl-tmp/UrbanVideo-Bench/urbanvideobench_val.parquet')

print(f"✅ Done! Kept {len(filtered_df)} rows (<= 100s).")