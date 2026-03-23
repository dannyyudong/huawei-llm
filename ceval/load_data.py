from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import os

# 📚 所有初高中科目列表
JUNIOR_HIGH_SUBJECTS = [
    "middle_school_biology",
    "middle_school_chemistry", 
    "middle_school_geography",
    "middle_school_history",
    "middle_school_mathematics",
    "middle_school_physics",
    "middle_school_politics"
]

SENIOR_HIGH_SUBJECTS = [
    "high_school_biology",
    "high_school_chemistry",
    "high_school_chinese",
    "high_school_geography",
    "high_school_history",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_politics"
]

ALL_SUBJECTS = JUNIOR_HIGH_SUBJECTS + SENIOR_HIGH_SUBJECTS

# 🎯 配置参数
SPLIT = "test"  # 可选: "dev" / "val" / "test"
SAVE_DIR = "ceval_data"  # 本地保存路径
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔄 批量加载并保存
results = {}
for subject in tqdm(ALL_SUBJECTS, desc="Loading C-Eval subjects"):
    try:
        # 加载单个科目数据集
        dataset = load_dataset("ceval/ceval-exam", name=subject)
        
        # 提取指定split（如test集）
        df = dataset[SPLIT].to_pandas()
        df['subject'] = subject  # 添加科目列便于区分
        results[subject] = df
        
        # 可选：保存为CSV文件
        save_path = os.path.join(SAVE_DIR, f"{subject}_{SPLIT}.csv")
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
    except Exception as e:
        print(f"❌ 加载 {subject} 失败: {e}")
        continue

print(f"\n✅ 成功加载 {len(results)} 个科目，数据已保存至 '{SAVE_DIR}' 目录")