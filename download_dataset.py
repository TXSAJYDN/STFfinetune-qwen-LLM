"""下载 ChatHaruhi-54K 角色扮演对话数据集"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

print("正在通过镜像下载 ChatHaruhi-54K 数据集...")
try:
    local_path = snapshot_download(
        repo_id="silk-road/ChatHaruhi-54K-Role-Playing-Dialogue",
        repo_type="dataset",
        local_dir="./data/ChatHaruhi-54K",
    )
    print(f"下载完成: {local_path}")
    for root, dirs, files in os.walk(local_path):
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            print(f"  {f} ({size / 1024:.1f} KB)")
except Exception as e:
    print(f"镜像下载失败: {e}")
    print("\n尝试使用 datasets 库直接加载...")
    from datasets import load_dataset
    ds = load_dataset("silk-road/ChatHaruhi-54K-Role-Playing-Dialogue", split="train[:5]")
    print(ds)
    print(ds.column_names)
    for i in range(min(3, len(ds))):
        print(f"\n[样本 {i+1}]")
        for k, v in ds[i].items():
            print(f"  {k}: {str(v)[:300]}")
