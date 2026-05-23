"""
启动角色扮演大模型前端界面（FastAPI + Jinja2 模板）
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

if __name__ == "__main__":
    print("正在启动 Web 服务...")
    print("请在浏览器中打开 http://localhost:8000")
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
