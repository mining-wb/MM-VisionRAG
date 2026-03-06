# pytest 配置与公共 fixture
import sys
from pathlib import Path

# 保证从项目根运行时能 import src
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
