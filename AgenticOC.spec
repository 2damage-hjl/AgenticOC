# -*- mode: python ; coding: utf-8 -*-
"""
AgenticOC PyInstaller Spec
用法:  在 ai/ 目录下运行  pyinstaller AgenticOC.spec
输出:  dist/AgenticOC-Server/  (整个文件夹就是最终可分发目录)
"""
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# ---- 基础路径 ----
SRC = os.path.abspath('.')

# ---- 需要打包的数据文件 ----
datas = []

# 1. config.json (用户需要编辑的文件, 放在 exe 同级)
#    注意: 不打包到 exe 内部, 而是在启动时从 exe 同级目录读取

# 2. NPC 人设 JSON
npc_json_dir = os.path.join(SRC, 'prompt_construction', 'npc')
if os.path.exists(npc_json_dir):
    datas.append((npc_json_dir, 'prompt_construction/npc'))

# 3. Few-shot 数据 (LanceDB)
lancedb_dir = os.path.join(SRC, 'prompt_construction', 'data')
if os.path.exists(lancedb_dir):
    datas.append((lancedb_dir, 'prompt_construction/data'))

# 4. Jinja2 模板
templates_dir = os.path.join(SRC, 'prompt_construction', 'templates')
if os.path.exists(templates_dir):
    datas.append((templates_dir, 'prompt_construction/templates'))

# 5. Prompt 模块
prompt_dir = os.path.join(SRC, 'prompt_construction', 'prompt')
if os.path.exists(prompt_dir):
    datas.append((prompt_dir, 'prompt_construction/prompt'))

# 6. Retrieval 模块
retrieval_dir = os.path.join(SRC, 'prompt_construction', 'retrieval')
if os.path.exists(retrieval_dir):
    datas.append((retrieval_dir, 'prompt_construction/retrieval'))

# 7. Utils 模块
utils_dir = os.path.join(SRC, 'prompt_construction', 'utils')
if os.path.exists(utils_dir):
    datas.append((utils_dir, 'prompt_construction/utils'))

# 8. Configs (YAML)
configs_dir = os.path.join(SRC, 'prompt_construction', 'configs')
if os.path.exists(configs_dir):
    datas.append((configs_dir, 'prompt_construction/configs'))

# 9. Scripts
scripts_dir = os.path.join(SRC, 'memory', 'scripts')
if os.path.exists(scripts_dir):
    datas.append((scripts_dir, 'memory/scripts'))

# 10. Web templates (OC Builder UI)
web_templates_dir = os.path.join(SRC, 'web', 'templates')
if os.path.exists(web_templates_dir):
    datas.append((web_templates_dir, 'web/templates'))

# 11. sentence-transformers 模型缓存 (如果本地有)
# BGE-M3 会在首次运行时自动下载到用户目录, 不需要打包

# 11. 收集隐式数据文件
for pkg in ['sentence_transformers', 'chromadb', 'langchain', 'langchain_chroma',
            'langchain_openai', 'langchain_core', 'langchain_google_genai',
            'pydantic', 'yaml', 'tqdm', 'huggingface_hub']:
    try:
        datas += collect_data_files(pkg, include_py_files=False)
    except Exception:
        pass

# ---- Hidden imports ----
hiddenimports = [
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    'uvicorn.lifespan',
    'fastapi',
    'pydantic',
    'langchain',
    'langchain_openai',
    'langchain_chroma',
    'langchain_google_genai',
    'langchain_core',
    'chromadb',
    'sentence_transformers',
    'huggingface_hub',
    'numpy',
    'yaml',
    'tqdm',
    'dotenv',
    'flask',
    'anyio',
    'httpcore',
    'httpx',
    'sniffio',
]

# 自动收集子模块
for pkg in ['langchain', 'langchain_core', 'langchain_openai', 'langchain_chroma',
            'chromadb', 'sentence_transformers', 'pydantic']:
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

a = Analysis(
    [os.path.join(SRC, 'server.py')],
    pathex=[SRC],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'pandas', 'PIL', 'IPython',
              'notebook', 'pytest', 'setuptools', 'pip'],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AgenticOC-Server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name='AgenticOC-Server',
)
