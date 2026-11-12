"""全局配置 —— 改行为只看这一个文件。

- `LLM_MODELS` 是原始声明,归一化在 `infra/llm`。
- import 期不碰磁盘(建目录靠 `ensure_dir`)。
- 实测定死的实现细节(批大小、并发等)不进这里 —— 调错会静默污染基线。
- 最底座:任何层都能 import 它,它不 import 任何层。
"""

from pathlib import Path

from .common.env import csv_list, text

WEB_PORT: int = 9001

# ══════════════════════════════════════════════════════════════
# 一、策略参数
# ══════════════════════════════════════════════════════════════

MIN_STAR: int = 400

STAR_GROWTH_THRESHOLD: int = 1100

MAX_STAR: int = 12000

# 榜单黑名单:列在这里的仓库永不进榜(综合榜 / 新项目榜 / 对话里的排名都生效),
# 用来干掉 star 特别多但跟自己不相关的老仓库。写 owner/repo,大小写不敏感,下一次生成报告生效。
EXCLUDED_REPOS: frozenset[str] = frozenset(r.lower() for r in (
     "public-apis/public-apis",
))

# ── 窗口与数量 ──

GROWTH_CALC_DAYS: int = 7        # 增长统计窗口(天);实际跨度以锚点快照为准,本值是请求值
DAYS_SINCE_CREATED: int = 45     # 新项目判定阈值(天)
DESC_REFRESH_DAYS: int = 60      # LLM 描述刷新周期(天)
HOT_PROJECT_COUNT: int = 100     # 综合/关键词榜默认输出上限(有几个出几个)
HOT_NEW_PROJECT_COUNT: int = 12  # 新项目榜默认输出数量
SNAPSHOT_KEEP_DAYS: int = 35     # 快照保留天数(按日期截断,够覆盖月度窗口)

MAX_DYNAMIC_SEARCH_KEYWORDS: int = 30

# ── 最近爆发加成(仅综合/关键词榜):窗口总增长基础分之上乘一个「最近爆发强度」加成 ──
#     acceleration = (recent_growth / RECENT_GROWTH_DAYS) / (window_growth / window_days)
#     boost        = 1 + BURST_ALPHA * min(max(acceleration - 1, 0), BURST_CAP)

RECENT_GROWTH_DAYS: int = 4   # 「最近几天」窗口
BURST_ALPHA: float = 0.15     # 加成强度,越大则最近爆发对排名影响越大
BURST_CAP: float = 2.0        # acceleration-1 的封顶(boost 最高 1 + ALPHA*CAP)


# ══════════════════════════════════════════════════════════════
# 二、凭据
# ══════════════════════════════════════════════════════════════


def github_tokens() -> list[str]:
    """GitHub token 列表(逗号分隔)。未配置返回 []。做成函数是为了运行期能重读(中途补 token)。"""
    return csv_list("GITHUB_TOKENS")


def llm_key(env_name: str) -> str:
    """按环境变量名取 LLM key。空串表示该平台未配置,调用方应跳过它而不是发空 key 请求。"""
    return text(env_name)


def serverchan_sendkey() -> str:
    """Server酱推送 key。空串 = 关闭推送(周报生成后不推),不是错误。"""
    return text("SERVERCHAN_SENDKEY")


# ══════════════════════════════════════════════════════════════
# 三、LLM 平台目录
# ══════════════════════════════════════════════════════════════

LLM_MODELS: list[dict] = [
    {
        "id": "azure01",
        "label": "GPT-6-Sol",
        "backend": "azure",
        "url": "https://group03-8353-resource.openai.azure.com/openai/v1/chat/completions?api-version=preview",
        "model": "gpt-6-sol",
        "lite_model": "gpt-6-luna",
        "key_env": "LLM_A_KEY",
        "enabled": 1,
        "desc": "三组测试用",
    },
    {
        "id": "azure02",
        "label": "gpt-5.5",
        "backend": "azure",
        "url": "https://sub0228-project003003.openai.azure.com/openai/v1/chat/completions?api-version=preview",
        "model": "gpt-5.5",
        "lite_model": "gpt-5.6-terra",
        "key_env": "LLM_B_KEY",
        "enabled": 1,
        "desc": "三组专用",
    },
    {
        "id": "azure03",
        "label": "GPT-5.6-Terra",
        "backend": "foundry",
        # Foundry 项目端点的 OpenAI 兼容面:项目根地址后面接 /openai/v1,不带 api-version
        # (这条路隐式版本化)。认证是项目 key 走 Bearer,不是 Azure OpenAI 资源那套 api-key。
        "url": "https://group03-2471-resource.services.ai.azure.com/api/projects/group03-2471/openai/v1/chat/completions",
        # Azure 里这个字段匹配的是**部署名**,不必等于底层模型名;部署不存在会回 404。
        "model": "gpt-5.6-terra",
        "lite_model": "",
        "key_env": "LLM_F_KEY",
        "enabled": 0,
        "desc": "Foundry 三组",
    },
    {
        "id": "siliconflow",
        "label": "GLM-5.1(雷达)",
        "backend": "openai",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "model": "Pro/zai-org/GLM-5.2",
        "lite_model": "Qwen/Qwen3.6-35B-A3B,Qwen/Qwen3.5-35B-A3B",
        "key_env": "LLM_C_KEY",
        "enabled": 1,
        "desc": "开源雷达",
    },
    {
        "id": "aliyun01",
        "label": "GLM-5.2(阿里A)",
        "backend": "openai",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "glm-5.1",
        # 这个 key 只开了 GLM 系:所有 qwen 都是 403 Model.AccessDenied(名字没错,模型清单里
        # 有它 —— 百炼列的是整个目录,不是这个 key 的权限)。留空让它去借别家的子模型,
        # 否则选中这个平台时它自己的子模型会排第一,每次摘要先白撞一个 403 再回退
        "lite_model": "",
        "key_env": "LLM_D_KEY",
        "enabled": 1,
        "desc": "开源个人",
    },
    {
        "id": "aliyun02",
        "label": "Qwen3.7-Max(阿里B)",
        "backend": "openai",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen3.7-max",
        "lite_model": "qwen-flash",
        "key_env": "LLM_E_KEY",
        "enabled": 0,
        "desc": "阿里百炼三组",
    },
]


# ══════════════════════════════════════════════════════════════
# 四、搜索关键词词典
# ══════════════════════════════════════════════════════════════

SEARCH_KEYWORDS: dict[str, list[str]] = {
    # ─── AI 重点方向 ───
    "AI-Agent": [
        "ai agent", "agent framework", "multi-agent", "agent sdk",
        "coding agent", "browser-use", "computer-use", "web agent",
        "autonomous agent", "agent orchestration", "ai assistant",
        "tool calling", "deep research agent", "voice agent",
        "agent memory", "context engineering", "a2a protocol",
    ],
    "AI-MCP": [
        "mcp server", "mcp client", "model context protocol", "mcp sdk",
        "mcp tools", "mcp bridge", "mcp registry", "mcp integration",
    ],
    "AI-Skill-Prompt-Workflow": [
        "ai skill", "agent skill", "claude skills", "ai plugin",
        "prompt engineering", "prompt library", "ai workflow",
        "workflow automation", "langgraph", "spec driven development",
    ],
    "AI-CLI-DevTool": [
        "ai cli", "ai terminal", "ai devtool", "coding assistant",
        "code review ai", "code generation", "ai ide", "ai copilot",
        "ai coding", "claude code", "codex cli", "gemini cli",
    ],
    # 用嘴造应用:整块生成一个能跑的东西,是当前增长最快的品类
    "AI-Builder-VibeCoding": [
        "vibe coding", "ai app builder", "text to app", "prompt to app",
        "ai website builder", "no-code ai", "ai site generator",
        "ai generated app", "built with ai", "app generator",
        "text to website", "text to ui", "ui generator ai",
        "screenshot to code", "design to code", "figma to code",
        "text to game", "ai fullstack",
    ],
    # AI 写出来的成品:能直接用的产品/模板,而不是造它们的框架
    "AI-Product-App": [
        "ai saas", "saas boilerplate", "ai starter kit", "ai template",
        "ai chrome extension", "ai vscode extension", "ai bot",
        "ai note taking", "ai search engine", "ai presentation",
        "ai resume", "ai translator", "ai browser",
    ],
    # 喂给 AI 编码工具的规则/配置本身也在成仓库,而且涨得极快
    "AI-Coding-Config": [
        "cursorrules", "awesome cursor rules", "agents.md", "claude.md",
        "system prompt collection", "ai coding rules", "subagents",
    ],
    "AI-LLM-Core": [
        "large language model", "llm framework", "llm sdk",
        "open source llm", "llm api", "foundation model",
        "reasoning model", "small language model", "mixture of experts",
    ],
    "AI-RAG": [
        "rag", "retrieval augmented", "vector database",
        "embedding model", "semantic search", "document retrieval",
        "knowledge base", "graphrag", "agentic rag",
    ],
    "AI-Inference-Serving": [
        "llm inference", "llm serving", "vllm", "sglang",
        "quantization", "kv-cache", "speculative decoding",
        "model serving", "inference engine", "tensor parallel",
    ],
    "AI-Training-Finetune": [
        "fine-tuning", "instruction tuning", "lora", "peft",
        "rlhf", "dpo", "grpo", "reward model", "distillation",
        "agentic rl", "rl environment", "pretraining framework",
    ],
    "AI-Infra": [
        "triton kernel", "cuda kernel", "ml compiler",
        "distributed training", "model gateway", "gpu scheduling",
        "ml platform", "ai infrastructure",
    ],
    "AI-Multimodal": [
        "multimodal llm", "vision language model", "text to image",
        "text to video", "text to speech", "speech to text",
        "image generation", "video generation", "diffusion model",
        "world model", "image editing ai", "voice cloning", "music generation",
    ],
    "AI-Observability": [
        "llm observability", "ai guardrails", "llm evaluation",
        "ai monitoring", "llm tracing", "ai safety", "model evaluation",
    ],
    "AI-Data-Synthetic": [
        "synthetic data", "data augmentation ai", "ai dataset",
        "llm data", "rlhf data", "instruction tuning data",
    ],
    "AI-Edge-OnDevice": [
        "on-device llm", "edge ai", "mobile llm", "ollama",
        "llama.cpp", "mlx", "webgpu llm",
    ],
    # ─── 通用类别(保证覆盖面)───
    "Database": [
        "database", "sql database", "nosql", "time series database",
        "graph database", "document database",
        "postgresql", "mysql", "redis", "sqlite", "elasticsearch",
        "mongodb", "clickhouse", "vector search", "olap database",
    ],
    "Cloud-Native": [
        "kubernetes", "docker", "terraform", "serverless", "service mesh",
        "helm chart", "prometheus", "grafana", "argo workflow", "cilium",
        "container runtime", "istio", "envoy proxy", "knative",
    ],
    "Frontend": [
        "react", "vue", "svelte", "ui component", "nextjs", "tailwindcss",
        "nuxt", "vite", "astro", "shadcn", "electron app",
        "flutter", "react native", "typescript", "webgl",
    ],
    "Backend": [
        "web framework", "api framework", "microservice", "graphql server", "rpc framework",
        "fastapi", "django", "spring boot", "golang http", "nodejs framework",
        "gin", "express", "nestjs", "bun",
    ],
    "DevOps": [
        "ci cd pipeline", "monitoring", "infrastructure as code", "gitops",
        "ansible", "pulumi", "github actions", "jenkins", "gitlab ci",
        "argo cd", "flux", "terraform provider",
    ],
    "Security": [
        "security tool", "authentication", "vulnerability scanner",
        "waf", "penetration testing", "security scanner",
        "cve scanner", "secret scanner", "supply chain security",
        "prompt injection", "ai red teaming",
    ],
    "Data-Engineering": [
        "data pipeline", "etl", "stream processing", "feature store",
        "data lake", "data warehouse",
        "apache spark", "kafka", "flink", "airflow", "dbt",
        "duckdb", "polars", "pandas",
    ],
    "System-Tool": [
        "terminal tool", "cli tool", "shell", "wasm runtime",
        "terminal emulator", "file sync", "backup tool", "text editor",
        "neovim", "helix editor", "zsh", "fish shell", "tmux",
    ],
    "Programming-Language": [
        "programming language", "compiler", "language server",
        "rust", "golang", "zig", "lua",
        "python tooling", "typescript compiler",
    ],
    # ─── 新兴领域补充 ───
    "Web3-Blockchain": [
        "blockchain", "ethereum", "smart contract", "defi",
        "web3", "solidity", "bitcoin", "solana",
    ],
    "Game-Engine": [
        "game engine", "unity", "unreal engine", "godot",
        "game framework", "game dev", "3d engine", "physics engine",
    ],
    "Audio-Video": [
        "video processing", "audio processing", "ffmpeg",
        "video editor", "audio editor", "media player",
        "streaming media", "video codec", "audio codec",
    ],
    # ─── AI 应用层补充 ───
    "AI-App-Chatbot": [
        "chatbot", "ai chat", "llm app", "ai webui",
        "openai compatible", "ai gateway", "llm proxy", "ai companion",
    ],
    # ─── 经典 ML / 深度学习(非 LLM)───
    "ML-DeepLearning": [
        "pytorch", "jax", "scikit-learn",
        "deep learning", "machine learning framework", "onnx",
        "reinforcement learning",
    ],
    "Computer-Vision": [
        "computer vision", "object detection", "ocr", "opencv",
        "image segmentation", "pose estimation", "face recognition",
    ],
    # ─── 移动 / 桌面 / 跨端 ───
    "Mobile": [
        "android", "ios", "swift", "kotlin", "jetpack compose", "swiftui",
    ],
    "Desktop": [
        "tauri", "qt framework", "gtk", "wails", "desktop app",
    ],
    # ─── 自托管 / Homelab ───
    "Self-Hosted": [
        "self-hosted", "home assistant", "media server", "dashboard",
        "home automation",
    ],
    # ─── 爬虫 / 自动化 ───
    "Scraping-Automation": [
        "web scraping", "crawler", "playwright", "browser automation", "rpa",
    ],
    # ─── 代理 / 网关 / 网络 ───
    "Proxy-Gateway": [
        "nginx", "caddy", "traefik", "api gateway", "reverse proxy",
        "load balancer",
    ],
    # ─── 文档 / 静态站点 ───
    "Docs-StaticSite": [
        "static site generator", "hugo", "docusaurus", "mkdocs", "documentation",
    ],
    # ─── 测试 / QA ───
    "Testing-QA": [
        "testing framework", "cypress", "selenium", "e2e testing", "test automation",
    ],
    # ─── 机器人 / IoT / 嵌入式 ───
    "Robotics-IoT": [
        "ros", "esp32", "raspberry pi", "embedded", "firmware",
        "embodied ai", "humanoid robot", "vla model",
    ],
}


# ══════════════════════════════════════════════════════════════
# 五、路径
# ══════════════════════════════════════════════════════════════

PACKAGE_DIR = Path(__file__).resolve().parent

DATA_DIR = PACKAGE_DIR / "data"
REPORT_DIR = DATA_DIR / "report"
LOG_DIR = PACKAGE_DIR / "logs"
WEB_DIR = PACKAGE_DIR / "web"

DB_PATH = DATA_DIR / "Github_DB.json"
FAVORITES_PATH = DATA_DIR / "favorites.json"
SNAPSHOT_DIR = DATA_DIR / "snapshots"


def ensure_dir(path: Path) -> Path:
    """确保目录存在并返回它。**只在真正要写之前调用**,不在 import 期调用。"""
    path.mkdir(parents=True, exist_ok=True)
    return path
