<div align="center">
  <img src="logo.png" width="180" height="180" alt="RongheDraw Logo">
  <h1>RongheDraw 多模式绘图插件</h1>
  <p>支持 Flow/Generic/Gemini/Dreamina 四种 API 模式的 AstrBot 绘图插件</p>
  <p>
    <b>作者:</b> Antigravity &nbsp;|&nbsp;
    <b>版本:</b> 1.2.13
  </p>
</div>

---

## ✨ 功能特点

- 🔀 **四种 API 模式**：Flow / Generic / Gemini (4K) / Dreamina
- 🎨 **文生图 & 图生图**：支持多图输入
- 📐 **智能模型选择**：Flow 模式根据图片比例自动选择横版/竖版模型
- 🌐 **翻译功能**：Flow 模式支持中英翻译
- 👥 **权限管理**：白名单用户/群聊分级权限
- ⏳ **并发控制**：每模式同时只允许1个非白名单用户生成
- 🎲 **预设效果**：9个内置预设 + 自定义预设
- 🤖 **LLM 工具**：让 AI 调用绘图功能
- 🔁 **默认模式切换**：支持手动/定时切换无前缀默认模式（白名单/普通用户/LLM）
- 🛑 **重试可控**：可关闭生成重试，避免无效等待
- 🧠 **LLM 续画**：支持“上一张图片”缓存继续修改
- 📏 **LLM 分辨率**：支持 1K/2K/4K 输出控制（Generic/Gemini）
- 🧾 **LLM 限制**：提示词 < 900 字符、图片 <= 10 张
- 📦 **图片压缩**：自动压缩大图避免 API 错误
- 🔒 **隐私保护**：自动过滤 @ 用户的昵称和QQ号

---

## 🚀 快速开始

### 安装

1. 将 `astrbot_plugin_ronghedraw` 文件夹放入 AstrBot 的 `plugins` 目录
2. 安装依赖：`pip install -r requirements.txt`

> 💡 插件启动时会自动检测依赖，缺少时会输出提醒

### 配置

在 AstrBot 管理面板中配置：

1. **Flow 模式**：`flow_api_url`、`flow_api_key`、`flow_default_model`
2. **Generic 模式**：`generic_api_url`、`generic_api_keys`
3. **Gemini 模式**：`gemini_api_url`、`gemini_api_keys`、`gemini_resolution` (默认4K)
4. **Dreamina 模式**：`dreamina_api_url`、`dreamina_api_keys`、`dreamina_ratio`

Dreamina2api Fork：https://github.com/wangyingxuan383-ai/dreamina2api

---

## 📝 命令列表

### 文生图
| 命令 | 说明 |
|------|------|
| `#f文 <描述>` | Flow 模式文生图（自动横竖版） |
| `#o文 <描述>` | Generic 模式文生图 |
| `#g文 <描述>` | Gemini 模式文生图 (4K) |
| `#d文 <描述>` | Dreamina 模式文生图 |
| `#文生图 <描述>` | 默认模式 |

### 图生图
| 命令 | 说明 |
|------|------|
| `#f图 [图片]` | Flow 模式（自动横竖版） |
| `#o图 [图片]` | Generic 模式 |
| `#g图 [图片]` | Gemini 模式 (4K) |
| `#d图 [图片]` | Dreamina 模式 |
| `#图生图 [图片]` | 默认模式 |

### 随机预设
| 命令 | 说明 |
|------|------|
| `#f随机 [图片]` | Flow 模式随机 |
| `#o随机 [图片]` | Generic 模式随机 |
| `#g随机 [图片]` | Gemini 模式随机 |
| `#d随机 [图片]` | Dreamina 模式随机 |
| `#随机 [图片]` | 默认模式随机 |

> 注：随机预设使用配置中的 `prompt_list` 自定义预设

### 管理
| 命令 | 说明 |
|------|------|
| `#查询次数` | 查询今日剩余 |
| `#预设列表` | 查看自定义预设 |
| `#生图菜单` | 显示完整菜单 |
| `#生图帮助` | 显示帮助 |
| `#切换到 <f/o/g/d>` | 切换无前缀默认模式（白名单/普通用户/LLM） |
| `#f切换模型` | 切换 Flow 模型 |
| `#f翻译开关` | 翻译功能开关 |

---

## 👥 权限系统

| 用户类型 | 可用模式 | 次数限制 | 并发限制 |
|----------|----------|----------|----------|
| 白名单用户 | 全部 (f/o/g/d) | ❌ 无限制 | ❌ 无限制 |
| 白名单群聊 | 全部 (f/o/g/d) | ✅ 有限制 | ✅ 有限制 |
| 普通用户 | 仅 f 模式 | ✅ 有限制 | ✅ 每模式1个 |

---

## ⚙️ 配置说明

### Flow 模式配置
- `flow_api_url`: API 地址（OpenAI 格式）
- `flow_api_key`: API 密钥
- `flow_default_model`: 默认模型（横版，自动切换竖版）
- `flow_model_list`: 可切换的模型列表
- `flow_enable_translate`: 启用翻译
- `flow_use_proxy`: 启用代理

### Generic 模式配置
- `generic_api_url`: API 地址
- `generic_api_key`: API 密钥
- `generic_default_model`: 默认模型
- `generic_api_format`: API 格式 (`openai`/`gemini`)，默认为 `openai`
- `generic_resolution`: 分辨率 (1K/2K/4K)
- `generic_aspect_ratio`: 默认纵横比
- `generic_use_proxy`: 启用代理

### Gemini 模式配置
- `gemini_api_url`: Google API 地址
- `gemini_api_keys`: API Key 池
- `gemini_default_model`: 默认模型（gemini-3-pro-image-preview）
- `gemini_resolution`: 输出分辨率 (1K/2K/4K，默认4K)
- `gemini_use_proxy`: 启用代理（默认开启）

### Dreamina 模式配置
- `dreamina_api_url`: Dreamina2api 地址（/v1/images/generations）
- `dreamina_api_keys`: SessionID 列表
- `dreamina_default_model`: 默认模型（dreamina-4.5）
- `dreamina_ratio`: 输出比例（自动/固定比例）
- `dreamina_use_proxy`: 启用代理（默认关闭）

### LLM 配置
- `enable_llm_tool`: 启用 LLM 工具
- `llm_default_mode`: LLM 工具默认模式
- `llm_tool_use_group_limit`: LLM 是否使用群级统计
- `llm_group_daily_limit`: LLM 群每日限额
- `llm_last_image_ttl`: 上一次图片缓存 TTL(秒)
- `llm_last_image_max_entries`: 上一次图片缓存最大条数

### 其他配置
- `help_text`: `#生图帮助` 的显示内容（支持自定义）
- `enable_retry`: 启用/关闭生成重试
- `max_retries`: 最大重试次数（启用重试时生效）
- `retry_delay`: 重试基础间隔（指数退避）
- `default_mode_schedule`: 定时切换默认模式，格式 `HH:MM=模式`（可多条）

### 代理配置
- `proxy_url`: 代理地址（默认: http://172.17.0.1:7890）

### 自定义预设
- `prompt_list`: 预设列表，格式: `["触发词:提示词内容"]`

---

## 🤖 LLM 对话式绘图

本插件支持通过 AI 对话自然绘图，无需记忆指令。

### 使用示例

**文生图**：
```
用户：画一只可爱的猫
AI：好的，我来画~ [调用绘图工具]
```

**图生图（需要用户提供图片URL）**：
```
用户：把这张图改成二次元风格
AI：请提供图片链接
用户：https://example.com/image.jpg
AI：收到！[调用绘图工具，基于该图片]
```

> `image_urls` 也支持 `data:image/...;base64,...` 或 `base64://...` 形式。

**使用头像绘图**：
```
用户：用我的头像画个手办
AI：好哦~ [先获取用户头像URL，再调用绘图工具]
```

**分辨率控制（仅Generic/Gemini）**：
```
AI：调用 generate_image(prompt="...", resolution="2K")
```

**Dreamina比例（通过配置）**：
```
在插件配置中设置 dreamina_ratio 为 自动 / 1:1 / 16:9 等
```

**继续改图（使用上一张）**：
```
用户：继续修改一下刚才那张
AI：好的，继续优化~ [调用 generate_image(prompt="...", use_last_image=true)]
```

### 输入限制

- 提示词长度需 **少于900字符**
- 图片数量 **最多10张**

### 人格定制建议（简版）

工具描述已包含头像、继续改图、分辨率与输入限制等规则，系统提示词建议保持简洁：

```text
【绘图触发】
仅当用户明确要求画图/改图时调用绘图工具；连续相似请求需确认。

【后续互动】
出图后用角色口吻简短评价。
```

---

## 💬 问题反馈

有问题加群：1075544978

---

## 📜 更新日志

### v1.2.13 (2026-01-29)
- 修复：QQ 图片下载返回非图片内容时被误当作输入
- 修复：支持 file:// 本地路径图片读取（适配 get_image 返回）

### v1.2.12 (2026-01-29)
- 修复：下载图片为空时的兜底逻辑（补充校验与请求头）

### v1.2.11 (2026-01-29)
- 修复：启动报错（字符串转义导致语法错误）

### v1.2.10 (2026-01-29)
- 修复：兼容 raw_message/CQ 码解析，提高回复图片与直接图片识别率
- 增强：reply 场景可从本地聊天记录兜底读取图片

### v1.2.9 (2026-01-29)
- 修复：回复图片/直接图片在部分平台无法识别的问题（补充file_id与reply拉取）

### v1.2.8 (2026-01-26)
- 新增：`enable_retry` 配置，可主动关闭生成重试
- 新增：`default_mode_schedule` 定时切换默认模式（支持多条）

### v1.2.7 (2026-01-26)

**🐛 修复**
- **Dreamina 错误提示**: 直接展示 Dreamina2api 的错误信息，便于定位问题

### v1.2.6 (2026-01-26)

**⚡ 优化**
- **LLM Base64**: LLM 工具 image_urls 支持 dataURL/base64://
- **Dreamina2api 限制提示**: README 增加请求体大小提示

### v1.2.5 (2026-01-26)

**⚡ 优化**
- **Dreamina Base64**: 图生图直接使用 base64 图片，无需额外图片托管
- **Dreamina 默认关闭**: 增加防御性默认关闭逻辑（未配置时不启用）

### v1.2.4 (2026-01-26)

**🚀 新特性**
- **Dreamina 模式**：新增 d 前缀命令与 Dreamina2api 对接
- **Dreamina 比例**：支持自动/固定比例（无图默认 1:1）

### v1.2.1 (2026-01-15)

**🚀 新特性**
- **默认模式切换**: 新增 `#切换到` 一键切换白名单/普通用户/LLM默认模式
- **LLM 续画**: 支持“上一张图片”缓存继续修改（会话级）
- **LLM 分辨率**: `generate_image` 支持 `resolution=1K/2K/4K`

**⚡ 优化**
- **LLM 稳定性**: 增加缓存 TTL/条数限制，避免长期占用内存
- **输入限制**: 提示词 < 900 字符，图片 <= 10 张

**🐛 修复**
- **Generic-Gemini**: 修复超时对象未定义问题
- **Flow 流式解析**: 兼容 SSE 分片解析，降低漏取 URL

### v1.2.0 (2025-12-28)

**🚀 新特性**
- **Generic 模式增强**: 支持切换 OpenAI/Gemini API 格式，完美兼容 Gemini 模型
- **性能优化**: 实现 HTTP Session 连接池与复用，显著减少连接建立时间
- **智能重试**: 优化重试机制，自动避开 401/403 等不可重试错误

**⚡ 优化**
- **连接池**: 限制最大并发连接数，启用 DNS 缓存
- **API 兼容**: 修复 Gemini 格式的 URL 构建与认证头问题
- **错误提示**: 更加友好、准确的错误信息反馈

**🐛 修复**
- 修复所有命令的参数签名问题，增强框架兼容性
- 修复部分 API 调用的缩进问题
- 修复图片下载未利用连接池的问题

### v1.0.1
- 修复: Flow模式自动选择横版/竖版模型
- 修复: Gemini 4K分辨率输出
- 修复: @用户信息泄露问题
- 修复: Generic模式"Chunk too big"错误（添加图片压缩）
- 优化: 代理默认设置为 http://172.17.0.1:7890
- 优化: 命令重命名（生图菜单/生图帮助/预设列表）
- 优化: 启动时自动检测依赖
- 移除: 内置预设，改用配置自定义预设

### v1.0.0
- 首次发布

---

## 📄 许可证

MIT License
