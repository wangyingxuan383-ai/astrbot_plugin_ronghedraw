"""
RongheDraw 多模式绘图插件
支持 Flow/Generic/Gemini 三种 API 模式
作者: Antigravity
版本: 1.2.8
"""
import asyncio
import base64
import hashlib
import io
import json
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import aiohttp

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import At, Image, Reply, Plain
try:
    from astrbot.core.message.message_event_result import MessageChain
except Exception:
    MessageChain = None
from astrbot.core.platform.astr_message_event import AstrMessageEvent
try:
    from astrbot.core.utils.io import download_image_by_url
except ImportError:
    download_image_by_url = None
from . import limit_manager


@register(
    "astrbot_plugin_ronghedraw",
    "Antigravity",
    "RongheDraw 多模式绘图插件 - 支持 Flow/Generic/Gemini/Dreamina 四种 API 模式",
    "1.2.8",
    "https://github.com/wangyingxuan383-ai/astrbot_plugin_ronghedraw",
)
class Main(Star):
    """RongheDraw 多模式绘图插件"""
    
    # ================== 初始化 ==================
    
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.prompt_map: Dict[str, str] = {}
        
        # Flow 模式状态
        self.flow_current_model_index = 0
        
        # Key 轮询索引
        self.generic_key_index = 0
        self.gemini_key_index = 0
        self.key_lock = asyncio.Lock()
        
        # 并发控制锁（每个模式同时只能有一个普通用户请求）
        self.mode_locks = {
            "flow": asyncio.Lock(),
            "generic": asyncio.Lock(),
            "gemini": asyncio.Lock(),
            "dreamina": asyncio.Lock()
        }
        
        # 加载预设
        self._load_prompt_map()
        
        # 内置预设映射
        self.builtin_presets = {
            "手办化": "Transform this image into a high-quality figurine/action figure style, maintaining the subject's features",
            "手办化2": "Convert to premium collectible figurine aesthetic with detailed sculpting",
            "Q版化": "Transform into cute chibi/Q-version style with big head and small body",
            "痛屋化": "Place the subject in an anime-decorated room with posters and figures",
            "痛车化": "Create an itasha car wrap design featuring the subject",
            "cos化": "Transform into a realistic cosplay photo style",
            "鬼图": "Create a spooky/horror style transformation",
            "第一视角": "Generate first-person perspective scene",
            "第三视角": "Generate third-person perspective scene",
        }
        
        # HTTP session (在initialize中创建)
        self._http_session = None
        
        # 跟踪pending的asyncio任务
        self.pending_tasks = set()

        # LLM工具“上一次图片”缓存（按会话）
        self.llm_last_image_cache: Dict[str, Dict[str, Any]] = {}
        
        # 检查依赖
        self._check_dependencies()
    
    def _check_dependencies(self):
        """检查依赖是否安装"""
        missing = []
        if PILImage is None:
            missing.append("Pillow")
        try:
            import aiohttp
        except ImportError:
            missing.append("aiohttp")
        
        if missing:
            logger.warning(f"[RongheDraw] WARNING: Missing dependencies: {', '.join(missing)}")
            logger.warning(f"[RongheDraw] Run: pip install {' '.join(missing)}")
    
    def _validate_config(self):
        """验证必需配置"""
        has_api = (self.config.get("flow_api_url") or 
                   self.config.get("generic_api_url") or 
                   self.config.get("gemini_api_url"))
        if not has_api:
            logger.warning("[RongheDraw] WARNING: No API URL configured, plugin functionality limited")

    def _apply_default_mode(self, mode: str, source: str = "manual") -> str:
        """应用默认模式切换（白名单/普通用户/LLM统一）"""
        self.config["llm_default_mode"] = mode
        self.config["normal_user_default_mode"] = mode
        self.config["default_mode"] = mode
        mode_name = {"flow": "Flow", "generic": "Generic", "gemini": "Gemini", "dreamina": "Dreamina"}[mode]
        if self.config.get("debug_mode", False):
            logger.info(f"[RongheDraw] 默认模式切换为 {mode_name} ({source})")
        return mode_name

    def _parse_schedule_item(self, item: str) -> Tuple[int, int, str] | None:
        """解析定时切换配置项: HH:MM=mode 或 HH:MM mode"""
        text = str(item).strip()
        if not text:
            return None
        if "=" in text:
            time_part, mode_part = text.split("=", 1)
        else:
            parts = re.split(r"\s+", text, maxsplit=1)
            if len(parts) < 2:
                return None
            time_part, mode_part = parts[0], parts[1]

        time_part = time_part.strip()
        mode_part = mode_part.strip()
        if not re.match(r"^\d{1,2}:\d{2}$", time_part):
            return None
        hour, minute = map(int, time_part.split(":", 1))
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            return None
        mode = self._parse_mode_token(mode_part)
        if not mode:
            return None
        return hour, minute, mode

    def _start_mode_schedule_tasks(self):
        """启动定时切换默认模式任务"""
        raw_list = self.config.get("default_mode_schedule", [])
        if isinstance(raw_list, dict) and "default" in raw_list:
            raw_list = raw_list["default"]
        if not isinstance(raw_list, list):
            raw_list = []
        if not raw_list:
            return

        valid = 0
        for item in raw_list:
            parsed = self._parse_schedule_item(item)
            if not parsed:
                logger.warning(f"[Schedule] 无效配置项: {item}")
                continue
            hour, minute, mode = parsed
            task = asyncio.create_task(self._mode_schedule_loop(hour, minute, mode, str(item)))
            self.pending_tasks.add(task)
            task.add_done_callback(self.pending_tasks.discard)
            valid += 1
        if valid > 0:
            logger.info(f"[RongheDraw] 已加载 {valid} 个默认模式定时任务")

    async def _mode_schedule_loop(self, hour: int, minute: int, mode: str, label: str):
        """定时循环切换默认模式"""
        while True:
            now = datetime.now()
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            delay = max(0, (target - now).total_seconds())
            if self.config.get("debug_mode", False):
                logger.info(f"[Schedule] {label} 下一次触发: {target.strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return

            try:
                enabled, mode_err = self._check_mode_enabled(mode)
                if not enabled:
                    logger.warning(f"[Schedule] 目标模式不可用，跳过 ({label}): {mode_err}")
                else:
                    mode_name = self._apply_default_mode(mode, source=f"定时 {label}")
                    logger.info(f"[Schedule] 已切换默认模式为 {mode_name} ({label})")
            except Exception as e:
                logger.warning(f"[Schedule] 定时切换失败 ({label}): {e}")

            # 防止极端时间漂移导致的短间隔重复触发
            await asyncio.sleep(1)
    
    async def initialize(self):
        """插件激活时调用，用于初始化资源"""
        # 创建带连接池/超时配置的HTTP session
        self._http_session = await self._get_session()
        
        # 验证配置
        self._validate_config()

        # 启动定时默认模式切换任务
        self._start_mode_schedule_tasks()
        
        logger.info('[RongheDraw] 插件已激活，资源已初始化')
    
    def _load_prompt_map(self):
        """加载预设提示词"""
        raw_list = self.config.get("prompt_list", [])
        if isinstance(raw_list, dict) and "default" in raw_list:
            raw_list = raw_list["default"]
        if not isinstance(raw_list, list):
            raw_list = []
        
        for item in raw_list:
            if isinstance(item, str) and ":" in item:
                key, val = item.split(":", 1)
                self.prompt_map[key.strip()] = val.strip()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建HTTP session (带连接池优化)"""
        if not self._http_session or self._http_session.closed:
            # 配置连接池：限制并发连接数，启用keepalive
            connector = aiohttp.TCPConnector(
                limit=10,              # 总连接数限制
                limit_per_host=5,      # 每个主机连接数限制
                ttl_dns_cache=300,     # DNS缓存5分钟
                enable_cleanup_closed=True
            )
            # 默认超时配置
            timeout_val = self.config.get("timeout", 120)
            timeout = aiohttp.ClientTimeout(
                total=timeout_val,      # 总超时
                connect=30,             # 连接超时
                sock_read=timeout_val   # 读取超时
            )
            self._http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
        return self._http_session
    
    # ================== 图片处理 ==================
    
    async def _download_image(self, url: str) -> bytes | None:
        """
        下载图片（参照gemini_image_ref实现）
        优先使用AstrBot内置工具download_image_by_url，它能正确处理QQ图片链接
        """
        # 方式1：尝试作为本地文件读取
        if Path(url).is_file():
            try:
                return Path(url).read_bytes()
            except Exception:
                pass
        
        # 方式2：使用AstrBot内置下载工具（推荐，能处理QQ图片链接）
        if download_image_by_url:
            try:
                path = await download_image_by_url(url)
                if path and Path(path).is_file():
                    data = Path(path).read_bytes()
                    if self.config.get("debug_mode", False):
                        logger.info(f"[download_image_by_url] 成功下载: {url[:60]}...")
                    return data
            except Exception as e:
                if self.config.get("debug_mode", False):
                    logger.warning(f"[download_image_by_url] 下载失败: {e}")
        
        # 方式3：回退到直接HTTP请求（公网URL）
        timeout = self.config.get("timeout", 120)
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        
        for i in range(3):
            try:
                session = await self._get_session()
                async with session.get(url, timeout=timeout_obj) as resp:
                    resp.raise_for_status()
                    return await resp.read()
            except Exception as e:
                if i < 2:
                    await asyncio.sleep(1)
                else:
                    logger.error(f"下载图片失败: {url[:60]}..., 错误: {e}")
        return None
    
    async def _get_avatar(self, user_id: str) -> bytes | None:
        """获取用户头像（参照gemini_image_ref使用q4.qlogo.cn）"""
        if not str(user_id).isdigit():
            return None
        # 使用q4.qlogo.cn，更稳定
        avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        return await self._download_image(avatar_url)
    
    def _extract_first_frame_sync(self, raw: bytes) -> bytes:
        """提取GIF第一帧（同步方法，供线程池调用）"""
        if PILImage is None:
            return raw
        try:
            img = PILImage.open(io.BytesIO(raw))
            if getattr(img, "is_animated", False):
                img.seek(0)
            img = img.convert("RGB")
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=85)
            return out.getvalue()
        except Exception:
            return raw
    
    async def _load_image_bytes(self, src: str) -> bytes | None:
        """从各种来源加载图片"""
        if Path(src).is_file():
            return Path(src).read_bytes()
        elif src.startswith("http"):
            raw = await self._download_image(src)
            if raw:
                # 使用线程池执行CPU密集型操作
                return await asyncio.to_thread(self._extract_first_frame_sync, raw)
        elif src.startswith("base64://"):
            try:
                return base64.b64decode(src[9:])
            except Exception:
                pass
        elif src.startswith("data:"):
            try:
                b64 = src.split(",", 1)[1]
                return base64.b64decode(b64)
            except Exception:
                pass
        return None
    
    async def get_images(self, event: AstrMessageEvent) -> List[bytes]:
        """
        获取消息中的所有图片（支持多图，合并回复图片、当前消息图片、@用户头像）
        参照gemini_image_ref实现At自动过滤：过滤触发机器人的@和回复自动@
        """
        images: List[bytes] = []
        chain = event.message_obj.message
        
        # 0. 预扫描：获取回复发送者ID和统计At次数（用于过滤自动@）
        reply_sender_id = None
        at_counts: dict = {}
        
        for seg in chain:
            if isinstance(seg, Reply):
                if hasattr(seg, 'sender_id') and seg.sender_id:
                    reply_sender_id = str(seg.sender_id)
            elif isinstance(seg, At):
                if hasattr(seg, 'qq') and seg.qq != "all":
                    uid = str(seg.qq)
                    at_counts[uid] = at_counts.get(uid, 0) + 1
        
        # 1. 回复链中的图片
        for seg in chain:
            if isinstance(seg, Reply) and hasattr(seg, 'chain') and seg.chain:
                for s in seg.chain:
                    if isinstance(s, Image):
                        if s.url and (img := await self._load_image_bytes(s.url)):
                            images.append(img)
                        elif hasattr(s, 'file') and s.file and (img := await self._load_image_bytes(s.file)):
                            images.append(img)
        
        # 2. 当前消息中的图片
        for seg in chain:
            if isinstance(seg, Image):
                if seg.url and (img := await self._load_image_bytes(seg.url)):
                    images.append(img)
                elif hasattr(seg, 'file') and seg.file and (img := await self._load_image_bytes(seg.file)):
                    images.append(img)
                elif hasattr(seg, 'base64') and seg.base64:
                    try:
                        images.append(base64.b64decode(seg.base64))
                    except Exception:
                        pass
        
        # 3. @用户头像（带自动过滤逻辑）
        self_id = str(event.get_self_id()).strip() if hasattr(event, 'get_self_id') else ""
        
        for seg in chain:
            if isinstance(seg, At) and hasattr(seg, 'qq') and seg.qq != "all":
                uid = str(seg.qq)
                
                # 过滤1：回复消息自动带的@（仅出现1次且是回复发送者）
                if reply_sender_id and uid == reply_sender_id:
                    if at_counts.get(uid, 0) == 1:
                        if self.config.get("debug_mode", False):
                            logger.debug(f"[get_images] 过滤回复自动@: {uid}")
                        continue
                
                # 过滤2：触发机器人的@（仅出现1次且是机器人自己）
                if self_id and uid == self_id:
                    if at_counts.get(uid, 0) == 1:
                        if self.config.get("debug_mode", False):
                            logger.debug(f"[get_images] 过滤机器人触发@: {uid}")
                        continue
                
                # 通过过滤，获取头像
                if avatar := await self._get_avatar(uid):
                    images.append(avatar)
        
        return images
    
    def _bytes_to_base64(self, data: bytes, mime: str = "image/jpeg") -> str:
        """转换为base64 URL格式"""
        b64 = base64.b64encode(data).decode()
        return f"data:{mime};base64,{b64}"
    

    def _create_image_from_bytes(self, data: bytes) -> Image:
        """从bytes创建Image组件（兼容不同AstrBot版本）"""
        try:
            # 优先使用fromBytes（如果存在）
            if hasattr(Image, 'fromBytes'):
                return Image.fromBytes(data)
        except Exception:
            pass
        # 回退到fromBase64
        b64 = base64.b64encode(data).decode()
        return Image.fromBase64(b64)
    
    def _compress_image(
        self,
        data: bytes,
        max_size: int = 2048,
        quality: int = 85,
        size_threshold: int = 2 * 1024 * 1024,
    ) -> bytes:
        """压缩大图，限制最大尺寸与体积"""
        if PILImage is None:
            return data
        try:
            img = PILImage.open(io.BytesIO(data))
            width, height = img.size
            if width <= max_size and height <= max_size and len(data) <= size_threshold:
                return data
            # 转换为RGB
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            # 限制最大尺寸
            if width > max_size or height > max_size:
                ratio = min(max_size / width, max_size / height)
                new_size = (int(width * ratio), int(height * ratio))
                img = img.resize(new_size, PILImage.Resampling.LANCZOS)
                logger.info(f"图片压缩: {width}x{height} -> {new_size[0]}x{new_size[1]}")
            # 保存为JPEG
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=quality, optimize=True)
            return out.getvalue()
        except Exception as e:
            logger.warning(f"图片压缩失败: {e}")
            return data

    async def _maybe_compress_images(self, images: List[bytes], mode: str) -> List[bytes]:
        """异步压缩大图，避免阻塞事件循环"""
        if not images or PILImage is None:
            return images
        if mode == "flow":
            max_size = 4096
            size_threshold = 6 * 1024 * 1024
        else:
            max_size = 2048
            size_threshold = 2 * 1024 * 1024
        result = []
        for img in images:
            try:
                result.append(
                    await asyncio.to_thread(
                        self._compress_image,
                        img,
                        max_size,
                        85,
                        size_threshold,
                    )
                )
            except Exception:
                result.append(img)
        return result

    def _clean_prompt(self, raw_text: str, event) -> str:
        """清理提示词，移除@用户信息（昵称和QQ号）"""
        text = raw_text
        # 移除 @ 开头的昵称和QQ号
        chain = event.message_obj.message if hasattr(event, 'message_obj') else []
        for seg in chain:
            if isinstance(seg, At):
                # 移除 @QQ号 格式
                text = re.sub(rf'@{seg.qq}\s*', '', text)
                # 移除昵称
                if hasattr(seg, 'name') and seg.name:
                    text = re.sub(rf'{re.escape(seg.name)}\s*', '', text)
        # 移除所有 @数字 格式
        text = re.sub(r'@\d+\s*', '', text)
        return text.strip()
    
    # ================== 翻译功能 (仅Flow模式) ==================
    
    async def _translate_to_english(self, text: str) -> str:
        """百度翻译中文到英文"""
        if not self.config.get("flow_enable_translate", False):
            return text
        
        appid = self.config.get("flow_baidu_appid", "")
        key = self.config.get("flow_baidu_key", "")
        if not appid or not key:
            return text
        
        # 检查是否包含中文
        if not any('\u4e00' <= c <= '\u9fff' for c in text):
            return text
        
        salt = str(random.randint(32768, 65536))
        sign = hashlib.md5((appid + text + salt + key).encode()).hexdigest()
        
        try:
            session = await self._get_session()
            params = {
                "q": text, "from": "zh", "to": "en",
                "appid": appid, "salt": salt, "sign": sign
            }
            async with session.get(
                "https://fanyi-api.baidu.com/api/trans/vip/translate",
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                data = await resp.json()
                if "trans_result" in data:
                    return data["trans_result"][0]["dst"]
        except Exception as e:
            logger.warning(f"翻译失败: {e}")
        
        return text
    
    # ================== API 调用 ==================
    
    async def _get_api_key(self, mode: str) -> str | None:
        """获取API密钥"""
        async with self.key_lock:
            if mode == "flow":
                return self.config.get("flow_api_key", "")
            elif mode == "dreamina":
                keys = self.config.get("dreamina_api_keys", [])
                if isinstance(keys, str):
                    keys = [k.strip() for k in keys.split(",") if k.strip()]
                elif isinstance(keys, list):
                    keys = [str(k).strip() for k in keys if str(k).strip()]
                if not keys:
                    return None
                return ",".join(keys)
            elif mode == "gemini":
                keys = self.config.get("gemini_api_keys", [])
                if not keys:
                    return None
                key = keys[self.gemini_key_index % len(keys)]
                self.gemini_key_index += 1
                return key
            else:  # generic
                keys = self.config.get("generic_api_keys", [])
                if not keys:
                    return None
                key = keys[self.generic_key_index % len(keys)]
                self.generic_key_index += 1
                return key
    
    async def _call_flow_api(self, images: List[bytes], prompt: str, model: str = None) -> Tuple[bool, Any]:
        """调用Flow API (OpenAI格式，支持翻译和多图，自动选择横竖版模型)"""
        api_url = self.config.get("flow_api_url", "")
        api_key = await self._get_api_key("flow")
        
        if not api_url or not api_key:
            return False, "Flow API 未配置"
        
        # 获取基础模型
        base_model = self.config.get("flow_default_model", "gemini-3.0-pro-image-landscape")
        
        # 根据图片比例自动选择模型
        if images and PILImage:
            try:
                img = PILImage.open(io.BytesIO(images[0]))
                width, height = img.size
                # 判断横版还是竖版
                if width >= height:
                    target_suffix = "landscape"
                else:
                    target_suffix = "portrait"
                # 替换模型后缀
                if "-landscape" in base_model or "-portrait" in base_model:
                    model = base_model.replace("-landscape", f"-{target_suffix}").replace("-portrait", f"-{target_suffix}")
                    # 确保最终只有一个后缀
                    model = model.replace(f"-{target_suffix}-{target_suffix}", f"-{target_suffix}")
                else:
                    model = base_model
                logger.info(f"[Flow] 图片 {width}x{height} -> 使用模型: {model}")
            except Exception as e:
                logger.warning(f"自动选择模型失败: {e}")
                model = base_model
        else:
            model = base_model
        
        # 翻译提示词
        translated = await self._translate_to_english(prompt)
        
        # 构建消息内容（不压缩图片）
        content = [{"type": "text", "text": translated}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._bytes_to_base64(img)}
            })
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": True
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        timeout = self.config.get("timeout", 120)
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        proxy = self.config.get("proxy_url") if self.config.get("flow_use_proxy") else None
        
        try:
            session = await self._get_session()
            async with session.post(
                api_url,
                json=payload,
                headers=headers,
                proxy=proxy,
                timeout=timeout_obj,
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return False, f"API错误 ({resp.status}): {text[:200]}"
                
                # 解析流式响应（兼容SSE分片）
                full_content = ""
                found_url = None
                buffer = ""
                done = False
                async for chunk in resp.content.iter_chunked(1024):
                    if not chunk:
                        continue
                    buffer += chunk.decode("utf-8", errors="ignore")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            done = True
                            break
                        try:
                            chunk_json = json.loads(data)
                            if "choices" in chunk_json and chunk_json["choices"]:
                                delta = chunk_json["choices"][0].get("delta", {})
                                if "content" in delta:
                                    full_content += delta["content"]
                                    if "http" in full_content:
                                        url_match = re.search(r'https?://[^\s<>")\\]]+', full_content)
                                        if url_match:
                                            found_url = url_match.group(0).rstrip(".,;:!?)")
                                            break
                        except Exception:
                            continue
                    if found_url or done:
                        break
                
                # 提取图片URL
                img_url = found_url
                if not img_url:
                    url_match = re.search(r'https?://[^\s<>")\\]]+', full_content)
                    if url_match:
                        img_url = url_match.group(0).rstrip(".,;:!?)")
                
                if img_url:
                    img_data = await self._download_image(img_url)
                    if img_data:
                        return True, img_data
                    return False, f"图片下载失败: {img_url}"
                
                return False, f"未找到图片URL: {full_content[:200]}"
        
        except asyncio.TimeoutError:
            return False, "请求超时"
        except Exception as e:
            return False, f"请求异常: {e}"
    
    async def _call_generic_api(self, images: List[bytes], prompt: str, resolution_override: str | None = None) -> Tuple[bool, Any]:
        """调用Generic API (支持OpenAI格式和Gemini原生格式)"""
        api_url = self.config.get("generic_api_url", "")
        api_key = await self._get_api_key("generic")
        model = self.config.get("generic_default_model", "nano-banana")
        api_format = self.config.get("generic_api_format", "openai")
        resolution = resolution_override or self.config.get("generic_resolution", "1K")
        aspect_ratio = self.config.get("generic_aspect_ratio", "自动")
        
        if not api_url or not api_key:
            return False, "Generic API 未配置"
        
        # 根据API格式选择不同的调用方式
        if api_format == "gemini":
            return await self._call_generic_gemini_format(api_url, api_key, model, images, prompt, resolution, aspect_ratio)
        else:
            return await self._call_generic_openai_format(api_url, api_key, model, images, prompt, resolution, aspect_ratio)
    
    async def _call_generic_openai_format(self, api_url: str, api_key: str, model: str, 
                                          images: List[bytes], prompt: str, 
                                          resolution: str, aspect_ratio: str) -> Tuple[bool, Any]:
        """Generic模式 - OpenAI格式调用"""
        # 构建消息
        if images:
            final_prompt = f"Re-imagine the attached image: {prompt}. Draw it directly."
            content = [{"type": "text", "text": final_prompt}]
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self._bytes_to_base64(img)}
                })
            messages = [{"role": "user", "content": content}]
        else:
            final_prompt = f"Generate a high quality image: {prompt}"
            messages = [{"role": "user", "content": final_prompt}]
        
        # 构建generationConfig
        image_config = {"imageSize": resolution}
        if not images and aspect_ratio and aspect_ratio != "自动":
            image_config["aspectRatio"] = aspect_ratio
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "modalities": ["image", "text"],
            "generationConfig": {"imageConfig": image_config}
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        timeout = self.config.get("timeout", 120)
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        proxy = self.config.get("proxy_url") if self.config.get("generic_use_proxy") else None
        
        if self.config.get("debug_mode", False):
            logger.info(f"[Generic-OpenAI] 请求: model={model}, resolution={resolution}, images={len(images)}")
        
        try:
            session = await self._get_session()
            async with session.post(api_url, json=payload, headers=headers,
                                    proxy=proxy, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return False, f"API错误 ({resp.status}): {text[:200]}"
                
                # 非流式响应直接解析JSON
                result = await resp.json()
                
                if self.config.get("debug_mode", False):
                    logger.info(f"[Generic] 响应: {str(result)[:500]}")
                
                # 提取内容
                full_content = ""
                if "choices" in result and result["choices"]:
                    message = result["choices"][0].get("message", {})
                    full_content = message.get("content", "")
                
                # 检查空响应
                if not full_content or not full_content.strip():
                    return False, "API返回空内容"
                
                # 提取base64或URL
                b64_match = re.search(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', full_content)
                if b64_match:
                    try:
                        b64 = b64_match.group(0).split(",")[1]
                        return True, base64.b64decode(b64)
                    except Exception:
                        pass
                
                url_match = re.search(r'https?://[^\s<>")\\]+', full_content)
                if url_match:
                    img_url = url_match.group(0).rstrip(".,;:!?)")
                    img_data = await self._download_image(img_url)
                    if img_data:
                        return True, img_data
                
                return False, f"响应中未找到图片: {full_content[:150]}"
        
        except asyncio.TimeoutError:
            return False, "请求超时"
        except Exception as e:
            return False, f"请求异常: {e}"
    
    async def _call_generic_gemini_format(self, api_url: str, api_key: str, model: str,
                                           images: List[bytes], prompt: str,
                                           resolution: str, aspect_ratio: str) -> Tuple[bool, Any]:
        """Generic模式 - Gemini原生格式调用"""
        # 构建URL - 需要清理OpenAI格式的路径，重构为Gemini格式
        # 例如: https://api.bltcy.ai/v1/chat/completions -> https://api.bltcy.ai/v1beta/models/{model}:generateContent
        base = api_url.rstrip("/")
        
        # 移除OpenAI格式的路径后缀
        for suffix in ["/chat/completions", "/completions", "/images/generations"]:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
        
        # 移除末尾的/v1，改用/v1beta
        if base.endswith("/v1"):
            base = base[:-3]
        
        base = base.rstrip("/")
        final_url = f"{base}/v1beta/models/{model}:generateContent"
        
        # 构建请求内容
        if images:
            final_prompt = f"Re-imagine the attached image: {prompt}. Draw it directly. Output high quality {resolution} resolution image."
        else:
            final_prompt = f"Generate a high quality {resolution} resolution image: {prompt}"
        
        parts = [{"text": final_prompt}]
        for img in images:
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": base64.b64encode(img).decode()
                }
            })
        
        # 构建生成配置
        use_image_config = self.config.get("gemini_use_image_config", True)
        generation_config = {
            "maxOutputTokens": 8192,
            "responseModalities": ["image", "text"]
        }
        if use_image_config:
            image_config = {"imageSize": resolution}
            if not images and aspect_ratio and aspect_ratio != "自动":
                image_config["aspectRatio"] = aspect_ratio
            generation_config["imageConfig"] = image_config
        else:
            if not images and aspect_ratio and aspect_ratio != "自动":
                generation_config["aspectRatio"] = aspect_ratio
        
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": generation_config,
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
            "Authorization": f"Bearer {api_key}"  # 兼容代理API
        }
        
        timeout = self.config.get("timeout", 120)
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        proxy = self.config.get("proxy_url") if self.config.get("generic_use_proxy") else None
        
        if self.config.get("debug_mode", False):
            logger.info(f"[Generic-Gemini] 请求: model={model}, resolution={resolution}, images={len(images)}")
        
        try:
            session = await self._get_session()
            async with session.post(final_url, json=payload, headers=headers,
                                    proxy=proxy, timeout=timeout_obj) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return False, f"API错误 ({resp.status}): {text[:200]}"
                
                data = await resp.json()
                
                if "error" in data:
                    return False, f"错误: {data['error']}"
                
                # 提取图片 - 首先尝试Gemini原生格式
                try:
                    all_images = []
                    for candidate in data.get("candidates", []):
                        for part in candidate.get("content", {}).get("parts", []):
                            if "inlineData" in part:
                                b64 = part["inlineData"]["data"]
                                all_images.append(base64.b64decode(b64))
                    
                    if all_images:
                        if self.config.get("debug_mode", False):
                            logger.info(f"[Generic-Gemini] 收到 {len(all_images)} 张图片，返回最后一张")
                        return True, all_images[-1]
                except Exception:
                    pass
                
                # 如果代理API返回OpenAI格式，尝试提取
                if "choices" in data:
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not content or not content.strip():
                        return False, "API返回空内容"
                    # 尝试从content中提取图片
                    b64_match = re.search(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', content)
                    if b64_match:
                        try:
                            return True, base64.b64decode(b64_match.group(0).split(",")[1])
                        except Exception:
                            pass
                
                return False, f"未找到图片: {str(data)[:200]}"
        
        except asyncio.TimeoutError:
            return False, "请求超时"
        except Exception as e:
            return False, f"请求异常: {e}"
    
    async def _call_gemini_api(self, images: List[bytes], prompt: str, resolution_override: str | None = None) -> Tuple[bool, Any]:
        """调用Gemini官方API（支持4K分辨率）"""
        base_url = self.config.get("gemini_api_url", "https://generativelanguage.googleapis.com")
        api_key = await self._get_api_key("gemini")
        model = self.config.get("gemini_default_model", "gemini-2.5-flash-preview-image")
        resolution = resolution_override or self.config.get("gemini_resolution", "4K")
        
        if not api_key:
            return False, "Gemini API Key 未配置"
        
        base = base_url.rstrip("/")
        if not base.endswith("v1beta"):
            base += "/v1beta"
        final_url = f"{base}/models/{model}:generateContent"
        
        # 构建请求 - 添加分辨率设置
        use_image_config = self.config.get("gemini_use_image_config", True)
        aspect_ratio = self.config.get("generic_aspect_ratio", "自动")
        
        if images:
            final_prompt = f"Re-imagine the attached image: {prompt}. Draw it directly. Output high quality {resolution} resolution image."
        else:
            final_prompt = f"Generate a high quality {resolution} resolution image: {prompt}"
        
        # 不压缩图片
        parts = [{"text": final_prompt}]
        for img in images:
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": base64.b64encode(img).decode()
                }
            })
        
        # 构建生成配置 - 可选imageConfig/旧版aspectRatio
        generation_config = {
            "maxOutputTokens": 8192,
            "responseModalities": ["image", "text"]
        }
        if use_image_config:
            generation_config["imageConfig"] = {"imageSize": resolution}
        else:
            if not images and aspect_ratio and aspect_ratio != "自动":
                generation_config["aspectRatio"] = aspect_ratio
        
        if self.config.get("debug_mode", False):
            logger.info(f"[Gemini] 请求: model={model}, resolution={resolution}, use_image_config={use_image_config}")
        
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": generation_config,
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        
        timeout = self.config.get("timeout", 120)
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        proxy = self.config.get("proxy_url") if self.config.get("gemini_use_proxy") else None
        
        try:
            session = await self._get_session()
            async with session.post(final_url, json=payload, headers=headers,
                                    proxy=proxy, timeout=timeout_obj) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return False, f"API错误 ({resp.status}): {text[:200]}"
                
                data = await resp.json()
                
                if "error" in data:
                    return False, f"错误: {data['error']}"
                
                # 提取图片 - 获取最后一张（第一张可能是1K预览，最后一张才是高分辨率）
                try:
                    all_images = []
                    for candidate in data.get("candidates", []):
                        for part in candidate.get("content", {}).get("parts", []):
                            if "inlineData" in part:
                                b64 = part["inlineData"]["data"]
                                all_images.append(base64.b64decode(b64))
                    
                    if all_images:
                        # 返回最后一张图片（高分辨率版本）
                        if self.config.get("debug_mode", False):
                            logger.info(f"[Gemini] 收到 {len(all_images)} 张图片，返回最后一张")
                        return True, all_images[-1]
                except Exception:
                    pass
                
                return False, f"未找到图片: {str(data)[:200]}"
        
        except asyncio.TimeoutError:
            return False, "请求超时"
        except Exception as e:
            return False, f"请求异常: {e}"

    async def _call_dreamina_api(self, images: List[bytes], prompt: str) -> Tuple[bool, Any]:
        """调用Dreamina API（支持base64图片）"""
        api_url = self.config.get("dreamina_api_url", "")
        api_key = await self._get_api_key("dreamina")
        model = self.config.get("dreamina_default_model", "dreamina-4.5")

        if not api_url or not api_key:
            return False, "Dreamina API 未配置"

        is_img2img = bool(images)
        ratio = self._get_dreamina_ratio(images)

        # 选择端点（文生图/图生图）
        endpoint = api_url
        if is_img2img:
            if endpoint.endswith("/generations"):
                endpoint = endpoint[: -len("/generations")] + "/compositions"
            elif endpoint.endswith("/v1/images"):
                endpoint = endpoint + "/compositions"
            elif endpoint.endswith("/v1/images/"):
                endpoint = endpoint + "compositions"
        else:
            if endpoint.endswith("/compositions"):
                endpoint = endpoint[: -len("/compositions")] + "/generations"
            elif endpoint.endswith("/v1/images"):
                endpoint = endpoint + "/generations"
            elif endpoint.endswith("/v1/images/"):
                endpoint = endpoint + "generations"

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "ratio": ratio,
            "response_format": "b64_json",
        }
        if is_img2img:
            payload["images"] = [self._bytes_to_base64(img) for img in images]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        timeout = self.config.get("timeout", 120)
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        proxy = self.config.get("proxy_url") if self.config.get("dreamina_use_proxy") else None

        if self.config.get("debug_mode", False):
            logger.info(f"[Dreamina] 请求: url={endpoint}, model={model}, ratio={ratio}, images={len(images)}")

        try:
            session = await self._get_session()
            async with session.post(
                endpoint,
                json=payload,
                headers=headers,
                proxy=proxy,
                timeout=timeout_obj,
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return False, f"API错误 ({resp.status}): {text[:200]}"

                data = await resp.json()
                if isinstance(data, dict):
                    if "code" in data and data.get("code") not in (0, None):
                        return False, f"Dreamina2api错误 ({data.get('code')}): {data.get('message', 'Unknown error')}"
                    if "errmsg" in data and data.get("errmsg"):
                        return False, f"Dreamina2api错误: {data.get('errmsg')}"
                    if "error" in data and data.get("error"):
                        return False, f"Dreamina2api错误: {data.get('error')}"
                items = data.get("data", [])
                if not items:
                    return False, f"API返回空内容: {str(data)[:200]}"

                # 优先base64
                b64 = items[0].get("b64_json")
                if b64:
                    try:
                        return True, base64.b64decode(b64)
                    except Exception:
                        pass

                # 兼容url格式
                img_url = items[0].get("url")
                if img_url:
                    img_data = await self._download_image(img_url)
                    if img_data:
                        return True, img_data
                    return False, f"图片下载失败: {img_url}"

                return False, "未找到可用图片数据"
        except asyncio.TimeoutError:
            return False, "请求超时"
        except Exception as e:
            return False, f"请求异常: {e}"
    
    async def generate(self, mode: str, images: List[bytes], prompt: str, resolution: str | None = None) -> Tuple[bool, Any]:
        """统一生成入口（带重试机制，指数退避）"""
        enable_retry = self.config.get("enable_retry", True)
        max_retries = self.config.get("max_retries", 3)
        base_delay = self.config.get("retry_delay", 2)
        try:
            max_retries = int(max_retries)
        except Exception:
            max_retries = 3
        if max_retries < 1:
            max_retries = 1
        
        # 不可重试的错误关键词（配置错误、权限问题等）
        non_retryable = ["未配置", "API Key", "配置错误", "权限", "Unauthorized", "Forbidden", "Invalid"]
        
        last_error = "未知错误"

        if images:
            images = await self._maybe_compress_images(images, mode)

        if not enable_retry:
            if mode == "flow":
                success, result = await self._call_flow_api(images, prompt)
            elif mode == "dreamina":
                success, result = await self._call_dreamina_api(images, prompt)
            elif mode == "gemini":
                success, result = await self._call_gemini_api(images, prompt, resolution)
            else:
                success, result = await self._call_generic_api(images, prompt, resolution)
            if success:
                if self.config.get("debug_mode", False):
                    logger.info(f"[{mode}] 生成成功 (单次尝试)")
                return True, result
            return False, result
        
        for attempt in range(max_retries):
            if mode == "flow":
                success, result = await self._call_flow_api(images, prompt)
            elif mode == "dreamina":
                success, result = await self._call_dreamina_api(images, prompt)
            elif mode == "gemini":
                success, result = await self._call_gemini_api(images, prompt, resolution)
            else:
                success, result = await self._call_generic_api(images, prompt, resolution)
            
            if success:
                if self.config.get("debug_mode", False):
                    logger.info(f"[{mode}] 生成成功 (第{attempt + 1}次尝试)")
                return True, result
            
            last_error = result
            error_str = str(result)
            
            # 如果是不可重试的错误，直接返回
            if any(err in error_str for err in non_retryable):
                return False, result
            
            # 还有重试机会
            if attempt < max_retries - 1:
                # 指数退避: 2, 4, 8秒...
                delay = base_delay * (2 ** attempt)
                if self.config.get("debug_mode", False):
                    logger.info(f"[{mode}] 第{attempt + 1}次失败，{delay}秒后重试: {result}")
                await asyncio.sleep(delay)
        
        return False, f"重试{max_retries}次后仍失败: {last_error}"
    
    # ================== 撤回功能 ==================
    
    async def _auto_recall(self, event: AstrMessageEvent, message_id: Any):
        """自动撤回消息"""
        if not self.config.get("enable_auto_recall", False):
            return
        
        delay = self.config.get("auto_recall_delay", 15)
        await asyncio.sleep(delay)
        
        try:
            # 尝试撤回消息
            if hasattr(event, 'bot') and hasattr(event.bot, 'recall_message'):
                await event.bot.recall_message(message_id)
        except Exception as e:
            logger.debug(f"撤回失败: {e}")

    async def _delayed_recall(self, bot: Any, message_id: Any, delay: int):
        """延迟撤回消息（OneBot优先）"""
        await asyncio.sleep(delay)
        try:
            if isinstance(message_id, (str, int)) and str(message_id).isdigit():
                message_id = int(message_id)
            if hasattr(bot, "recall_message"):
                await bot.recall_message(message_id)
            elif hasattr(bot, "delete_msg"):
                await bot.delete_msg(message_id=message_id)
        except Exception as e:
            if self.config.get("debug_mode", False):
                logger.warning(f"[AutoRecall] 撤回失败: {e}")

    async def _send_chain_with_recall(self, event: AstrMessageEvent, chain: List[Any]) -> bool:
        """按OneBot方式发送并创建撤回任务，成功则返回True"""
        if not self.config.get("enable_auto_recall", False):
            return False
        delay = self.config.get("auto_recall_delay", 15)
        if not delay:
            return False

        bot = getattr(event, "bot", None)
        if bot is None or MessageChain is None or not hasattr(event, "_parse_onebot_json"):
            return False

        group_id = event.get_group_id()
        user_id = event.get_sender_id()
        if not group_id and not user_id:
            return False

        try:
            obmsg = await event._parse_onebot_json(MessageChain(chain=chain))
        except Exception as e:
            if self.config.get("debug_mode", False):
                logger.warning(f"[AutoRecall] 解析OneBot消息失败: {e}")
            return False

        try:
            result = None
            if group_id and hasattr(bot, "send_group_msg"):
                result = await bot.send_group_msg(group_id=int(group_id), message=obmsg)
            elif user_id and hasattr(bot, "send_private_msg"):
                result = await bot.send_private_msg(user_id=int(user_id), message=obmsg)
            else:
                return False
        except Exception as e:
            if self.config.get("debug_mode", False):
                logger.warning(f"[AutoRecall] 发送消息失败: {e}")
            return False

        message_id = self._extract_message_id_from_obj(result)
        if not message_id:
            if self.config.get("debug_mode", False):
                logger.warning("[AutoRecall] 未获取到message_id，无法撤回")
            return False

        task = asyncio.create_task(self._delayed_recall(bot, message_id, delay))
        self.pending_tasks.add(task)
        task.add_done_callback(self.pending_tasks.discard)
        if self.config.get("debug_mode", False):
            logger.info(f"[AutoRecall] 已创建撤回任务，{delay}s 后撤回 {message_id}")
        return True
    
    async def _send_and_recall(self, event: AstrMessageEvent, text: str):
        """发送消息并计划撤回"""
        result = event.plain_result(text)
        # 注：实际撤回需要平台支持，这里只是预留接口
        return result

    def _extract_message_id_from_obj(self, obj: Any) -> Any:
        """从不同对象中提取message_id"""
        if obj is None:
            return None
        if isinstance(obj, dict):
            for key in ("message_id", "messageId", "msg_id", "id"):
                val = obj.get(key)
                if val:
                    return val
            data = obj.get("data")
            if isinstance(data, dict):
                for key in ("message_id", "messageId", "msg_id", "id"):
                    val = data.get(key)
                    if val:
                        return val
            return None
        for key in ("message_id", "messageId", "msg_id", "id"):
            if hasattr(obj, key):
                val = getattr(obj, key)
                if val:
                    return val
        if hasattr(obj, "data"):
            data = getattr(obj, "data")
            if isinstance(data, dict):
                for key in ("message_id", "messageId", "msg_id", "id"):
                    val = data.get(key)
                    if val:
                        return val
        return None
    
    # ================== 命令处理 ==================
    
    def _parse_mode_from_command(self, cmd: str) -> Tuple[str, str]:
        """从命令解析模式和实际命令"""
        if cmd.startswith("f"):
            return "flow", cmd[1:]
        elif cmd.startswith("o"):
            return "generic", cmd[1:]
        elif cmd.startswith("g"):
            return "gemini", cmd[1:]
        elif cmd.startswith("d"):
            return "dreamina", cmd[1:]
        return None, cmd  # 无前缀

    def _parse_mode_token(self, token: str | None) -> str | None:
        """解析模式参数"""
        if not token:
            return None
        text = str(token).strip().lower()
        mapping = {
            "f": "flow",
            "flow": "flow",
            "f模式": "flow",
            "flow模式": "flow",
            "o": "generic",
            "generic": "generic",
            "gen": "generic",
            "o模式": "generic",
            "generic模式": "generic",
            "通用": "generic",
            "通用模式": "generic",
            "g": "gemini",
            "gemini": "gemini",
            "g模式": "gemini",
            "gemini模式": "gemini",
            "d": "dreamina",
            "dreamina": "dreamina",
            "d模式": "dreamina",
            "dreamina模式": "dreamina",
        }
        return mapping.get(text)
    
    def _get_effective_mode(self, requested_mode: str | None, user_id: str, group_id: str) -> str:
        """获取实际使用的模式"""
        # 如果指定了模式，检查权限
        if requested_mode:
            return requested_mode
        
        # 无前缀时
        if limit_manager.is_user_whitelisted(user_id, self.config):
            return self.config.get("default_mode", "generic")
        if limit_manager.is_group_whitelisted(group_id, self.config):
            return self.config.get("default_mode", "generic")
        
        # 普通用户使用配置的默认模式
        return self.config.get("normal_user_default_mode", "flow")
    
    def _check_mode_enabled(self, mode: str) -> Tuple[bool, str]:
        """
        检查模式是否启用
        
        返回: (是否启用, 错误提示)
        """
        mode_switch = {
            "flow": "enable_flow_mode",
            "generic": "enable_generic_mode",
            "gemini": "enable_gemini_mode",
            "dreamina": "enable_dreamina_mode"
        }
        
        enabled = self.config.get(mode_switch[mode], True)
        if mode == "dreamina" and mode_switch[mode] not in self.config:
            enabled = False

        if not enabled:
            # 找出当前可用的模式
            available = []
            for m, switch in mode_switch.items():
                if self.config.get(switch, True):
                    available.append(m)
            
            if not available:
                return False, "❌ 所有绘图模式均已关闭"
            
            mode_names = {
                "flow": "Flow (f)",
                "generic": "Generic (o)",
                "gemini": "Gemini (g)",
                "dreamina": "Dreamina (d)"
            }
            
            current_name = mode_names[mode]
            available_names = [mode_names[m] for m in available]
            
            return False, f"❌ {current_name} 模式当前不可用\n💡 可用模式: {', '.join(available_names)}"
        
        return True, ""

    def _normalize_resolution(self, resolution: str | None) -> str | None:
        """标准化分辨率参数"""
        if resolution is None:
            return None
        text = str(resolution).strip().upper()
        if not text:
            return None
        if text in {"1K", "2K", "4K"}:
            return text
        return None

    def _get_dreamina_ratio(self, images: List[bytes]) -> str:
        """获取Dreamina比例（自动或固定比例）"""
        ratio = self.config.get("dreamina_ratio", "自动")
        ratio_text = str(ratio).strip()
        if not ratio_text:
            ratio_text = "自动"

        # 允许中英文自动标记
        if ratio_text.lower() in {"auto", "自动"}:
            if images and PILImage is not None:
                try:
                    img = PILImage.open(io.BytesIO(images[0]))
                    width, height = img.size
                    if height and width:
                        return self._match_dreamina_ratio(width, height)
                except Exception:
                    pass
            return "1:1"

        # 固定比例
        supported = {
            "1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9"
        }
        if ratio_text in supported:
            return ratio_text

        logger.warning(f"[Dreamina] 不支持的比例: {ratio_text}, 回退到 1:1")
        return "1:1"

    def _match_dreamina_ratio(self, width: int, height: int) -> str:
        """根据宽高匹配Dreamina标准比例"""
        aspect = width / height if height else 1.0
        ratio_map = {
            "1:1": 1.0,
            "4:3": 4 / 3,
            "3:4": 3 / 4,
            "16:9": 16 / 9,
            "9:16": 9 / 16,
            "3:2": 3 / 2,
            "2:3": 2 / 3,
            "21:9": 21 / 9,
        }
        # 优先匹配接近比例
        for key, val in ratio_map.items():
            if abs(aspect - val) < 0.1:
                return key
        # 否则选最接近的
        return min(ratio_map.keys(), key=lambda k: abs(aspect - ratio_map[k]))

    def _get_llm_cache_key(self, event: AstrMessageEvent) -> str:
        """获取LLM图片缓存Key（会话粒度）"""
        platform = event.get_platform_name() if hasattr(event, "get_platform_name") else ""
        group_id = event.get_group_id() if hasattr(event, "get_group_id") else None
        user_id = event.get_sender_id() if hasattr(event, "get_sender_id") else None
        if group_id and user_id:
            return f"{platform}:group:{group_id}:user:{user_id}"
        if user_id:
            return f"{platform}:user:{user_id}"
        unified = getattr(event, "unified_msg_origin", None)
        if unified:
            return str(unified)
        if group_id:
            return f"{platform}:group:{group_id}"
        return platform or "default"

    def _get_llm_cache_ttl(self) -> int:
        """获取LLM图片缓存TTL（秒）"""
        ttl = self.config.get("llm_last_image_ttl", 3600)
        try:
            ttl = int(ttl)
        except Exception:
            ttl = 3600
        return max(0, ttl)

    def _get_llm_cache_max_entries(self) -> int:
        """获取LLM图片缓存最大条数"""
        limit = self.config.get("llm_last_image_max_entries", 100)
        try:
            limit = int(limit)
        except Exception:
            limit = 100
        return max(0, limit)

    def _prune_llm_cache(self):
        """清理过期或超限的LLM图片缓存"""
        if not self.llm_last_image_cache:
            return
        now = time.time()
        ttl = self._get_llm_cache_ttl()
        if ttl > 0:
            for key, entry in list(self.llm_last_image_cache.items()):
                ts = entry.get("ts", 0)
                if now - ts > ttl:
                    self.llm_last_image_cache.pop(key, None)

        max_entries = self._get_llm_cache_max_entries()
        if max_entries > 0 and len(self.llm_last_image_cache) > max_entries:
            items = sorted(
                self.llm_last_image_cache.items(),
                key=lambda kv: kv[1].get("ts", 0)
            )
            for key, _ in items[: max(0, len(items) - max_entries)]:
                self.llm_last_image_cache.pop(key, None)

    def _set_llm_last_image(self, cache_key: str, image_bytes: bytes):
        """写入LLM“上一次图片”缓存"""
        if not cache_key or not image_bytes:
            return
        self._prune_llm_cache()
        self.llm_last_image_cache[cache_key] = {
            "image": image_bytes,
            "ts": time.time(),
        }
        self._prune_llm_cache()

    def _get_llm_last_image(self, cache_key: str) -> bytes | None:
        """读取LLM“上一次图片”缓存"""
        if not cache_key:
            return None
        self._prune_llm_cache()
        entry = self.llm_last_image_cache.get(cache_key)
        if not entry:
            return None
        ttl = self._get_llm_cache_ttl()
        if ttl > 0 and time.time() - entry.get("ts", 0) > ttl:
            self.llm_last_image_cache.pop(cache_key, None)
            return None
        entry["ts"] = time.time()
        return entry.get("image")

    def _is_followup_request(self, event: AstrMessageEvent, prompt: str) -> bool:
        """判断是否为“继续修改/沿用上一张”意图"""
        parts = [prompt or ""]
        if hasattr(event, "get_message_str"):
            parts.append(event.get_message_str() or "")
        text = " ".join(parts).strip()
        if not text:
            return False
        text_lower = text.lower()
        keywords = [
            "继续", "接着", "再改", "改一下", "修改", "调整", "优化",
            "上次", "上一张", "上一幅", "上一个", "刚才", "同上",
            "继续这张", "继续这个", "在此基础", "基于上次", "基于上一张",
            "沿用", "参考上次", "参考上一张",
            "refine", "revise", "modify", "tweak", "iterate",
            "previous", "last one", "same image",
        ]
        return any(k in text_lower or k in text for k in keywords)
    
    # ================== 文生图命令 ==================
    
    @filter.command("f文", alias={"f文生图"})
    async def cmd_flow_text2img(self, event: AstrMessageEvent):
        """Flow模式文生图"""
        async for result in self._handle_text2img(event, "flow"):
            yield result
    
    @filter.command("o文", alias={"o文生图"})
    async def cmd_generic_text2img(self, event: AstrMessageEvent):
        """Generic模式文生图"""
        async for result in self._handle_text2img(event, "generic"):
            yield result
    
    @filter.command("g文", alias={"g文生图"})
    async def cmd_gemini_text2img(self, event: AstrMessageEvent):
        """Gemini模式文生图"""
        async for result in self._handle_text2img(event, "gemini"):
            yield result

    @filter.command("d文", alias={"d文生图"})
    async def cmd_dreamina_text2img(self, event: AstrMessageEvent):
        """Dreamina模式文生图"""
        async for result in self._handle_text2img(event, "dreamina"):
            yield result
    
    @filter.command("文生图", alias={"文"})
    async def cmd_default_text2img(self, event: AstrMessageEvent):
        """默认模式文生图"""
        user_id = event.get_sender_id()
        group_id = event.get_group_id()
        mode = self._get_effective_mode(None, user_id, group_id)
        async for result in self._handle_text2img(event, mode):
            yield result
    
    async def _handle_text2img(self, event: AstrMessageEvent, mode: str):
        """处理文生图请求"""
        user_id = event.get_sender_id()
        group_id = event.get_group_id()
        
        # 权限检查
        allowed, actual_mode, err_msg = limit_manager.check_permission(user_id, group_id, mode, self.config)
        if not allowed:
            yield event.plain_result(err_msg)
            return
        
        # 模式启用检查
        enabled, mode_err = self._check_mode_enabled(actual_mode)
        if not enabled:
            yield event.plain_result(mode_err)
            return
        
        # 提取提示词并清理@用户信息
        raw = event.get_message_str().strip()
        prompt = re.sub(r'^[fogd]?文(生图)?\s*', '', raw, flags=re.IGNORECASE).strip()
        prompt = self._clean_prompt(prompt, event)
        
        if not prompt:
            yield event.plain_result("❌ 请输入描述\n用法: #f文 一只可爱的猫")
            return
        
        # 次数检查
        ok, limit_msg = limit_manager.check_and_consume(user_id, group_id, self.config)
        if not ok:
            yield event.plain_result(f"❌ {limit_msg}")
            return
        
        mode_name = {"flow": "Flow", "generic": "Generic", "gemini": "Gemini", "dreamina": "Dreamina"}[actual_mode]
        
        # 并发控制 - 白名单用户不受限制
        is_whitelisted = limit_manager.is_user_whitelisted(user_id, self.config)
        mode_lock = self.mode_locks[actual_mode]
        
        if not is_whitelisted and mode_lock.locked():
            yield event.plain_result(f"⏳ [{mode_name}] 当前有其他用户正在生成，请稍候...")
            return
        
        async def do_generate():
            yield event.plain_result(f"🎨 [{mode_name}] 文生图: {prompt[:20]}...")
            
            start = time.time()
            success, result = await self.generate(actual_mode, [], prompt)
            elapsed = time.time() - start
            
            if success:
                chain = [
                    self._create_image_from_bytes(result),
                    Plain(f"✅ [{mode_name}] 生成成功 ({elapsed:.1f}s) | {limit_msg}")
                ]
                if await self._send_chain_with_recall(event, chain):
                    return
                yield event.chain_result(chain)
            else:
                yield event.plain_result(f"❌ [{mode_name}] 生成失败 ({elapsed:.1f}s)\n原因: {result}")
        
        if is_whitelisted:
            async for r in do_generate():
                yield r
        else:
            async with mode_lock:
                async for r in do_generate():
                    yield r
    
    # ================== 图生图命令 ==================
    
    @filter.command("f图", alias={"f图生图"})
    async def cmd_flow_img2img(self, event: AstrMessageEvent):
        """Flow模式图生图"""
        async for result in self._handle_img2img(event, "flow"):
            yield result
    
    @filter.command("o图", alias={"o图生图"})
    async def cmd_generic_img2img(self, event: AstrMessageEvent):
        """Generic模式图生图"""
        async for result in self._handle_img2img(event, "generic"):
            yield result
    
    @filter.command("g图", alias={"g图生图"})
    async def cmd_gemini_img2img(self, event: AstrMessageEvent):
        """Gemini模式图生图"""
        async for result in self._handle_img2img(event, "gemini"):
            yield result

    @filter.command("d图", alias={"d图生图"})
    async def cmd_dreamina_img2img(self, event: AstrMessageEvent):
        """Dreamina模式图生图"""
        async for result in self._handle_img2img(event, "dreamina"):
            yield result
    
    @filter.command("图生图", alias={"图"})
    async def cmd_default_img2img(self, event: AstrMessageEvent):
        """默认模式图生图"""
        user_id = event.get_sender_id()
        group_id = event.get_group_id()
        mode = self._get_effective_mode(None, user_id, group_id)
        async for result in self._handle_img2img(event, mode):
            yield result
    
    async def _handle_img2img(self, event: AstrMessageEvent, mode: str):
        """处理图生图请求"""
        user_id = event.get_sender_id()
        group_id = event.get_group_id()
        
        # 权限检查
        allowed, actual_mode, err_msg = limit_manager.check_permission(user_id, group_id, mode, self.config)
        if not allowed:
            yield event.plain_result(err_msg)
            return
        
        # 模式启用检查
        enabled, mode_err = self._check_mode_enabled(actual_mode)
        if not enabled:
            yield event.plain_result(mode_err)
            return
        
        # 提取提示词并清理@用户信息
        raw = event.get_message_str().strip()
        prompt = re.sub(r'^[fogd]?图(生图)?\s*', '', raw, flags=re.IGNORECASE).strip()
        prompt = self._clean_prompt(prompt, event)
        
        if not prompt:
            prompt = "transform this image with artistic style"
        
        # 获取图片
        images = await self.get_images(event)
        if not images:
            yield event.plain_result("❌ 请发送或引用一张图片\n用法: #f图 [发送图片]")
            return
        
        # 次数检查
        ok, limit_msg = limit_manager.check_and_consume(user_id, group_id, self.config)
        if not ok:
            yield event.plain_result(f"❌ {limit_msg}")
            return
        
        mode_name = {"flow": "Flow", "generic": "Generic", "gemini": "Gemini", "dreamina": "Dreamina"}[actual_mode]
        
        # 并发控制 - 白名单用户不受限制
        is_whitelisted = limit_manager.is_user_whitelisted(user_id, self.config)
        mode_lock = self.mode_locks[actual_mode]
        
        if not is_whitelisted and mode_lock.locked():
            yield event.plain_result(f"⏳ [{mode_name}] 当前有其他用户正在生成，请稍候...")
            return
        
        async def do_generate():
            yield event.plain_result(f"🎨 [{mode_name}] 图生图: {len(images)}张图片...")
            
            start = time.time()
            success, result = await self.generate(actual_mode, images, prompt)
            elapsed = time.time() - start
            
            if success:
                chain = [
                    self._create_image_from_bytes(result),
                    Plain(f"✅ [{mode_name}] 生成成功 ({elapsed:.1f}s) | {limit_msg}")
                ]
                if await self._send_chain_with_recall(event, chain):
                    return
                yield event.chain_result(chain)
            else:
                yield event.plain_result(f"❌ [{mode_name}] 生成失败 ({elapsed:.1f}s)\n原因: {result}")
        
        if is_whitelisted:
            async for r in do_generate():
                yield r
        else:
            async with mode_lock:
                async for r in do_generate():
                    yield r
    
    # ================== 自定义预设命令监听器 ==================
    
    @filter.event_message_type(filter.EventMessageType.ALL, priority=5)
    async def on_custom_preset(self, event: AstrMessageEvent, ctx=None):
        """处理自定义预设命令（从prompt_list配置加载的预设）"""
        # 检查是否需要前缀
        if self.config.get("prefix", True):
            # 兼容不同版本AstrBot
            is_wake = getattr(event, 'is_at_or_wake_command', True)
            if not is_wake:
                return
        
        # 安全获取消息文本
        text = getattr(event, 'message_str', '')
        if not text:
            text = str(event.message_obj.message) if hasattr(event, 'message_obj') else ''
        
        text = text.strip()
        if not text:
            return
        
        # 提取命令词（第一个token）
        tokens = text.split()
        if not tokens:
            return
        
        raw_cmd = tokens[0].strip()
        
        # 解析命令前缀 (f/o/g/d) 和基础命令
        prefix_mode = None
        base_cmd = raw_cmd
        
        if len(raw_cmd) > 1:
            first_char = raw_cmd[0].lower()
            if first_char in ('f', 'o', 'g', 'd'):
                # 检查去掉前缀后的命令是否在自定义预设中
                potential_cmd = raw_cmd[1:]
                if potential_cmd in self.prompt_map:
                    prefix_mode = {"f": "flow", "o": "generic", "g": "gemini", "d": "dreamina"}.get(first_char)
                    base_cmd = potential_cmd
        
        # 检查是否匹配自定义预设（排除已硬编码的内置预设命令）
        if base_cmd not in self.prompt_map:
            return  # 不是自定义预设，让其他处理器处理
        
        # 排除内置预设（它们有专门的@filter.command装饰器）
        if base_cmd in self.builtin_presets:
            return  # 内置预设由专门的命令处理器处理
        
        # 是自定义预设，处理它
        user_id = event.get_sender_id()
        group_id = event.get_group_id()
        
        # 确定使用的模式
        if prefix_mode:
            mode = prefix_mode
        else:
            mode = self._get_effective_mode(None, user_id, group_id)
        
        # 调用预设处理
        async for r in self._handle_preset(event, mode, base_cmd):
            yield r
        
        # 停止事件传播
        event.stop_event()
    
    # ================== 预设命令 ==================

    
    @filter.command("f手办化")
    async def cmd_flow_figurine(self, event: AstrMessageEvent):
        async for r in self._handle_preset(event, "flow", "手办化"):
            yield r
    
    @filter.command("o手办化")
    async def cmd_generic_figurine(self, event: AstrMessageEvent):
        async for r in self._handle_preset(event, "generic", "手办化"):
            yield r
    
    @filter.command("g手办化")
    async def cmd_gemini_figurine(self, event: AstrMessageEvent):
        async for r in self._handle_preset(event, "gemini", "手办化"):
            yield r

    @filter.command("d手办化")
    async def cmd_dreamina_figurine(self, event: AstrMessageEvent):
        async for r in self._handle_preset(event, "dreamina", "手办化"):
            yield r
    
    @filter.command("手办化")
    async def cmd_default_figurine(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()
        group_id = event.get_group_id()
        mode = self._get_effective_mode(None, user_id, group_id)
        async for r in self._handle_preset(event, mode, "手办化"):
            yield r
    
    async def _handle_preset(self, event: AstrMessageEvent, mode: str, preset_name: str):
        """处理预设命令"""
        user_id = event.get_sender_id()
        group_id = event.get_group_id()
        
        # 权限检查
        allowed, actual_mode, err_msg = limit_manager.check_permission(user_id, group_id, mode, self.config)
        if not allowed:
            yield event.plain_result(err_msg)
            return
        
        # 模式启用检查
        enabled, mode_err = self._check_mode_enabled(actual_mode)
        if not enabled:
            yield event.plain_result(mode_err)
            return
        
        # 获取预设提示词
        prompt = self.prompt_map.get(preset_name) or self.builtin_presets.get(preset_name, preset_name)
        
        # 获取图片
        images = await self.get_images(event)
        if not images:
            # 尝试获取发送者头像
            if avatar := await self._get_avatar(user_id):
                images = [avatar]
            else:
                yield event.plain_result("❌ 请发送或引用一张图片")
                return
        
        # 次数检查
        ok, limit_msg = limit_manager.check_and_consume(user_id, group_id, self.config)
        if not ok:
            yield event.plain_result(f"❌ {limit_msg}")
            return
        
        mode_name = {"flow": "Flow", "generic": "Generic", "gemini": "Gemini", "dreamina": "Dreamina"}[actual_mode]
        
        # 并发控制 - 白名单用户不受限制
        is_whitelisted = limit_manager.is_user_whitelisted(user_id, self.config)
        mode_lock = self.mode_locks[actual_mode]
        
        if not is_whitelisted and mode_lock.locked():
            yield event.plain_result(f"⏳ [{mode_name}] 当前有其他用户正在生成，请稍候...")
            return
        
        async def do_generate():
            yield event.plain_result(f"🎨 [{mode_name}] {preset_name}...")
            
            start = time.time()
            success, result = await self.generate(actual_mode, images, prompt)
            elapsed = time.time() - start
            
            if success:
                chain = [
                    self._create_image_from_bytes(result),
                    Plain(f"✅ [{mode_name}] {preset_name}成功 ({elapsed:.1f}s) | {limit_msg}")
                ]
                if await self._send_chain_with_recall(event, chain):
                    return
                yield event.chain_result(chain)
            else:
                yield event.plain_result(f"❌ [{mode_name}] {preset_name}失败 ({elapsed:.1f}s)\n原因: {result}")
        
        if is_whitelisted:
            async for r in do_generate():
                yield r
        else:
            async with mode_lock:
                async for r in do_generate():
                    yield r
    
    # ================== 管理命令 ==================
    
    @filter.command("查询次数")
    async def cmd_query_limit(self, event: AstrMessageEvent):
        """查询剩余次数"""
        user_id = event.get_sender_id()
        remaining = limit_manager.get_user_remaining(user_id, self.config)
        yield event.plain_result(f"👤 用户: {user_id}\n📊 今日剩余: {remaining}")
    
    @filter.command("f切换模型")
    async def cmd_switch_flow_model(self, event: AstrMessageEvent):
        """切换Flow模式模型"""
        model_list = self.config.get("flow_model_list", [])
        if not model_list:
            yield event.plain_result("❌ Flow模式模型列表为空")
            return
        
        self.flow_current_model_index = (self.flow_current_model_index + 1) % len(model_list)
        current = model_list[self.flow_current_model_index]
        
        msg = "🔄 Flow模式模型已切换\n"
        for i, m in enumerate(model_list):
            prefix = "➤ " if i == self.flow_current_model_index else "  "
            msg += f"{prefix}{i+1}. {m}\n"
        
        yield event.plain_result(msg)
    
    @filter.command("f翻译开关")
    async def cmd_toggle_translate(self, event: AstrMessageEvent):
        """切换翻译功能"""
        current = self.config.get("flow_enable_translate", False)
        self.config["flow_enable_translate"] = not current
        status = "开启" if not current else "关闭"
        yield event.plain_result(f"🌐 翻译功能已{status}")

    @filter.command("切换到", alias={"切换默认", "切换默认模式"})
    async def cmd_switch_default_modes(self, event: AstrMessageEvent):
        """切换默认模式（白名单/普通用户/LLM工具）"""
        raw = event.get_message_str().strip()
        target = re.sub(r"^切换到\s*", "", raw, flags=re.IGNORECASE).strip()
        if not target:
            yield event.plain_result("用法: #切换到 f/o/g/d")
            return
        target = target.split()[0]

        mode = self._parse_mode_token(target)
        if not mode:
            yield event.plain_result("无效模式，请输入 f/o/g 或 flow/generic/gemini")
            return

        enabled, mode_err = self._check_mode_enabled(mode)
        if not enabled:
            yield event.plain_result(mode_err)
            return

        mode_name = self._apply_default_mode(mode, source="manual")
        yield event.plain_result(f"✅ 已切换默认模式为 {mode_name}（白名单/普通用户/LLM）")
    
    @filter.command("预设列表")
    async def cmd_list_presets(self, event: AstrMessageEvent):
        """列出所有预设"""
        builtin = list(self.builtin_presets.keys())
        custom = list(self.prompt_map.keys())
        
        msg = "📜 可用预设列表\n━━━━━━━━━━\n"
        msg += f"📌 内置: {', '.join(builtin)}\n"
        msg += f"✨ 自定义: {', '.join(custom) if custom else '(无)'}\n"
        msg += "━━━━━━━━━━\n用法: #f<预设名> [图片]"
        
        yield event.plain_result(msg)
    
    @filter.command("生图帮助")
    async def cmd_help(self, event: AstrMessageEvent):
        """显示帮助"""
        help_text = self.config.get("help_text", "帮助未配置")
        yield event.plain_result(help_text)
    
    @filter.command("生图菜单")
    async def cmd_menu(self, event: AstrMessageEvent):
        """显示菜单"""
        menu = """🎨 RongheDraw 绘图插件 v1.2.8

━━━━ 📌 快速开始 ━━━━
#f文 <描述>      文字生成图片
#f图 [图片]      图片风格转换
#f随机 [图片]    随机预设效果

━━━━ 🔀 API模式 ━━━━
f = Flow (自动横竖版，支持翻译)
o = Generic (仅白名单)
g = Gemini (仅白名单, 4K输出)
d = Dreamina (仅白名单, 比例可配置)

例: #o文 <描述>  #g图 [图片]  #d文 <描述>

━━━━ ⚙️ 权限/并发 ━━━━
普通用户: 仅 #f 命令，有并发限制
白名单群: 全部模式，有并发限制
白名单用户: 全部模式，无并发限制
每模式同时只允许1个非白名单用户

无前缀命令:
  普通用户 → f模式
  白名单 → 默认配置模式

━━━━ 🔧 管理 ━━━━
#查询次数 | #预设列表
#生图菜单 | #生图帮助
#切换到 f/o/g
#f切换模型 | #f翻译开关

━━━━ 🤖 LLM提示 ━━━━
继续改图: 说“继续修改/基于上次”
分辨率: resolution=1K/2K/4K (仅Generic/Gemini)
Dreamina比例: 配置 dreamina_ratio (自动/固定)
限制: 提示词<900, 图片<=10"""
        yield event.plain_result(menu)
    
    # ================== 随机预设命令 ==================
    
    def _get_all_presets(self) -> list:
        """获取所有可用预设（内置+自定义）"""
        all_presets = list(self.builtin_presets.keys()) + list(self.prompt_map.keys())
        return all_presets if all_presets else []
    
    @filter.command("f随机")
    async def cmd_flow_random(self, event: AstrMessageEvent):
        """Flow模式随机预设"""
        all_presets = self._get_all_presets()
        if not all_presets:
            yield event.plain_result("❌ 暂无可用预设")
            return
        preset = random.choice(all_presets)
        async for r in self._handle_preset(event, "flow", preset):
            yield r
    
    @filter.command("o随机")
    async def cmd_generic_random(self, event: AstrMessageEvent):
        """Generic模式随机预设"""
        all_presets = self._get_all_presets()
        if not all_presets:
            yield event.plain_result("❌ 暂无可用预设")
            return
        preset = random.choice(all_presets)
        async for r in self._handle_preset(event, "generic", preset):
            yield r
    
    @filter.command("g随机")
    async def cmd_gemini_random(self, event: AstrMessageEvent):
        """Gemini模式随机预设"""
        all_presets = self._get_all_presets()
        if not all_presets:
            yield event.plain_result("❌ 暂无可用预设")
            return
        preset = random.choice(all_presets)
        async for r in self._handle_preset(event, "gemini", preset):
            yield r

    @filter.command("d随机")
    async def cmd_dreamina_random(self, event: AstrMessageEvent):
        """Dreamina模式随机预设"""
        all_presets = self._get_all_presets()
        if not all_presets:
            yield event.plain_result("❌ 暂无可用预设")
            return
        preset = random.choice(all_presets)
        async for r in self._handle_preset(event, "dreamina", preset):
            yield r
    
    @filter.command("随机", alias={"随机预设"})
    async def cmd_default_random(self, event: AstrMessageEvent):
        """默认模式随机预设"""
        all_presets = self._get_all_presets()
        if not all_presets:
            yield event.plain_result("❌ 暂无可用预设")
            return
        user_id = event.get_sender_id()
        group_id = event.get_group_id()
        mode = self._get_effective_mode(None, user_id, group_id)
        preset = random.choice(all_presets)
        async for r in self._handle_preset(event, mode, preset):
            yield r
    
    # ================== LLM 工具 ==================
    
    @filter.llm_tool(name="generate_image")
    async def llm_tool_generate_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        use_message_images: bool = False,
        image_urls: Optional[List[str]] = None,
        use_last_image: Optional[bool] = None,
        resolution: Optional[str] = None
    ):
        '''
        绘图工具：根据用户要求生成图片。
        
        何时调用：
        - 仅当用户明确要求“画图/改图/画某人/修改头像”等时调用。
        - 若用户需求不清晰或连续相似请求，请先确认再调用。
        
        参数选择原则（给AI用）：
        - 有用户发图：优先设 use_message_images=true，不需要 image_urls。
        - 有头像/公网参考图：用 image_urls（如先 get_avatar 得到头像URL）。
        - 继续改上一张：只有用户明确提到“沿用上一张/继续上一张/按上一张修改”等时才可设 use_last_image=true，且必须没有新图。
        - 想要分辨率：仅用户明确要求时再传 resolution。
        
        使用流程示例：
        1) 画群友/改头像：先调用 get_avatar(user_id) 获取头像URL（不要发给用户），再调用 generate_image(prompt, image_urls=[URL])。
        2) 纯场景/文生图：直接调用 generate_image(prompt)。
        3) 用户已发图：用 generate_image(prompt, use_message_images=true)。
        4) 继续改图：仅当用户明确要求“参照上一张/继续上一张”时，用 generate_image(prompt, use_last_image=true)。
        5) 多人同框：将多个头像URL放入 image_urls 列表。
        
        重要注意：
        - 图片生成后系统会自动发送，不要发送链接或URL给用户。
        - gchat.qpic.cn 等临时链接不可用，优先 use_message_images。
        - image_urls 支持 dataURL/base64://，可直接传 Base64。
        - 使用头像时，prompt 不要描述人物外貌/性别，除非用户明确要求。
        - 未明确要求画人/头像时不要调用 get_avatar。
        - 图片最多10张，提示词需少于900字符。
        - Dreamina 使用 ratio 配置控制比例，不支持 resolution 参数。
        - resolution 仅对 Generic/Gemini 生效，Flow/Dreamina 模式会忽略。
        
        Args:
            prompt (string): 必填。画面描述或修改要求，尽量具体，长度 < 900 字符。
            use_message_images (boolean, optional): 当用户消息里有图时设为 true，自动取图（推荐，支持QQ群聊图）。
            image_urls (array[string], optional): 参考图URL列表（公网稳定URL或Base64 dataURL/base64://）。
            use_last_image (boolean, optional): 仅在用户明确要求“参照上一张/继续上一张”等且没有新图时设为 true。
            resolution (string, optional): 1K/2K/4K。仅在用户明确要求分辨率/清晰度时传。
        '''
        if not self.config.get("enable_llm_tool", False):
            yield event.plain_result("LLM 绘图工具未启用")
            return
        
        # 输入验证
        if len(prompt) >= 900:
            yield event.plain_result("提示词过长（需少于900字符）")
            return

        if image_urls and len(image_urls) > 10:
            yield event.plain_result("图片数量过多（最大10张）")
            return
        
        user_id = event.get_sender_id()
        group_id = event.get_group_id()
        cache_key = self._get_llm_cache_key(event)

        # 统一使用llm_default_mode（不区分白名单/普通用户）
        mode = self.config.get("llm_default_mode", "generic")
        
        # 模式启用检查
        enabled, mode_err = self._check_mode_enabled(mode)
        if not enabled:
            yield event.plain_result(mode_err)
            return

        # 分辨率参数（仅Generic/Gemini有效）
        resolution_override = self._normalize_resolution(resolution)
        if resolution and not resolution_override:
            yield event.plain_result("分辨率仅支持 1K/2K/4K")
            return
        if resolution_override and mode in ("flow", "dreamina"):
            resolution_override = None
        
        # 次数检查（按配置选择群统计或个人统计）
        use_group_limit = self.config.get("llm_tool_use_group_limit", True)
        if use_group_limit and group_id:
            ok, limit_msg = limit_manager.check_and_consume_group(group_id, self.config)
        else:
            ok, limit_msg = limit_manager.check_and_consume(user_id, None, self.config)
        
        if not ok:
            yield event.plain_result(f"❌ {limit_msg}")
            return
        
        # 处理图片
        images = []
        
        # 方式1: 从用户消息中直接获取图片（推荐，支持QQ群聊图片）
        if use_message_images:
            images = await self.get_images(event)
            if self.config.get("debug_mode", False):
                logger.info(f"[LLM Tool] 从消息中获取了 {len(images)} 张图片")
            if len(images) > 10:
                yield event.plain_result("图片数量过多（最大10张）")
                return
        
        # 方式2: 从AI提供的URL列表下载图片（仅支持公网URL）
        if image_urls and not images:  # 只有在use_message_images未获取到图片时才使用URL
            skipped_qq_urls = []
            for url in image_urls:
                if url.startswith(("data:", "base64://")):
                    img_data = await self._load_image_bytes(url)
                    if img_data:
                        images.append(img_data)
                    continue
                # URL格式检查
                if not url.startswith(('http://', 'https://')):
                    continue  # 静默跳过无效URL
                
                # 检测QQ群聊图片临时链接（这些URL无法直接下载）
                if 'gchat.qpic.cn' in url or 'multimedia.nt.qq.com.cn' in url:
                    skipped_qq_urls.append(url)
                    logger.warning(f"[LLM Tool] 跳过无法下载的QQ图片临时链接: {url[:80]}...")
                    continue
                
                # 下载图片
                img_data = await self._download_image(url)
                if img_data:
                    images.append(img_data)
            
            # 如果所有URL都被跳过，提示使用use_message_images
            if skipped_qq_urls and not images:
                yield event.plain_result("无法使用QQ群聊图片，请使用头像URL或其他可访问的图片链接")
                return
            if len(images) > 10:
                yield event.plain_result("图片数量过多（最大10张）")
                return
        
        # 清理提示词中的@用户信息
        clean_prompt = self._clean_prompt(prompt, event)

        # 处理“上一次图片”缓存（仅在未提供图片时）
        if not images:
            cached = self._get_llm_last_image(cache_key)
            if use_last_image is True:
                if cached:
                    images = [cached]
                else:
                    yield event.plain_result("没有可用的上一张图片，请发送或回复图片")
                    return
            elif use_last_image is None and cached:
                if self._is_followup_request(event, clean_prompt):
                    images = [cached]

        # 静默处理无效URL
        
        
        start = time.time()
        success, result = await self.generate(mode, images, clean_prompt, resolution_override)
        elapsed = time.time() - start
        
        if success:
            # 成功：仅发送图片给用户，不发送状态消息
            self._set_llm_last_image(cache_key, result)
            chain = [self._create_image_from_bytes(result)]
            if await self._send_chain_with_recall(event, chain):
                return
            yield event.chain_result(chain)
        else:
            # 失败：发送简短错误信息
            yield event.plain_result("生成失败")
    
    
    
    @filter.llm_tool(name="get_avatar")
    async def llm_tool_get_avatar(self, event: AstrMessageEvent, user_id: str):
        '''
        获取QQ头像URL，供 generate_image 使用。
        仅在明确要画群友/头像/改头像时调用，不要把URL发送给用户。
        
        重要提示:
        1. 必须使用真实的QQ号，不要编造或猜测
        2. 如需获取群成员QQ号，请先调用 get_group_members_info 工具获取群成员列表
        3. 从消息中的[At:xxx]格式可直接提取被@用户的QQ号（xxx即为QQ号）
        4. 从 get_group_members_info 返回的 user_id 字段获取成员QQ号
        
        Args:
            user_id (string): 真实的QQ号（必须是数字）
        '''
        user_id = str(user_id).strip()
        if not user_id.isdigit():
            return f"错误：无效的QQ号 {user_id}"
        
        # 构造并返回头像URL（不发送消息，只返回给AI）
        # 使用q4.qlogo.cn与_get_avatar保持一致
        avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        return avatar_url
    
    
    
    # ================== 自动撤回 ==================
    
    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent, ctx=None):
        """消息发送后钩子 - 实现自动撤回"""
        if not self.config.get("enable_auto_recall", False):
            return
        
        try:
            result = event.get_result()
            if not result or not hasattr(result, 'chain'):
                return
            
            # 统一尝试撤回（不区分是否包含图片）
            delay = self.config.get("auto_recall_delay", 15)
            message_id = self._extract_message_id_from_obj(ctx)

            if not message_id and hasattr(event, "get_extra"):
                for key in ("message_id", "sent_message_id", "msg_id", "messageId", "result", "send_result"):
                    value = event.get_extra(key)
                    message_id = self._extract_message_id_from_obj(value)
                    if not message_id and isinstance(value, (str, int)):
                        message_id = value
                    if message_id:
                        break

            if not message_id:
                message_id = self._extract_message_id_from_obj(result)

            if not message_id:
                msg_obj = getattr(event, "message_obj", None)
                candidate = self._extract_message_id_from_obj(msg_obj)
                if candidate:
                    sender_id = getattr(getattr(msg_obj, "sender", None), "user_id", None)
                    self_id = event.get_self_id() if hasattr(event, "get_self_id") else None
                    if sender_id and self_id and str(sender_id) == str(self_id):
                        message_id = candidate

            if message_id and hasattr(event, 'bot'):
                if self.config.get("debug_mode", False):
                    logger.info(f"[AutoRecall] 将在 {delay}s 后撤回消息 {message_id}")
                
                async def delayed_recall():
                    await asyncio.sleep(delay)
                    try:
                        # 尝试调用平台的撤回方法
                        if hasattr(event.bot, 'recall_message'):
                            await event.bot.recall_message(message_id)
                            if self.config.get("debug_mode", False):
                                logger.info(f"[AutoRecall] 已撤回消息 {message_id}")
                        elif hasattr(event.bot, 'delete_msg'):
                            await event.bot.delete_msg(message_id=message_id)
                            if self.config.get("debug_mode", False):
                                logger.info(f"[AutoRecall] 已删除消息 {message_id}")
                    except Exception as e:
                        if self.config.get("debug_mode", False):
                            logger.warning(f"[AutoRecall] 撤回失败: {e}")
                
                task = asyncio.create_task(delayed_recall())
                self.pending_tasks.add(task)
                task.add_done_callback(self.pending_tasks.discard)
            elif self.config.get("debug_mode", False):
                logger.warning("[AutoRecall] 未能获取已发送消息的message_id，跳过撤回")
        except Exception as e:
            if self.config.get("debug_mode", False):
                logger.warning(f"[AutoRecall] 钩子执行出错: {e}")
    
    # ================== 生命周期 ==================
    
    async def terminate(self):
        """插件卸载"""
        # 取消所有pending的任务
        for task in list(self.pending_tasks):
            if not task.done():
                task.cancel()
        
        # 等待任务完成（包括被取消的）
        if self.pending_tasks:
            await asyncio.gather(*self.pending_tasks, return_exceptions=True)
        
        # 关闭HTTP session
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        
        logger.info("[RongheDraw] 插件已卸载，资源已清理")
