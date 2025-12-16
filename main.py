"""
RongheDraw 多模式绘图插件
支持 Flow/Generic/Gemini 三种 API 模式
作者: Antigravity
版本: 1.0.0
"""
import asyncio
import base64
import hashlib
import io
import json
import random
import re
import time
from datetime import datetime
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
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from . import limit_manager


@register(
    "astrbot_plugin_ronghedraw",
    "Antigravity",
    "RongheDraw 多模式绘图插件 - 支持 Flow/Generic/Gemini 三种 API 模式",
    "1.0.0",
    "https://github.com/Antigravity/astrbot_plugin_ronghedraw",
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
            "gemini": asyncio.Lock()
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
            logger.warning(f"[RongheDraw] ⚠️ 缺少依赖: {', '.join(missing)}")
            logger.warning(f"[RongheDraw] 请运行: pip install {' '.join(missing)}")
    
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
    
    # ================== 图片处理 ==================
    
    async def _download_image(self, url: str) -> bytes | None:
        """下载图片"""
        timeout = self.config.get("timeout", 120)
        proxy = self.config.get("proxy_url") if self.config.get("use_proxy") else None
        
        for i in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, proxy=proxy, timeout=timeout) as resp:
                        resp.raise_for_status()
                        return await resp.read()
            except Exception as e:
                if i < 2:
                    await asyncio.sleep(1)
                else:
                    logger.error(f"下载图片失败: {url}, 错误: {e}")
        return None
    
    async def _get_avatar(self, user_id: str) -> bytes | None:
        """获取用户头像"""
        if not str(user_id).isdigit():
            return None
        avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
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
        """获取消息中的所有图片（支持多图，合并回复图片、当前消息图片、@用户头像）"""
        images: List[bytes] = []
        at_users: List[str] = []
        
        chain = event.message_obj.message
        
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
        
        # 3. @用户头像（始终收集，与其他图片合并）
        for seg in chain:
            if isinstance(seg, At):
                at_users.append(str(seg.qq))
        
        for uid in at_users:
            if avatar := await self._get_avatar(uid):
                images.append(avatar)
        
        return images
    
    def _bytes_to_base64(self, data: bytes, mime: str = "image/jpeg") -> str:
        """转换为base64 URL格式"""
        b64 = base64.b64encode(data).decode()
        return f"data:{mime};base64,{b64}"
    
    def _compress_image(self, data: bytes, max_size: int = 1024, quality: int = 85) -> bytes:
        """压缩图片，限制最大尺寸"""
        if PILImage is None:
            return data
        try:
            img = PILImage.open(io.BytesIO(data))
            # 转换为RGB
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            # 限制最大尺寸
            width, height = img.size
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
            async with aiohttp.ClientSession() as session:
                params = {
                    "q": text, "from": "zh", "to": "en",
                    "appid": appid, "salt": salt, "sign": sign
                }
                async with session.get("https://fanyi-api.baidu.com/api/trans/vip/translate", 
                                       params=params, timeout=10) as resp:
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
        proxy = self.config.get("proxy_url") if self.config.get("flow_use_proxy") else None
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, headers=headers, 
                                        proxy=proxy, timeout=timeout) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return False, f"API错误 ({resp.status}): {text[:200]}"
                    
                    # 解析流式响应
                    full_content = ""
                    async for line in resp.content:
                        line = line.decode().strip()
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                chunk = json.loads(line[6:])
                                if "choices" in chunk and chunk["choices"]:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        full_content += delta["content"]
                            except Exception:
                                pass
                    
                    # 提取图片URL
                    url_match = re.search(r'https?://[^\s<>")\]]+', full_content)
                    if url_match:
                        img_url = url_match.group(0).rstrip(".,;:!?)")
                        img_data = await self._download_image(img_url)
                        if img_data:
                            return True, img_data
                        return False, f"图片下载失败: {img_url}"
                    
                    return False, f"未找到图片URL: {full_content[:200]}"
        
        except asyncio.TimeoutError:
            return False, "请求超时"
        except Exception as e:
            return False, f"请求异常: {e}"
    
    async def _call_generic_api(self, images: List[bytes], prompt: str) -> Tuple[bool, Any]:
        """调用Generic API (OpenAI通用格式，非流式)"""
        api_url = self.config.get("generic_api_url", "")
        api_key = await self._get_api_key("generic")
        model = self.config.get("generic_default_model", "nano-banana")
        
        if not api_url or not api_key:
            return False, "Generic API 未配置"
        
        # 构建消息（不压缩图片）
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
        
        # 非流式请求
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": 4000
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        timeout = self.config.get("timeout", 120)
        proxy = self.config.get("proxy_url") if self.config.get("generic_use_proxy") else None
        
        if self.config.get("debug_mode", False):
            logger.info(f"[Generic] 请求: model={model}, stream=False, images={len(images)}")
        
        try:
            async with aiohttp.ClientSession() as session:
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
                    
                    # 提取base64或URL
                    b64_match = re.search(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', full_content)
                    if b64_match:
                        try:
                            b64 = b64_match.group(0).split(",")[1]
                            return True, base64.b64decode(b64)
                        except Exception:
                            pass
                    
                    url_match = re.search(r'https?://[^\s<>")\\]]+', full_content)
                    if url_match:
                        img_url = url_match.group(0).rstrip(".,;:!?)")
                        img_data = await self._download_image(img_url)
                        if img_data:
                            return True, img_data
                    
                    return False, f"未找到图片: {full_content[:200]}"
        
        except asyncio.TimeoutError:
            return False, "请求超时"
        except Exception as e:
            return False, f"请求异常: {e}"
    
    async def _call_gemini_api(self, images: List[bytes], prompt: str) -> Tuple[bool, Any]:
        """调用Gemini官方API（支持4K分辨率）"""
        base_url = self.config.get("gemini_api_url", "https://generativelanguage.googleapis.com")
        api_key = await self._get_api_key("gemini")
        model = self.config.get("gemini_default_model", "gemini-2.5-flash-preview-image")
        resolution = self.config.get("gemini_resolution", "4K")
        
        if not api_key:
            return False, "Gemini API Key 未配置"
        
        base = base_url.rstrip("/")
        if not base.endswith("v1beta"):
            base += "/v1beta"
        final_url = f"{base}/models/{model}:generateContent"
        
        # 构建请求 - 添加分辨率设置
        resolution_map = {"1K": "1024x1024", "2K": "2048x2048", "4K": "4096x4096"}
        target_size = resolution_map.get(resolution, "4096x4096")
        
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
        
        # 构建生成配置 - 包含图片尺寸设置
        generation_config = {
            "maxOutputTokens": 8192,
            "responseModalities": ["image", "text"],
            "imageConfig": {
                "imageSize": resolution  # "1K", "2K", "4K"
            }
        }
        
        if self.config.get("debug_mode", False):
            logger.info(f"[Gemini] 请求: model={model}, resolution={resolution}, imageSize={resolution}")
        
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
        proxy = self.config.get("proxy_url") if self.config.get("gemini_use_proxy") else None
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(final_url, json=payload, headers=headers,
                                        proxy=proxy, timeout=timeout) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return False, f"API错误 ({resp.status}): {text[:200]}"
                    
                    data = await resp.json()
                    
                    if "error" in data:
                        return False, f"错误: {data['error']}"
                    
                    # 提取图片
                    try:
                        for candidate in data.get("candidates", []):
                            for part in candidate.get("content", {}).get("parts", []):
                                if "inlineData" in part:
                                    b64 = part["inlineData"]["data"]
                                    return True, base64.b64decode(b64)
                    except Exception:
                        pass
                    
                    return False, f"未找到图片: {str(data)[:200]}"
        
        except asyncio.TimeoutError:
            return False, "请求超时"
        except Exception as e:
            return False, f"请求异常: {e}"
    
    async def generate(self, mode: str, images: List[bytes], prompt: str) -> Tuple[bool, Any]:
        """统一生成入口"""
        if mode == "flow":
            return await self._call_flow_api(images, prompt)
        elif mode == "gemini":
            return await self._call_gemini_api(images, prompt)
        else:
            return await self._call_generic_api(images, prompt)
    
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
    
    async def _send_and_recall(self, event: AstrMessageEvent, text: str):
        """发送消息并计划撤回"""
        result = event.plain_result(text)
        # 注：实际撤回需要平台支持，这里只是预留接口
        return result
    
    # ================== 命令处理 ==================
    
    def _parse_mode_from_command(self, cmd: str) -> Tuple[str, str]:
        """从命令解析模式和实际命令"""
        if cmd.startswith("f"):
            return "flow", cmd[1:]
        elif cmd.startswith("o"):
            return "generic", cmd[1:]
        elif cmd.startswith("g"):
            return "gemini", cmd[1:]
        return None, cmd  # 无前缀
    
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
        
        # 普通用户默认用flow
        return "flow"
    
    # ================== 文生图命令 ==================
    
    @filter.command("f文", alias={"f文生图"}, prefix_optional=True)
    async def cmd_flow_text2img(self, event: AstrMessageEvent):
        """Flow模式文生图"""
        async for result in self._handle_text2img(event, "flow"):
            yield result
    
    @filter.command("o文", alias={"o文生图"}, prefix_optional=True)
    async def cmd_generic_text2img(self, event: AstrMessageEvent):
        """Generic模式文生图"""
        async for result in self._handle_text2img(event, "generic"):
            yield result
    
    @filter.command("g文", alias={"g文生图"}, prefix_optional=True)
    async def cmd_gemini_text2img(self, event: AstrMessageEvent):
        """Gemini模式文生图"""
        async for result in self._handle_text2img(event, "gemini"):
            yield result
    
    @filter.command("文生图", alias={"文"}, prefix_optional=True)
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
        
        # 提取提示词并清理@用户信息
        raw = event.message_str.strip()
        prompt = re.sub(r'^[fog]?文(生图)?\s*', '', raw, flags=re.IGNORECASE).strip()
        prompt = self._clean_prompt(prompt, event)
        
        if not prompt:
            yield event.plain_result("❌ 请输入描述\n用法: #f文 一只可爱的猫")
            return
        
        # 次数检查
        ok, limit_msg = limit_manager.check_and_consume(user_id, group_id, self.config)
        if not ok:
            yield event.plain_result(f"❌ {limit_msg}")
            return
        
        mode_name = {"flow": "Flow", "generic": "Generic", "gemini": "Gemini"}[actual_mode]
        
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
                yield event.chain_result([
                    Image.fromBytes(result),
                    Plain(f"✅ [{mode_name}] 生成成功 ({elapsed:.1f}s) | {limit_msg}")
                ])
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
    
    @filter.command("f图", alias={"f图生图"}, prefix_optional=True)
    async def cmd_flow_img2img(self, event: AstrMessageEvent):
        """Flow模式图生图"""
        async for result in self._handle_img2img(event, "flow"):
            yield result
    
    @filter.command("o图", alias={"o图生图"}, prefix_optional=True)
    async def cmd_generic_img2img(self, event: AstrMessageEvent):
        """Generic模式图生图"""
        async for result in self._handle_img2img(event, "generic"):
            yield result
    
    @filter.command("g图", alias={"g图生图"}, prefix_optional=True)
    async def cmd_gemini_img2img(self, event: AstrMessageEvent):
        """Gemini模式图生图"""
        async for result in self._handle_img2img(event, "gemini"):
            yield result
    
    @filter.command("图生图", alias={"图"}, prefix_optional=True)
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
        
        # 提取提示词并清理@用户信息
        raw = event.message_str.strip()
        prompt = re.sub(r'^[fog]?图(生图)?\s*', '', raw, flags=re.IGNORECASE).strip()
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
        
        mode_name = {"flow": "Flow", "generic": "Generic", "gemini": "Gemini"}[actual_mode]
        
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
                yield event.chain_result([
                    Image.fromBytes(result),
                    Plain(f"✅ [{mode_name}] 生成成功 ({elapsed:.1f}s) | {limit_msg}")
                ])
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
        
        text = event.message_str.strip()
        if not text:
            return
        
        # 提取命令词（第一个token）
        tokens = text.split()
        if not tokens:
            return
        
        raw_cmd = tokens[0].strip()
        
        # 解析命令前缀 (f/o/g) 和基础命令
        prefix_mode = None
        base_cmd = raw_cmd
        
        if len(raw_cmd) > 1:
            first_char = raw_cmd[0].lower()
            if first_char in ('f', 'o', 'g'):
                # 检查去掉前缀后的命令是否在自定义预设中
                potential_cmd = raw_cmd[1:]
                if potential_cmd in self.prompt_map:
                    prefix_mode = {"f": "flow", "o": "generic", "g": "gemini"}.get(first_char)
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

    
    @filter.command("f手办化", prefix_optional=True)
    async def cmd_flow_figurine(self, event: AstrMessageEvent):
        async for r in self._handle_preset(event, "flow", "手办化"):
            yield r
    
    @filter.command("o手办化", prefix_optional=True)
    async def cmd_generic_figurine(self, event: AstrMessageEvent):
        async for r in self._handle_preset(event, "generic", "手办化"):
            yield r
    
    @filter.command("g手办化", prefix_optional=True)
    async def cmd_gemini_figurine(self, event: AstrMessageEvent):
        async for r in self._handle_preset(event, "gemini", "手办化"):
            yield r
    
    @filter.command("手办化", prefix_optional=True)
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
        
        mode_name = {"flow": "Flow", "generic": "Generic", "gemini": "Gemini"}[actual_mode]
        
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
                yield event.chain_result([
                    Image.fromBytes(result),
                    Plain(f"✅ [{mode_name}] {preset_name}成功 ({elapsed:.1f}s) | {limit_msg}")
                ])
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
    
    @filter.command("查询次数", prefix_optional=True)
    async def cmd_query_limit(self, event: AstrMessageEvent):
        """查询剩余次数"""
        user_id = event.get_sender_id()
        remaining = limit_manager.get_user_remaining(user_id, self.config)
        yield event.plain_result(f"👤 用户: {user_id}\n📊 今日剩余: {remaining}")
    
    @filter.command("f切换模型", prefix_optional=True)
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
    
    @filter.command("f翻译开关", prefix_optional=True)
    async def cmd_toggle_translate(self, event: AstrMessageEvent):
        """切换翻译功能"""
        current = self.config.get("flow_enable_translate", False)
        self.config["flow_enable_translate"] = not current
        status = "开启" if not current else "关闭"
        yield event.plain_result(f"🌐 翻译功能已{status}")
    
    @filter.command("预设列表", prefix_optional=True)
    async def cmd_list_presets(self, event: AstrMessageEvent):
        """列出所有预设"""
        builtin = list(self.builtin_presets.keys())
        custom = list(self.prompt_map.keys())
        
        msg = "📜 可用预设列表\n━━━━━━━━━━\n"
        msg += f"📌 内置: {', '.join(builtin)}\n"
        msg += f"✨ 自定义: {', '.join(custom) if custom else '(无)'}\n"
        msg += "━━━━━━━━━━\n用法: #f<预设名> [图片]"
        
        yield event.plain_result(msg)
    
    @filter.command("生图帮助", prefix_optional=True)
    async def cmd_help(self, event: AstrMessageEvent):
        """显示帮助"""
        help_text = self.config.get("help_text", "帮助未配置")
        yield event.plain_result(help_text)
    
    @filter.command("生图菜单", prefix_optional=True)
    async def cmd_menu(self, event: AstrMessageEvent):
        """显示菜单"""
        menu = """🎨 RongheDraw 绘图插件 v1.0.0

━━━━ 📌 快速开始 ━━━━
#f文 <描述>      文字生成图片
#f图 [图片]      图片风格转换
#f随机 [图片]    随机预设效果

━━━━ 🔀 API模式 ━━━━
f = Flow (自动横竖版，支持翻译)
o = Generic (仅白名单)
g = Gemini (仅白名单, 4K输出)

例: #o文 <描述>  #g图 [图片]

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
#f切换模型 | #f翻译开关"""
        yield event.plain_result(menu)
    
    # ================== 随机预设命令 ==================
    
    def _get_all_presets(self) -> list:
        """获取所有可用预设（内置+自定义）"""
        all_presets = list(self.builtin_presets.keys()) + list(self.prompt_map.keys())
        return all_presets if all_presets else []
    
    @filter.command("f随机", prefix_optional=True)
    async def cmd_flow_random(self, event: AstrMessageEvent):
        """Flow模式随机预设"""
        all_presets = self._get_all_presets()
        if not all_presets:
            yield event.plain_result("❌ 暂无可用预设")
            return
        preset = random.choice(all_presets)
        async for r in self._handle_preset(event, "flow", preset):
            yield r
    
    @filter.command("o随机", prefix_optional=True)
    async def cmd_generic_random(self, event: AstrMessageEvent):
        """Generic模式随机预设"""
        all_presets = self._get_all_presets()
        if not all_presets:
            yield event.plain_result("❌ 暂无可用预设")
            return
        preset = random.choice(all_presets)
        async for r in self._handle_preset(event, "generic", preset):
            yield r
    
    @filter.command("g随机", prefix_optional=True)
    async def cmd_gemini_random(self, event: AstrMessageEvent):
        """Gemini模式随机预设"""
        all_presets = self._get_all_presets()
        if not all_presets:
            yield event.plain_result("❌ 暂无可用预设")
            return
        preset = random.choice(all_presets)
        async for r in self._handle_preset(event, "gemini", preset):
            yield r
    
    @filter.command("随机", alias={"随机预设"}, prefix_optional=True)
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
    async def llm_tool_generate_image(self, event: AstrMessageEvent, prompt: str):
        '''
        根据描述生成图片。当用户请求绘制、生成、创作图片时调用此工具。
        
        Args:
            prompt (string): 图片描述，描述你想生成的图片内容
        '''
        if not self.config.get("enable_llm_tool", False):
            yield event.plain_result("LLM 绘图工具未启用")
            return
        
        user_id = event.get_sender_id()
        group_id = event.get_group_id()
        
        # 确定使用的模式
        if limit_manager.is_user_whitelisted(user_id, self.config) or \
           limit_manager.is_group_whitelisted(group_id, self.config):
            mode = self.config.get("llm_default_mode", "generic")
        else:
            mode = "flow"
        
        # 权限检查
        allowed, actual_mode, err_msg = limit_manager.check_permission(user_id, group_id, mode, self.config)
        if not allowed:
            yield event.plain_result(err_msg)
            return
        
        # 次数检查
        ok, limit_msg = limit_manager.check_and_consume(user_id, group_id, self.config)
        if not ok:
            yield event.plain_result(f"❌ {limit_msg}")
            return
        
        mode_name = {"flow": "Flow", "generic": "Generic", "gemini": "Gemini"}[actual_mode]
        
        # 获取消息中的图片（支持图生图）
        images = await self.get_images(event)
        
        if images:
            yield event.plain_result(f"🤖 [LLM-{mode_name}] 图生图: {prompt[:30]}...")
        else:
            yield event.plain_result(f"🤖 [LLM-{mode_name}] 文生图: {prompt[:30]}...")
        
        start = time.time()
        success, result = await self.generate(actual_mode, images, prompt)
        elapsed = time.time() - start
        
        if success:
            yield event.chain_result([
                Image.fromBytes(result),
                Plain(f"✅ [LLM-{mode_name}] 生成成功 ({elapsed:.1f}s) | {limit_msg}")
            ])
        else:
            yield event.plain_result(f"❌ [LLM-{mode_name}] 生成失败 ({elapsed:.1f}s)\n原因: {result}")
    
    @filter.llm_tool(name="get_avatar")
    async def llm_tool_get_avatar(self, event: AstrMessageEvent, qq_number: str):
        '''
        通过QQ号获取用户头像图片。用于获取指定用户的头像进行绘图或展示。
        
        Args:
            qq_number (string): QQ号码，纯数字字符串
        '''
        # 获取头像是通用功能，不受绘图开关限制
        
        qq_number = str(qq_number).strip()
        if not qq_number.isdigit():
            yield event.plain_result(f"❌ 无效的QQ号: {qq_number}")
            return
        
        avatar = await self._get_avatar(qq_number)
        if avatar:
            yield event.chain_result([
                Image.fromBytes(avatar),
                Plain(f"✅ 已获取用户 {qq_number} 的头像")
            ])
        else:
            yield event.plain_result(f"❌ 获取头像失败: {qq_number}")
    
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
            
            # 检查消息链中是否包含图片
            has_image = False
            for comp in result.chain:
                if isinstance(comp, Image):
                    has_image = True
                    break
            
            # 如果没有图片（纯文本消息），延迟后撤回
            if not has_image:
                delay = self.config.get("auto_recall_delay", 15)
                message_id = event.message_obj.message_id if hasattr(event.message_obj, 'message_id') else None
                
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
                    
                    asyncio.create_task(delayed_recall())
        except Exception as e:
            if self.config.get("debug_mode", False):
                logger.warning(f"[AutoRecall] 钩子执行出错: {e}")
    
    # ================== 生命周期 ==================
    
    async def terminate(self):
        """插件卸载"""
        logger.info("[RongheDraw] 插件已卸载")

