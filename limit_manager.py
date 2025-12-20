"""
RongheDraw 次数管理模块
支持白名单用户/群聊权限检查和每日使用次数限制
"""
import sqlite3
import datetime
import os
import json

# 使用标准数据目录（符合AstrBot规范）
_db_file = None
_db_initialized = False


def _get_db_file():
    """获取数据库文件路径（延迟初始化）"""
    global _db_file
    if _db_file is None:
        try:
            from astrbot.api.star import StarTools
            data_dir = StarTools.get_data_dir("astrbot_plugin_ronghedraw")
            _db_file = os.path.join(str(data_dir), "user_usage_data.db")
        except Exception:
            # 回退到插件目录（兼容旧版本）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            _db_file = os.path.join(current_dir, "user_usage_data.db")
    return _db_file


def _init_db():
    """初始化数据库（延迟调用）"""
    global _db_initialized
    if _db_initialized:
        return
    
    db_file = _get_db_file()
    # 确保目录存在
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    
    # 个人次数统计表
    c.execute('''CREATE TABLE IF NOT EXISTS usage_stats
                 (user_id TEXT PRIMARY KEY, count INTEGER, last_date TEXT)''')
    
    # 群级LLM统计表
    c.execute('''CREATE TABLE IF NOT EXISTS group_llm_usage
                 (group_id TEXT PRIMARY KEY, count INTEGER, last_date TEXT)''')
    
    conn.commit()
    conn.close()
    _db_initialized = True


def _get_connection():
    """获取数据库连接"""
    _init_db()
    return sqlite3.connect(_get_db_file())


def _parse_list(raw_list):
    """解析配置中的列表"""
    if isinstance(raw_list, list):
        return [str(x).strip() for x in raw_list if x]
    return []


def _parse_custom_limits(config):
    """解析自定义用户额度"""
    custom_limits_raw = config.get("custom_limits", {})
    if isinstance(custom_limits_raw, dict) and "default" in custom_limits_raw:
        custom_limits_raw = custom_limits_raw["default"]
    
    if isinstance(custom_limits_raw, str):
        try:
            if custom_limits_raw.strip():
                return json.loads(custom_limits_raw)
        except json.JSONDecodeError:
            pass
        return {}
    elif isinstance(custom_limits_raw, dict):
        return custom_limits_raw
    return {}


def is_user_whitelisted(user_id: str, config: dict) -> bool:
    """检查用户是否在白名单中"""
    whitelist = _parse_list(config.get("user_whitelist", []))
    return str(user_id).strip() in whitelist


def is_group_whitelisted(group_id: str, config: dict) -> bool:
    """检查群聊是否在白名单中"""
    if not group_id:
        return False
    whitelist = _parse_list(config.get("group_whitelist", []))
    return str(group_id).strip() in whitelist


def check_permission(user_id: str, group_id: str, requested_mode: str, config: dict) -> tuple:
    """
    检查用户权限
    返回: (是否允许, 实际使用的模式, 提示信息)
    """
    user_id = str(user_id).strip()
    
    # 白名单用户 - 无任何限制
    if is_user_whitelisted(user_id, config):
        return True, requested_mode, None
    
    # 白名单群聊 - 可用所有模式，但受次数限制
    if is_group_whitelisted(group_id, config):
        return True, requested_mode, None
    
    # 普通用户 - 只能使用 flow 模式
    if requested_mode != "flow":
        return False, "flow", "❌ 此命令需要白名单权限\n💡 普通用户请使用 #f文 或 #f图 命令"
    
    return True, "flow", None


def check_and_consume(user_id: str, group_id: str, config: dict) -> tuple:
    """
    检查并消耗次数
    返回: (是否允许, 提示信息)
    """
    user_id = str(user_id).strip()
    
    # 白名单用户不消耗次数
    if is_user_whitelisted(user_id, config):
        return True, "剩余: ∞"
    
    daily_limit = config.get("daily_limit", 5)
    custom_limits = _parse_custom_limits(config)
    user_limit = custom_limits.get(user_id, daily_limit)
    
    today_str = datetime.date.today().isoformat()
    conn = _get_connection()
    try:
        c = conn.cursor()
        
        c.execute("SELECT count, last_date FROM usage_stats WHERE user_id=?", (user_id,))
        row = c.fetchone()
        
        current_count = 0
        if row:
            if row[1] != today_str:
                c.execute("UPDATE usage_stats SET count=0, last_date=? WHERE user_id=?", (today_str, user_id))
            else:
                current_count = row[0]
        else:
            c.execute("INSERT INTO usage_stats (user_id, count, last_date) VALUES (?, 0, ?)", (user_id, today_str))
        
        if current_count >= user_limit:
            conn.commit()
            return False, f"今日额度已用尽 ({current_count}/{user_limit})，请明日再来"
        
        c.execute("UPDATE usage_stats SET count = count + 1 WHERE user_id=?", (user_id,))
        conn.commit()
    finally:
        conn.close()
    
    remaining = user_limit - (current_count + 1)
    return True, f"剩余: {remaining}/{user_limit}"


def get_user_remaining(user_id: str, config: dict) -> str:
    """查询用户剩余次数"""
    user_id = str(user_id).strip()
    
    if is_user_whitelisted(user_id, config):
        return "∞ (白名单)"
    
    daily_limit = config.get("daily_limit", 5)
    custom_limits = _parse_custom_limits(config)
    user_limit = custom_limits.get(user_id, daily_limit)
    
    today_str = datetime.date.today().isoformat()
    conn = _get_connection()
    try:
        c = conn.cursor()
        c.execute("SELECT count, last_date FROM usage_stats WHERE user_id=?", (user_id,))
        row = c.fetchone()
    finally:
        conn.close()
    
    if not row or row[1] != today_str:
        return f"{user_limit}/{user_limit}"
    
    remaining = max(0, user_limit - row[0])
    return f"{remaining}/{user_limit}"


def check_and_consume_group(group_id: str, config: dict) -> tuple:
    """
    群级次数检查（仅LLM工具使用）
    
    返回: (是否允许, 提示信息)
    """
    group_id = str(group_id).strip() if group_id else None
    
    # 私聊场景无群ID，返回错误
    if not group_id:
        return False, "LLM绘图功能仅在群聊中可用"
    
    # 检查群白名单
    llm_whitelist = _parse_list(config.get("llm_group_whitelist", []))
    if group_id in llm_whitelist:
        return True, "剩余: ∞"
    
    # 群统计
    limit = config.get("llm_group_daily_limit", 10)
    today_str = datetime.date.today().isoformat()
    
    conn = _get_connection()
    try:
        c = conn.cursor()
        
        c.execute("SELECT count, last_date FROM group_llm_usage WHERE group_id=?", (group_id,))
        row = c.fetchone()
        
        current_count = 0
        if row:
            if row[1] != today_str:
                c.execute("UPDATE group_llm_usage SET count=0, last_date=? WHERE group_id=?", (today_str, group_id))
            else:
                current_count = row[0]
        else:
            c.execute("INSERT INTO group_llm_usage VALUES (?, 0, ?)", (group_id, today_str))
        
        if current_count >= limit:
            conn.commit()
            return False, f"本群LLM绘图额度已用尽 ({current_count}/{limit})，请明日再来"
        
        c.execute("UPDATE group_llm_usage SET count = count + 1 WHERE group_id=?", (group_id,))
        conn.commit()
    finally:
        conn.close()
    
    remaining = limit - (current_count + 1)
    return True, f"本群剩余: {remaining}/{limit}"


def get_group_remaining(group_id: str, config: dict) -> str:
    """查询群LLM绘图剩余次数"""
    group_id = str(group_id).strip() if group_id else None
    
    if not group_id:
        return "N/A (仅群聊)"
    
    # 检查群白名单
    llm_whitelist = _parse_list(config.get("llm_group_whitelist", []))
    if group_id in llm_whitelist:
        return "∞ (白名单)"
    
    limit = config.get("llm_group_daily_limit", 10)
    today_str = datetime.date.today().isoformat()
    
    conn = _get_connection()
    try:
        c = conn.cursor()
        c.execute("SELECT count, last_date FROM group_llm_usage WHERE group_id=?", (group_id,))
        row = c.fetchone()
    finally:
        conn.close()
    
    if not row or row[1] != today_str:
        return f"{limit}/{limit}"
    
    remaining = max(0, limit - row[0])
    return f"{remaining}/{limit}"
