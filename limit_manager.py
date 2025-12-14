"""
RongheDraw 次数管理模块
支持白名单用户/群聊权限检查和每日使用次数限制
"""
import sqlite3
import datetime
import os
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(CURRENT_DIR, "user_usage_data.db")


def _init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS usage_stats
                 (user_id TEXT PRIMARY KEY, count INTEGER, last_date TEXT)''')
    conn.commit()
    conn.close()


_init_db()


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
    conn = sqlite3.connect(DB_FILE)
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
        conn.close()
        return False, f"今日额度已用尽 ({current_count}/{user_limit})，请明日再来"
    
    c.execute("UPDATE usage_stats SET count = count + 1 WHERE user_id=?", (user_id,))
    conn.commit()
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
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute("SELECT count, last_date FROM usage_stats WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    
    if not row or row[1] != today_str:
        return f"{user_limit}/{user_limit}"
    
    remaining = max(0, user_limit - row[0])
    return f"{remaining}/{user_limit}"
