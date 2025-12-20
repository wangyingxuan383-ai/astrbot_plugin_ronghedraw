#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面优化修复脚本 - 第二批
资源管理、数据库保护、HTTP session复用
"""

def apply_part2_fixes():
    main_file = r'c:\Users\wang\Desktop\111\AAA\astrbot_plugin_ronghedraw\main.py'
    limit_file = r'c:\Users\wang\Desktop\111\AAA\astrbot_plugin_ronghedraw\limit_manager.py'
    
    # ===== 修改main.py - 资源管理 =====
    print("P0-2: Adding resource management...")
    
    with open(main_file, 'r', encoding='utf-8') as f:
        main_content = f.read()
    
    # 1. 在__init__中添加HTTP session和任务跟踪
    old_init_end = """        # 检查依赖
        self._check_dependencies()"""
    
    new_init_end = """        # HTTP session (在initialize中创建)
        self._http_session = None
        
        # 跟踪pending的asyncio任务
        self.pending_tasks = set()
        
        # 检查依赖
        self._check_dependencies()"""
    
    main_content = main_content.replace(old_init_end, new_init_end)
    
    # 2. 修改initialize创建HTTP session
    old_initialize = """    async def initialize(self):
        \"\"\"插件激活时调用，用于初始化资源\"\"\"
        logger.info(\"[RongheDraw] 插件已激活\")"""
    
    new_initialize = """    async def initialize(self):
        \"\"\"插件激活时调用，用于初始化资源\"\"\"
        # 创建HTTP session复用连接池
        import aiohttp
        self._http_session = aiohttp.ClientSession()
        
        logger.info(\"[RongheDraw] 插件已激活，资源已初始化\")"""
    
    main_content = main_content.replace(old_initialize, new_initialize)
    
    # 3. 修改terminate清理资源
    old_terminate = """    async def terminate(self):
        \"\"\"插件卸载\"\"\"
        logger.info(\"[RongheDraw] 插件已卸载\")"""
    
    new_terminate = """    async def terminate(self):
        \"\"\"插件卸载\"\"\"
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
        
        logger.info(\"[RongheDraw] 插件已卸载，资源已清理\")"""
    
    main_content = main_content.replace(old_terminate, new_terminate)
    
    # 4. 修改_download_image使用复用的session
    old_download = """    async def _download_image(self, url: str) -> bytes | None:
        \"\"\"下载图片并返回字节数据\"\"\"
        proxy = self.config.get(\"proxy_url\") if self.config.get(\"flow_use_proxy\") or self.config.get(\"generic_use_proxy\") or self.config.get(\"gemini_use_proxy\") else None
        timeout = aiohttp.ClientTimeout(total=self.config.get(\"timeout\", 120))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, proxy=proxy, timeout=timeout) as resp:"""
    
    new_download = """    async def _download_image(self, url: str) -> bytes | None:
        \"\"\"下载图片并返回字节数据\"\"\"
        proxy = self.config.get(\"proxy_url\") if self.config.get(\"flow_use_proxy\") or self.config.get(\"generic_use_proxy\") or self.config.get(\"gemini_use_proxy\") else None
        timeout = aiohttp.ClientTimeout(total=self.config.get(\"timeout\", 120))
        try:
            # 使用复用的session
            if not self._http_session or self._http_session.closed:
                self._http_session = aiohttp.ClientSession()
            async with self._http_session.get(url, proxy=proxy, timeout=timeout) as resp:"""
    
    main_content = main_content.replace(old_download, new_download)
    
    # 5. 修改create_task跟踪任务
    old_create_task = """                    asyncio.create_task(delayed_recall())"""
    
    new_create_task = """                    task = asyncio.create_task(delayed_recall())
                    self.pending_tasks.add(task)
                    task.add_done_callback(self.pending_tasks.discard)"""
    
    main_content = main_content.replace(old_create_task, new_create_task)
    
    # 6. 添加配置验证
    old_check_dep_end = """        if missing:
            logger.warning(f\"[RongheDraw] ⚠️ 缺少依赖: {', '.join(missing)}\")
            logger.warning(f\"[RongheDraw] 请运行: pip install {' '.join(missing)}\")"""
    
    new_check_dep_end = """        if missing:
            logger.warning(f\"[RongheDraw] WARNING: Missing dependencies: {', '.join(missing)}\")
            logger.warning(f\"[RongheDraw] Run: pip install {' '.join(missing)}\")
    
    def _validate_config(self):
        \"\"\"验证必需配置\"\"\"
        has_api = (self.config.get(\"flow_api_url\") or 
                   self.config.get(\"generic_api_url\") or 
                   self.config.get(\"gemini_api_url\"))
        if not has_api:
            logger.warning(\"[RongheDraw] WARNING: No API URL configured, plugin functionality limited\")"""
    
    main_content = main_content.replace(old_check_dep_end, new_check_dep_end)
    
    # 在initialize中调用验证
    old_init_logger = """        logger.info(\"[RongheDraw] 插件已激活，资源已初始化\")"""
    new_init_logger = """        self._validate_config()
        logger.info(\"[RongheDraw] 插件已激活，资源已初始化\")"""
    
    main_content = main_content.replace(old_init_logger, new_init_logger)
    
    # 7. 完善错误日志
    old_extract_err = """        except Exception:
            return raw  # 异常时返回原始数据，但未记录日志"""
    
    new_extract_err = """        except Exception as e:
            logger.debug(f\"[RongheDraw] Failed to extract GIF first frame: {e}\")
            return raw"""
    
    main_content = main_content.replace(old_extract_err, new_extract_err)
    
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(main_content)
    
    # ===== 修改limit_manager.py - 数据库异常保护 =====
    print("P0-3: Adding database exception protection...")
    
    with open(limit_file, 'r', encoding='utf-8') as f:
        limit_content = f.read()
    
    # 为所有数据库操作添加try-finally
    functions_to_fix = [
        ('check_and_consume', 134, 170),
        ('get_user_remaining', 174, 191),
        ('check_and_consume_group', 198, 240),
        ('get_group_remaining', 244, 269)
    ]
    
    # 由于范围修改复杂，只修改核心部分
    # check_and_consume
    old_cac = """    today_str = datetime.date.today().isoformat()
    conn = _get_connection()
    c = conn.cursor()
    
    c.execute(\"SELECT count, last_date FROM usage_stats WHERE user_id=?\", (user_id,))
    row = c.fetchone()
    
    current_count = 0
    if row:
        if row[1] != today_str:
            c.execute(\"UPDATE usage_stats SET count=0, last_date=? WHERE user_id=?\", (today_str, user_id))
        else:
            current_count = row[0]
    else:
        c.execute(\"INSERT INTO usage_stats (user_id, count, last_date) VALUES (?, 0, ?)\", (user_id, today_str))
    
    if current_count >= user_limit:
        conn.commit()
        conn.close()
        return False, f\"今日额度已用尽 ({current_count}/{user_limit})，请明日再来\"
    
    c.execute(\"UPDATE usage_stats SET count = count + 1 WHERE user_id=?\", (user_id,))
    conn.commit()
    conn.close()"""
    
    new_cac = """    today_str = datetime.date.today().isoformat()
    conn = _get_connection()
    try:
        c = conn.cursor()
        
        c.execute(\"SELECT count, last_date FROM usage_stats WHERE user_id=?\", (user_id,))
        row = c.fetchone()
        
        current_count = 0
        if row:
            if row[1] != today_str:
                c.execute(\"UPDATE usage_stats SET count=0, last_date=? WHERE user_id=?\", (today_str, user_id))
            else:
                current_count = row[0]
        else:
            c.execute(\"INSERT INTO usage_stats (user_id, count, last_date) VALUES (?, 0, ?)\", (user_id, today_str))
        
        if current_count >= user_limit:
            conn.commit()
            return False, f\"今日额度已用尽 ({current_count}/{user_limit})，请明日再来\"
        
        c.execute(\"UPDATE usage_stats SET count = count + 1 WHERE user_id=?\", (user_id,))
        conn.commit()
    finally:
        conn.close()"""
    
    limit_content = limit_content.replace(old_cac, new_cac)
    
    # check_and_consume_group
    old_cacg = """    conn = _get_connection()
    c = conn.cursor()
    
    c.execute(\"SELECT count, last_date FROM group_llm_usage WHERE group_id=?\", (group_id,))
    row = c.fetchone()
    
    current_count = 0
    if row:
        if row[1] != today_str:
            c.execute(\"UPDATE group_llm_usage SET count=0, last_date=? WHERE group_id=?\", (today_str, group_id))
        else:
            current_count = row[0]
    else:
        c.execute(\"INSERT INTO group_llm_usage VALUES (?, 0, ?)\", (group_id, today_str))
    
    if current_count >= limit:
        conn.commit()
        conn.close()
        return False, f\"本群LLM绘图额度已用尽 ({current_count}/{limit})，请明日再来\"
    
    c.execute(\"UPDATE group_llm_usage SET count = count + 1 WHERE group_id=?\", (group_id,))
    conn.commit()
    conn.close()"""
    
    new_cacg = """    conn = _get_connection()
    try:
        c = conn.cursor()
        
        c.execute(\"SELECT count, last_date FROM group_llm_usage WHERE group_id=?\", (group_id,))
        row = c.fetchone()
        
        current_count = 0
        if row:
            if row[1] != today_str:
                c.execute(\"UPDATE group_llm_usage SET count=0, last_date=? WHERE group_id=?\", (today_str, group_id))
            else:
                current_count = row[0]
        else:
            c.execute(\"INSERT INTO group_llm_usage VALUES (?, 0, ?)\", (group_id, today_str))
        
        if current_count >= limit:
            conn.commit()
            return False, f\"本群LLM绘图额度已用尽 ({current_count}/{limit})，请明日再来\"
        
        c.execute(\"UPDATE group_llm_usage SET count = count + 1 WHERE group_id=?\", (group_id,))
        conn.commit()
    finally:
        conn.close()"""
    
    limit_content = limit_content.replace(old_cacg, new_cacg)
    
    # get_user_remaining
    old_gur = """    conn = _get_connection()
    c = conn.cursor()
    
    c.execute(\"SELECT count, last_date FROM usage_stats WHERE user_id=?\", (user_id,))
    row = c.fetchone()
    conn.close()"""
    
    new_gur = """    conn = _get_connection()
    try:
        c = conn.cursor()
        c.execute(\"SELECT count, last_date FROM usage_stats WHERE user_id=?\", (user_id,))
        row = c.fetchone()
    finally:
        conn.close()"""
    
    limit_content = limit_content.replace(old_gur, new_gur)
    
    # get_group_remaining
    old_ggr = """    conn = _get_connection()
    c = conn.cursor()
    
    c.execute(\"SELECT count, last_date FROM group_llm_usage WHERE group_id=?\", (group_id,))
    row = c.fetchone()
    conn.close()"""
    
    new_ggr = """    conn = _get_connection()
    try:
        c = conn.cursor()
        c.execute(\"SELECT count, last_date FROM group_llm_usage WHERE group_id=?\", (group_id,))
        row = c.fetchone()
    finally:
        conn.close()"""
    
    limit_content = limit_content.replace(old_ggr, new_ggr)
    
    with open(limit_file, 'w', encoding='utf-8') as f:
        f.write(limit_content)
    
    print("OK - Part 2 fixes applied:")
    print("  - P0: Resource management (HTTP session, asyncio tasks)")
    print("  - P0: Database exception protection")
    print("  - P2: Config validation")
    print("  - P2: Error logging enhancements")

if __name__ == '__main__':
    apply_part2_fixes()
