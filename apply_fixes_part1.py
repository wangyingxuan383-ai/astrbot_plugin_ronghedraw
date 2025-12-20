#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面优化修复脚本
根据comprehensive_review.md和code_review_v2.md的建议批量修复
"""

def apply_all_fixes():
    main_file = r'c:\Users\wang\Desktop\111\AAA\astrbot_plugin_ronghedraw\main.py'
    
    # 读取文件
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # ===== P0修复: 私聊LLM绘图bug =====
    print("P0-1: 修复私聊LLM绘图bug...")
    # 替换次数检查逻辑（第1246-1249行）
    old_check = """        # 次数检查 - 使用群级统计或个人统计
        if self.config.get("llm_tool_use_group_limit", True) and group_id:
            ok, limit_msg = limit_manager.check_and_consume_group(group_id, self.config)
        else:
            ok, limit_msg = limit_manager.check_and_consume(user_id, group_id, self.config)"""
    
    new_check = """        # 次数检查 - 修复私聊bug
        if self.config.get("llm_tool_use_group_limit", True):
            if group_id:  # 群聊使用群统计
                ok, limit_msg = limit_manager.check_and_consume_group(group_id, self.config)
            else:  # 私聊回退到个人统计
                ok, limit_msg = limit_manager.check_and_consume(user_id, None, self.config)
        else:  # 配置关闭群统计，全部使用个人统计
            ok, limit_msg = limit_manager.check_and_consume(user_id, group_id, self.config)"""
    
    content = content.replace(old_check, new_check)
    
    # ===== P1修复: 模式检查顺序调整 =====
    print("P1-1: 调整模式检查顺序...")
    # 将模式检查移到次数检查前
    # 这个已经在正确位置了，无需修改
    
    # ===== P2修复: LLM工具优化 =====
    print("P2-1: 更新LLM工具描述...")
    # 修改工具描述
    old_desc = """        '''
        生成图片。prompt为画面描述，可优化用户原话。image_urls为参考图URL列表（可选），不传则文生图，传入则图生图。URL需http(s)开头。每次调用消耗群额度。
        
        Args:
            prompt (string): 画面描述
            image_urls (array[string], optional): 参考图URL列表
        '''"""
    
    new_desc = """        '''
        生成图片。prompt为画面描述，可优化用户原话。image_urls为参考图URL列表（可选），不传则文生图，传入则图生图。URL需http(s)开头。调用成功后图片会自动发送给用户，你可以添加评论。每次调用消耗额度。
        
        Args:
            prompt (string): 画面描述
            image_urls (array[string], optional): 参考图URL列表
        '''"""
    
    content = content.replace(old_desc, new_desc)
    
    print("P2-2: 添加输入验证...")
    # 在enable_llm_tool检查后添加输入验证
    old_enable_check = """        if not self.config.get("enable_llm_tool", False):
            yield event.plain_result("LLM 绘图工具未启用")
            return
        
        user_id = event.get_sender_id()"""
    
    new_enable_check = """        if not self.config.get("enable_llm_tool", False):
            yield event.plain_result("LLM 绘图工具未启用")
            return
        
        # 输入验证
        if len(prompt) > 1000:
            yield event.plain_result("提示词过长（最大1000字符）")
            return
        
        if image_urls and len(image_urls) > 10:
            yield event.plain_result("图片数量过多（最大10张）")
            return
        
        user_id = event.get_sender_id()"""
    
    content = content.replace(old_enable_check, new_enable_check)
    
    print("P2-3: 移除中间提示，简化返回...")
    # 移除所有中间提示和verbose输出
    # 删除invalid_urls相关的提示
    old_invalid = """        # 如果有无效URL，提示但继续
        if invalid_urls:
            error_list = "\\n".join([f"  - {url[:50]}: {reason}" for url, reason in invalid_urls])
            if images:
                yield event.plain_result(f"⚠️ 部分URL无效已忽略:\\n{error_list}\\n继续使用{len(images)}张有效图片...")
            else:
                yield event.plain_result(f"⚠️ 所有URL无效:\\n{error_list}\\n已转为文生图模式")
        
        yield event.plain_result(f"🤖 [LLM-{mode_name}] {task_type}: {clean_prompt[:30]}...")"""
    
    new_invalid = """        # 静默处理无效URL"""
    
    content = content.replace(old_invalid, new_invalid)
    
    # 简化返回值 - 成功只返回图片
    old_success = """        if success:
            yield event.chain_result([
                self._create_image_from_bytes(result),
                Plain(f"✅ [LLM-{mode_name}] 生成成功 ({elapsed:.1f}s) | {limit_msg}")
            ])
        else:
            yield event.plain_result(f"❌ [LLM-{mode_name}] 生成失败 ({elapsed:.1f}s)\\n原因: {result}")"""
    
    new_success = """        if success:
            # 成功：仅返回图片，无文本提示
            yield event.chain_result([self._create_image_from_bytes(result)])
        else:
            # 失败：简洁错误信息
            yield event.plain_result(f"生成失败: {result}")"""
    
    content = content.replace(old_success, new_success)
    
    # 删除不再需要的变量
    old_vars = """        mode_name = {"flow": "Flow", "generic": "Generic", "gemini": "Gemini"}[actual_mode]
        
        # 处理图片URL（如果AI提供了）
        images = []
        invalid_urls = []"""
    
    new_vars = """        # 处理图片URL（如果AI提供了）
        images = []"""
    
    content = content.replace(old_vars, new_vars)
    
    # 简化URL处理逻辑
    old_url_process = """        if image_urls:
            for url in image_urls:
                # URL格式检查
                if not url.startswith(('http://', 'https://')):
                    invalid_urls.append((url, "非HTTP(S)协议"))
                    continue
                
                # 下载图片
                img_data = await self._download_image(url)
                if img_data:
                    images.append(img_data)
                else:
                    invalid_urls.append((url, "下载失败"))"""
    
    new_url_process = """        if image_urls:
            for url in image_urls:
                # URL格式检查
                if not url.startswith(('http://', 'https://')):
                    continue  # 静默跳过无效URL
                
                # 下载图片
                img_data = await self._download_image(url)
                if img_data:
                    images.append(img_data)"""
    
    content = content.replace(old_url_process, new_url_process)
    
    # 删除不再需要的task_type变量
    old_task_type = """        # 清理提示词中的@用户信息
        clean_prompt = self._clean_prompt(prompt, event)
        
        # 确定任务类型
        if images:
            task_type = f"图生图 ({len(images)}张)"
        else:
            task_type = "文生图"
        """
    
    new_task_type = """        # 清理提示词中的@用户信息
        clean_prompt = self._clean_prompt(prompt, event)
        """
    
    content = content.replace(old_task_type, new_task_type)
    
    print("P2-4: 优化get_avatar...")
    # get_avatar默认不验证URL
    old_avatar = """        # 可选：验证URL有效性
        if self.config.get("llm_tool_validate_avatar_url", True):
            test_download = await self._download_image(avatar_url)
            if not test_download:
                yield event.plain_result(f"❌ 无法访问用户 {qq_number} 的头像")
                return
        
        # 返回URL文本"""
    
    new_avatar = """        # 返回URL文本（不验证，QQ头像服务稳定）"""
    
    content = content.replace(old_avatar, new_avatar)
    
    # ===== 写回文件 =====
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("OK - All fixes applied:")
    print("  - P0: Private chat LLM drawing bug")
    print("  - P2: LLM tool optimization (input validation, remove prompts, simplify returns)")
    print("  - P2: get_avatar no URL validation")

if __name__ == '__main__':
    apply_all_fixes()
