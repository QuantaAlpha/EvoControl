"""
Diff 应用器模块

提供基于 SEARCH/REPLACE 区块格式的 diff 解析与应用。

支持以下方法：
- apply_diff(original_code, diff_text): 直接应用完整的 SEARCH/REPLACE diff 文本
- extract_diffs(diff_text): 从文本中提取 (search_text, replace_text) 区块列表
- validate_diff_blocks(original_code, diff_blocks): 校验哪些区块在原文中可匹配
- apply_validated_diff_blocks(original_code, diff_blocks): 应用已校验的区块
- format_diff_blocks_string(diff_blocks): 将区块列表格式化为完整 diff 文本
"""

import logging
import re


class DiffApplier:
    """Diff 应用器，支持 SEARCH/REPLACE 区块格式"""

    @staticmethod
    def extract_diffs(diff_text: str) -> list[tuple[str, str]]:
        """
        提取 diff 区块

        期望格式：
        <<<<<<< SEARCH\n
        ...search content...
        =======\n
        ...replace content...
        >>>>>>> REPLACE

        返回：[(search_text, replace_text), ...]
        """
        if not diff_text or not diff_text.strip():
            return []
        diff_pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
        diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
        # 去除尾部空白（但保留前导空白和中间换行）
        return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]

    @staticmethod
    def validate_diff_blocks(original_code: str, diff_blocks: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """
        校验区块是否能在原始代码中找到对应的 SEARCH 内容。
        仅返回可匹配的区块。
        """
        if not diff_blocks:
            return []
        original_lines = original_code.split("\n")
        valid_diff_blocks: list[tuple[str, str]] = []
        for search_text, replace_text in diff_blocks:
            search_lines = search_text.split("\n")
            found = False
            # 线性扫描匹配连续子序列
            for i in range(len(original_lines) - len(search_lines) + 1):
                if original_lines[i : i + len(search_lines)] == search_lines:
                    found = True
                    break
            if found:
                valid_diff_blocks.append((search_text, replace_text))
        return valid_diff_blocks

    @staticmethod
    def apply_validated_diff_blocks(original_code: str, diff_blocks: list[tuple[str, str]]) -> str:
        """
        直接应用已校验的区块到原始代码。
        """
        if not diff_blocks:
            return original_code
        original_lines = original_code.split("\n")
        result_lines = original_lines.copy()
        for search_text, replace_text in diff_blocks:
            search_lines = search_text.split("\n")
            replace_lines = replace_text.split("\n")
            # 找到起始位置并替换
            applied = False
            for i in range(len(result_lines) - len(search_lines) + 1):
                if result_lines[i : i + len(search_lines)] == search_lines:
                    result_lines[i : i + len(search_lines)] = replace_lines
                    applied = True
                    break
            if not applied:
                logging.debug("Diff block not applied (no match found): %r", search_text)
        return "\n".join(result_lines)

    @staticmethod
    def apply_diff(original_code: str, diff_text: str) -> str:
        """
        应用完整的 SEARCH/REPLACE diff 文本到原始代码。
        """
        try:
            diff_blocks = DiffApplier.extract_diffs(diff_text)
            if not diff_blocks:
                logging.warning("No SEARCH/REPLACE diff blocks found; returning original code.")
                return original_code
            # 按顺序应用每个区块（不预验证，以保持与提供用法一致）
            original_lines = original_code.split("\n")
            result_lines = original_lines.copy()
            for search_text, replace_text in diff_blocks:
                search_lines = search_text.split("\n")
                replace_lines = replace_text.split("\n")
                applied = False
                for i in range(len(result_lines) - len(search_lines) + 1):
                    if result_lines[i : i + len(search_lines)] == search_lines:
                        result_lines[i : i + len(search_lines)] = replace_lines
                        applied = True
                        break
                if not applied:
                    logging.debug("Diff block not applied (no match found): %r", search_text)
            return "\n".join(result_lines)
        except Exception as e:
            logging.error(f"应用 SEARCH/REPLACE diff 失败: {e}")
            return original_code

    @staticmethod
    def format_diff_blocks_string(diff_blocks: list[tuple[str, str]]) -> str | None:
        """
        将区块列表格式化为完整的 SEARCH/REPLACE diff 字符串。
        """
        if not diff_blocks:
            return None
        formatted_lines: list[str] = []
        for i, (search_text, replace_text) in enumerate(diff_blocks):
            formatted_lines.append(f"Diff Block {i + 1}:")
            formatted_lines.append("<<<<<<< SEARCH")
            formatted_lines.append(search_text)
            formatted_lines.append("=======")
            formatted_lines.append(replace_text)
            formatted_lines.append(">>>>>>> REPLACE")
            if i < len(diff_blocks) - 1:
                formatted_lines.append("")
        return "\n".join(formatted_lines)
