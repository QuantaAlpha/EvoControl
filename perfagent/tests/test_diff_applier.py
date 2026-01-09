"""
DiffApplier 的 pytest 测试用例（SEARCH/REPLACE 区块格式）
"""

from perfagent.diff_applier import DiffApplier


class TestDiffApplier:
    """测试基于 SEARCH/REPLACE 的 Diff 应用器"""

    def test_extract_diffs_multiple_blocks(self):
        diff_text = (
            "<<<<<<< SEARCH\n"
            "line2\nline3\n"
            "=======\n"
            "LINE2\nLINE3\n"
            ">>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\n"
            "LINE3\n"
            "=======\n"
            "X3\n"
            ">>>>>>> REPLACE"
        )
        blocks = DiffApplier.extract_diffs(diff_text)
        assert len(blocks) == 2
        assert blocks[0] == ("line2\nline3", "LINE2\nLINE3")
        assert blocks[1] == ("LINE3", "X3")

    def test_validate_diff_blocks_and_apply_validated(self):
        original_code = "line1\nline2\nline3\nline4"
        diff_blocks = [
            ("line2\nline3", "LINE2\nLINE3"),  # valid
            ("missing", "added"),  # invalid
        ]

        valid_blocks = DiffApplier.validate_diff_blocks(original_code, diff_blocks)
        assert valid_blocks == [("line2\nline3", "LINE2\nLINE3")]

        applied = DiffApplier.apply_validated_diff_blocks(original_code, valid_blocks)
        assert applied == "line1\nLINE2\nLINE3\nline4"

    def test_apply_diff_with_search_replace_blocks(self):
        original_code = "line1\nline2\nline3\nline4"
        diff_text = (
            "<<<<<<< SEARCH\n"
            "line2\nline3\n"
            "=======\n"
            "LINE2\nLINE3\n"
            ">>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\n"
            "LINE3\n"
            "=======\n"
            "X3\n"
            ">>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\n"
            "not_in_code\n"
            "=======\n"
            "SHOULD_NOT_APPLY\n"
            ">>>>>>> REPLACE"
        )

        result = DiffApplier.apply_diff(original_code, diff_text)
        # 第二个区块在第一个区块应用后，将 LINE3 -> X3
        assert result == "line1\nLINE2\nX3\nline4"

    def test_apply_diff_no_blocks_returns_original(self):
        original_code = "a\nb\nc"
        diff_text = ""  # no blocks
        assert DiffApplier.apply_diff(original_code, diff_text) == original_code

        diff_text = "some unrelated text without markers"
        assert DiffApplier.apply_diff(original_code, diff_text) == original_code

    def test_format_diff_blocks_string(self):
        blocks = [("a\nb", "A\nB"), ("c", "C")]
        formatted = DiffApplier.format_diff_blocks_string(blocks)
        assert formatted is not None
        # 基本结构检查
        assert "Diff Block 1:" in formatted
        assert "<<<<<<< SEARCH" in formatted
        assert "=======" in formatted
        assert ">>>>>>> REPLACE" in formatted
        # 内容检查
        assert "a\nb" in formatted
        assert "A\nB" in formatted
        assert "Diff Block 2:" in formatted
        assert "c" in formatted
        assert "C" in formatted
