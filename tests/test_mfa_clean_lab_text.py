from __future__ import annotations

import pytest

from scripts.pilot_run_mfa import clean_lab_text as pilot_clean
from scripts.pilot_run_mfa import cn_number
from scripts.run_aishell1_n25_mfa import clean_lab_text as n25_clean


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("楼市地市交相升温房价会不会再度暴涨",
         "楼 市 地 市 交 相 升 温 房 价 会 不 会 再 度 暴 涨"),
        ("虚增耕地质量不达标的分别占百分之十和百分之三十",
         "虚 增 耕 地 质 量 不 达 标 的 分 别 占 百 分 之 十 和 百 分 之 三 十"),
        ("2013年房价上涨30%",
         "二千零一十三 年 房 价 上 涨 百分之三十"),
        ("iPhone 15 Pro 售价8999元",
         "iPhone 十五 Pro 售 价 八千九百九十九 元"),
        ("温度从-5度升到25度",
         "温 度 从 五 度 升 到 二十五 度"),
        ("2026年8月14日召开",
         "二千零二十六 年 八 月 十四 日 召 开"),
    ],
)
def test_clean_lab_text_chinese_and_numerals(raw: str, expected: str) -> None:
    assert pilot_clean(raw) == expected
    assert n25_clean(raw) == expected


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, "零"),
        (1, "一"),
        (5, "五"),
        (10, "十"),
        (11, "十一"),
        (14, "十四"),
        (15, "十五"),
        (20, "二十"),
        (30, "三十"),
        (99, "九十九"),
        (100, "一百"),
        (101, "一百零一"),
        (110, "一百一十"),
        (201, "二百零一"),
        (999, "九百九十九"),
        (1000, "一千"),
        (1001, "一千零一"),
        (2013, "二千零一十三"),
        (2026, "二千零二十六"),
        (8999, "八千九百九十九"),
        (10000, "一万"),
        (12345, "一万二千三百四十五"),
        (100000, "十万"),
        (123456, "十二万三千四百五十六"),
        (1000000, "一百万"),
        (99999999, "九千九百九十九万九千九百九十九"),
    ],
)
def test_cn_number_readings(n: int, expected: str) -> None:
    assert cn_number(n) == expected


def test_clean_lab_text_keeps_latin_tokens_for_oov_flagging() -> None:
    out = pilot_clean("像IG那么有钱")
    assert "IG" in out.split()
