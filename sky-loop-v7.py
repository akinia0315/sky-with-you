# sky-loop-v7.py — 光遇循环调度器 v7.0
#
# v6.0 → v7.0（感知层大换血）:
#   - 面板/弹窗/加载/场景检测全部交给 PanelDetector（panel_detector.py），
#     决策与动作只读 WorldState，不再自己看截图猜
#   - 删除 v6 的 panel_visual_open / 心跳恢复 / reopen 状态机
#     （"越按C越乱"的根源是感知不可靠，感知稳了状态机就不需要了）
#   - CS 状态机取消：面板开没开随时问 det.world.snapshot()，打字是发送动作内的瞬态
#   - 弹窗由检测器识别（模板+OCR），这里只负责"按空格接受"
#   - 聊天消息读取（内容 - 名字 解析 → AI 回复）沿用 v6 调优过的管线
#
# 启动:
#   1. 先开 MCP:  python sky-mcp-server.py --http --port 9900 --token 1234
#   2. 再开本程序: python sky-loop-v7.py
# -*- coding: utf-8 -*-

import os
# 限住推理库的线程数，必须在 onnxruntime/numpy 加载前设置。
# 不限的话 OCR 一跑就吃满全部核心，游戏和截图线程都被饿死（2026-07-02 实测爆卡）
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import openai
import requests
import json
import time
import re
import sys
import threading
import queue
from difflib import SequenceMatcher
import numpy as np
import cv2
from rapidocr_onnxruntime import RapidOCR

from panel_detector import (PanelDetector, DetectorConfig, Screen,
                            find_game_client_region, make_ocr_engine)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ===================== 配置 =====================

MCP_URL       = "http://127.0.0.1:9900"
MCP_TOKEN     = "1234"
def _load_api_key():
    """优先环境变量，其次脚本同目录的 key.txt（一次写好，重拷 v7 不再丢 key）"""
    k = os.environ.get("OPENROUTER_API_KEY", "")
    if k:
        return k
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "key.txt")
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""

API_KEY       = _load_api_key()
MODEL         = "anthropic/claude-sonnet-4.5"
SUMMARY_MODEL = "anthropic/claude-sonnet-4.5"

WATCH_INTERVAL   = 0.10
# 世界透过半透明面板永远在动 → 帧差几乎永远触发，cooldown 就是实际频率。
# 0.8s 会让 OCR 满负荷连轴转把游戏拖垮（2026-07-02 实测），聊天回复慢 2 秒无所谓。
OCR_COOLDOWN     = 3.0    # 帧差触发聊天OCR的最短间隔
OCR_FALLBACK     = 10.0   # 面板开着时的兜底OCR间隔
CONFIRM_COOLDOWN = 6.0    # 两次自动接弹窗的最短间隔
# 面板 = precious 的眼睛：确认关着且世界空闲时主动按 C 开回来。
# 没有它会死锁——读消息需要面板开，而开面板的唯一动机又是读到了消息。
REOPEN_COOLDOWN  = 6.0    # 两次开面板尝试的最短间隔
POST_ACTION_GRACE = 8.0   # 动作(接弹窗/牵手)刚结束的静默期，别立刻按C搅局
# 回家动作的硬冷却（保险丝）：不管 AI 说什么、HOME 判定灵不灵，
# 冷却内绝不第二次回家。07-03 深夜"无限回家循环"的兜底闸。
# 90s 会卡到 Aki 的连续实测（"非常久都不回"），45s 够拦住循环了
GOHOME_COOLDOWN  = 45.0

MAX_MSGS        = 40
SUMMARY_TRIGGER = 30
SUMMARY_KEEP    = 15

DEBUG = True

AKI = "Aki"
# 光遇不给自己显示名字（只能备注别人）：precious 自己发的消息在面板里是
# "无名字裸内容"行。extract 靠发送历史把这些行认出来，打上 [precious] 标签
# （SELF_NAMES[0]），既不污染别人消息正文，也永不触发 AI——
# 07-03 无限回家循环的根因就是这些裸行被粘进 Aki 消息开头，去重全废。
SELF_NAMES = ("precious",)

# ===================== System Prompt（与 v6 相同） =====================

def _load_persona() -> str:
    """人设外置：precious 是谁只住在你机器上，不进仓库。
    照 persona.example.txt 写一个 persona.txt 放在脚本旁边。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona.txt")
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""

SYSTEM = _load_persona()

# ===================== 停止信号 =====================

shutdown_event = threading.Event()

# ===================== MCP =====================

_rpc_counters = {}
_rpc_lock = threading.Lock()


def _next_rpc_id(name: str) -> int:
    with _rpc_lock:
        _rpc_counters[name] = _rpc_counters.get(name, 0) + 1
        return _rpc_counters[name]


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    if MCP_TOKEN:
        s.headers["Authorization"] = "Bearer " + MCP_TOKEN
    return s


def mcp(tool, args=None, session=None):
    sess = session or _make_session()
    payload = {
        "jsonrpc": "2.0", "id": _next_rpc_id(tool),
        "method": "tools/call",
        "params": {"name": tool, "arguments": args or {}}
    }
    for attempt in range(3):
        try:
            r = sess.post(MCP_URL, json=payload, timeout=30)
            d = r.json()
            if "error" in d:
                raise RuntimeError("MCP: " + str(d["error"]))
            res = d.get("result", {})
            text = "\n".join(
                i["text"] for i in res.get("content", [])
                if isinstance(i, dict) and i.get("type") == "text"
            )
            if res.get("isError"):
                # MCP 工具级错误（如前台守卫拒绝按键）走 isError 字段，
                # 不检查会被当成正常返回悄悄吞掉（07-03"键盘没动"事件）
                raise RuntimeError(f"MCP拒绝({tool}): {text}")
            return text
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            if attempt < 2:
                print(f"  [MCP] 超时，重试 ({attempt+2}/3)...")
                time.sleep(1)
            else:
                raise RuntimeError(f"MCP连续3次超时: {e}")

# ===================== SharedState（只剩对话相关） =====================

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.conversation: list[dict] = []
        # 近期已见消息 (文本, 时间)：去重只记"最后一条"会被旧消息 A/B 轮跳骗过
        # （07-03 回家后复读机 bug），改记 2 分钟窗口内的全部
        self.recent_seen: list[tuple[str, float]] = []
        # 最近发出的消息 (文本, 时间)，10 分钟窗口：extract 用它认领面板里
        # 无名字的裸行（光遇不显示自己名字，这是他自己消息的唯一特征）
        self.sent_history: list[tuple[str, float]] = []
        self.action_busy: bool = False
        self.last_confirm_time: float = 0
        self.last_chat_ocr_time: float = 0
        self.last_reopen_time: float = 0
        self.last_action_end: float = 0
        self.last_fhand_time: float = 0
        self.last_fhand_busy_log: float = 0
        self.last_gohome_time: float = 0

# ===================== 面板操作（读 WorldState，不再猜） =====================

def panel_open(det: PanelDetector) -> bool | None:
    """None = 加载中/看不清，此时不要做任何面板操作"""
    snap = det.world.snapshot()
    if snap.screen == Screen.LOADING:
        return None
    return snap.chat_open.value


def wait_panel(det, want: bool, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if panel_open(det) is want:
            return True
        time.sleep(0.05)
    return False


def _key(sess, k, ms=80):
    mcp("press_key", {"key": k, "duration_ms": ms, "backend": "arduino"},
        session=sess)


def _open_panel(det, sess) -> bool:
    if panel_open(det) is True:
        return True
    if det.world.snapshot().screen == Screen.LOADING:
        # 过场里盲按 C 会掐动画/落地瞬间误开面板，宁可这轮不开
        return False
    for i in range(2):
        _key(sess, "c", 150)
        if wait_panel(det, True, 1.2):
            return True
        if DEBUG:
            print(f"  [Action] 按C后面板未打开 (试{i+1}/2)")
    return panel_open(det) is True


def _close_panel(det, sess) -> bool:
    """尽力关面板。只有检测器等足时间后仍坚持"开着"才算失败。
    None（看不清，局部暗景常态）不算失败：ESC 已退出输入态，
    打字风险的源头是输入焦点（enter），不是面板开着本身。"""
    if panel_open(det) is not True:
        return True
    # 可能停留在输入态，先 ESC 退回面板态再 C 收起
    mcp("press_key", {"key": "escape", "duration_ms": 100, "backend": "arduino"},
        session=sess)
    time.sleep(0.15)
    _key(sess, "c", 150)
    # 判关需要 0.8s 持续证据 + 感知周期，1.2s 几乎必超时
    # （07-03 Aki："面板明明关了还说放弃"→回家/举火全卡死），给足 2.6s
    if wait_panel(det, False, 2.6):
        return True
    # 不追加按 C：C 是开关键，检测滞后时多按会把真关掉的面板又打开
    return panel_open(det) is not True


def _send_msg(det, sess, msg) -> bool:
    # 传送/回家过场中别碰键盘：enter 会把聊天框打开、后续按键变打字
    deadline = time.time() + 30.0
    while time.time() < deadline and not shutdown_event.is_set() \
            and det.world.snapshot().screen == Screen.LOADING:
        time.sleep(0.3)
    if not _open_panel(det, sess):
        print("  [Action] 打不开面板，放弃这条消息")
        return False
    for i in range(6):
        try:
            mcp("send_chat", {"message": msg, "backend": "arduino"}, session=sess)
            break
        except RuntimeError as e:
            if "前台" in str(e) and i < 5:
                # Aki 切出去看日志了：等她回游戏再发，最多等 25 秒
                if i == 0:
                    print("  [Action] 游戏不在前台，等回前台再发…")
                time.sleep(5)
                continue
            print(f"  [Action] 发送失败: {e}")
            return False
    time.sleep(0.2)
    # 发完退回面板态（保持面板开着以便继续读消息）
    mcp("press_key", {"key": "escape", "duration_ms": 100, "backend": "arduino"},
        session=sess)
    time.sleep(0.15)
    return True


def _wait_arrive(det, timeout: float, target: Screen | None = None) -> bool:
    """等传送/加载结束。target=None 时只要求离开 LOADING 并稳定"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if shutdown_event.is_set():
            return False
        snap = det.world.snapshot()
        if target is not None and snap.screen == target:
            return True
        if target is None and snap.screen in (Screen.IN_WORLD, Screen.HOME):
            return True
        time.sleep(0.3)
    return False


def _confirm_action(det, state, sess):
    """检测器看到确认弹窗 → 关面板、按空格接受、等落地、牵手"""
    snap = det.world.snapshot()
    print(f"  [Action] 接弹窗: kw={'/'.join(snap.confirm_keywords)}")
    # ⚠不关面板！_close_panel 的第一下是 ESC，而 ESC=弹窗的"取消"键——
    # 07-03 Aki 实测"弹窗一出来马上就被关掉"就是它干的。
    # 弹窗在场时会吃掉一切按键（Aki 实证 8+space 面板开着也能回家），
    # 面板开着不碍事；弹窗不在时 space 才危险（会打开输入框）
    snap = det.world.snapshot()
    if not snap.confirm_dialog.value:
        print("  [Action] 弹窗已经不在了，取消接受（space 会落进面板变打字）")
        return
    # 传送弹窗默认选中的是 ✗（Aki 07-03 实测：直接 space=拒绝），
    # 右箭头挪到 ✓ 再确认。需要 server 端 ARDUINO_KEY_MAP 有 "right"
    _key(sess, "right", 80)
    time.sleep(0.25)
    _key(sess, "space", 100)
    time.sleep(1.0)

    # 传送飞行动画不是加载屏，检测器看它=普通世界 → 原来这里立刻判"到了"，
    # f×3 按进动画把它掐掉，后续排队的发消息还会 enter 出聊天框（07-03 "fff"）。
    # 铁律：过场里不碰键盘。必须先见到 LOADING 才算真传送了
    deadline = time.time() + 8.0
    saw_loading = False
    while time.time() < deadline and not shutdown_event.is_set():
        if det.world.snapshot().screen == Screen.LOADING:
            saw_loading = True
            break
        time.sleep(0.3)
    if not saw_loading:
        print("  [Action] 8秒没见到传送过场（同图传送或没接上），不按F，牵手交给 Aki 伸手")
        return
    if not _wait_arrive(det, 25.0):
        print("  [Action] 25秒没落地，放弃后续动作")
        return
    # 不再盲按 F×3：没人伸手时按 F=打开好友树菜单，会卡在里面，
    # 退出得按 ESC 而 ESC 无界面时=游戏菜单，风险链没有尽头（07-03 Aki）。
    # 牵手走唯一安全路径：Aki 伸手 → f_hand 图标 → accept_f 单击 F 直接牵上
    print("  [Action] 到了！等 Aki 伸手（f_hand 自动接）")


def _go_home_action(det, state, sess):
    now = time.time()
    with state.lock:
        last = state.last_gohome_time
    if now - last < GOHOME_COOLDOWN:
        print(f"  [Action] {int(now - last)}s 前刚回过家，这次忽略（防循环保险丝）")
        return
    with state.lock:
        state.last_gohome_time = now
    if not _close_panel(det, sess):
        # 8+space 面板开着关着都能回家（弹窗默认选项就是回家，Aki 实测），
        # 序列已无 enter 零打字风险；关面板只为让到家判定(OCR/白屏)看得清
        print("  [Action] 面板可能还开着，不影响回家，继续")
    print("  [Action] 回家...")
    _key(sess, "8", 80);       time.sleep(0.5)
    _key(sess, "space", 100);  time.sleep(1.0)

    if _wait_arrive(det, 20.0, target=Screen.HOME):
        print("  [Action] 到遇境了！")
        time.sleep(1.2)   # 白屏/黑屏过场后再站稳一拍
        # 出生点脚下就是星盘按钮，这里按 F 会打开星盘界面（07-03 实测）。
        # 退开两步，牵手的 F 留给 Aki 主动发起——她牵他，不是他扑她。
        _key(sess, "s", 900); time.sleep(0.5)
        _key(sess, "s", 900); time.sleep(0.5)
        print("  [Action] 已退开星盘，等 Aki 过来牵手")
    else:
        # 没确认到家就别退步：8 若没生效、space 已把输入框打开的话，
        # 长按 s 会变成往聊天框里灌"ssss"（07-03 气泡实证）
        print("  [Action] 没等到遇境画面，不退步（防 s 变打字）")


ACT_KEYS = {'点火': '1', '收火': '1', '鞠躬': '2'}  # 火是开关键：再按一下 1 收起
# '牵手' 不在此表：F 的语义取决于脚下站着什么（点火/交互/好友树/牵手），
# 只有 f_hand 图标在场时按 F 才保证是牵手——在 _execute_reply 里特判


def _execute_reply(det, state, sess, reply):
    for m in re.finditer(r'\[CHAT\](.*?)\[/CHAT\]', reply, re.DOTALL):
        msg = m.group(1).strip()
        if msg:
            print(f"  [Action] 发消息: {msg}")
            if _send_msg(det, sess, msg):
                # 记两处：recent_seen 防当"新消息"触发；sent_history 给
                # extract 认领面板里的无名裸行（自己的消息不显示名字）
                with state.lock:
                    state.recent_seen.append((msg, time.time()))
                    state.sent_history.append((msg, time.time()))
            time.sleep(0.3)

    for m in re.finditer(r'\[ACT\](.*?)\[/ACT\]', reply, re.DOTALL):
        act = m.group(1).strip()
        if act == '回家牵手':
            if det.world.snapshot().screen == Screen.HOME:
                print("  [Action] 已经在遇境了，跳过回家")
                continue
            _go_home_action(det, state, sess)
            continue
        if act == '牵手':
            # 盲按 F 不保证是牵手（Aki 07-03 指出：可能变点火/交互/好友树）。
            # 等牵手图标最多 5s，等到才按；等不到就作罢，别乱按
            deadline = time.time() + 5.0
            while time.time() < deadline and not shutdown_event.is_set():
                snap = det.world.snapshot()
                if snap.f_prompt.value and snap.f_prompt_name == "f_hand":
                    print("  [Action] 牵手图标在，按 F 牵住")
                    _key(sess, "f", 120)
                    with state.lock:
                        # 记时间戳压住守望线程的 accept_f，防止图标 TTL
                        # 消散前再补一下 F（那一下会打开好友树）
                        state.last_fhand_time = time.time()
                    break
                time.sleep(0.2)
            else:
                print("  [Action] 没等到牵手图标，不盲按 F（Aki 还没伸手？）")
            continue
        k = ACT_KEYS.get(act)
        if not k:
            print(f"  [Action] 未知动作: {act}")
            continue
        if not _close_panel(det, sess):
            # 1/2 是纯游戏键，面板开着（输入未聚焦）也不会变打字，照做
            print(f"  [Action] 面板可能还开着，{act} 是游戏键，照做")
        print(f"  [Action] {act} -> {k}")
        _key(sess, k, 80); time.sleep(0.3)

    for m in re.finditer(r'\[KEY\](.*?)\[/KEY\]', reply, re.DOTALL):
        raw = m.group(1).strip()
        km = re.match(r'([a-zA-Z\-]+)\s*(\d+)?', raw)
        if not km:
            continue
        k = km.group(1).lower()
        ms = min(int(km.group(2) or 80), 3000)
        if not _close_panel(det, sess) and k in ("enter", "return", "space"):
            # enter 和 space 都会在面板开着时打开输入框（space 是 07-03
            # SSSS 事故真凶），其余键面板开着也安全
            print(f"  [Action] 面板没关掉，{k} 会打开聊天输入框，跳过")
            continue
        print(f"  [Action] 按键 {k} {ms}ms")
        _key(sess, k, ms); time.sleep(ms / 1000 + 0.05)

# ===================== 聊天 OCR（沿用 v6 调优管线） =====================

OCR_FIX = {
    '舍': '啥', '尔': '你', '咐': '吧', '巳': '已', '宄': '究',
    '迏': '达', '苎': '苦', '亇': '个', '対': '对', '呮': '只',
    '凊': '清', '莪': '我', '毎': '每', '飬': '养', '児': '儿',
    '経': '经', '気': '气', '関': '关', '臫': '自', '収': '收',
}

NAME_FIX = {
    '生人': '陌生人',
    '陌生': '陌生人',
    '陌牛人': '陌生人',
}


def fix(t):
    for w, r in OCR_FIX.items():
        t = t.replace(w, r)
    return t


def ocr_str(items):
    return "\n".join(
        fix(t["text"].strip()) for t in items
        if t.get("text", "").strip() and float(t.get("confidence", 0)) > 0.35
    )


_UI_SKIP = {
    '聊天', 'ESC', '退后', 'ENTER', '语音输入',
    'F', 'R', 'T', 'C', '发送', 'NTER', 'SPACE', 'ESCAPE',
}

_NAME_SEP_RE = re.compile(
    r'^(.+)\s*[\-‒–—―~〜]\s*(\S+)$')
# 孤儿名字行："-Aki" 这种。长消息换行后正文和落款被 OCR 拆成两条，
# 名字行单独出现（07-03 Aki 实测：长消息变无名氏认不出是她的）
_NAME_ONLY_RE = re.compile(r'^[\-‒–—―~〜]\s*(\S{1,16})$')


def _is_own_line(line: str, own_texts) -> bool:
    """无名行是不是 precious 自己发的消息（对照发送历史，容 OCR 错字/换行截断）"""
    n = _PUNCT_RE.sub('', line)
    if not n:
        return False
    for t in own_texts:
        tn = _PUNCT_RE.sub('', t)
        if not tn:
            continue
        if n == tn or (len(n) >= 2 and n in tn) or _msg_similar(line, t):
            return True
    return False


def extract(text, own_texts=()):
    msgs = []
    pending = ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line in _UI_SKIP or line.isdigit():
            continue
        # 光遇不显示自己的名字：precious 的消息是无名裸行，和"别人长消息的
        # 换行续行"长得一样。靠发送历史认领，否则会被粘进下一条别人消息的
        # 正文开头（07-03 无限回家循环：去重面对的全是被污染的新组合文本）
        if _is_own_line(line, own_texts):
            msgs.append(f"[{SELF_NAMES[0]}] {line}")
            pending = ""
            continue
        # 孤儿名字行（"-Aki" 或光秃秃 "Aki"）：把攒下的正文认领给这个名字
        mo = _NAME_ONLY_RE.match(line)
        if mo or line == AKI:
            name = AKI if line == AKI else NAME_FIX.get(mo.group(1), mo.group(1))
            if pending:
                msgs.append(f"[{name}] {pending.rstrip('-‒–—―~〜 ')}")
            pending = ""
            continue
        m = _NAME_SEP_RE.match(line)
        if not m:
            pending += line
            continue
        content = m.group(1).strip()
        name = m.group(2).strip()
        name = NAME_FIX.get(name, name)
        if pending:
            content = pending + content
            pending = ""
        if not content or not name:
            continue
        if all(c in '.·…●' for c in content):
            continue
        msgs.append(f"[{name}] {content}")
    return msgs


def _seen_before(state, msg: str, now: float) -> bool:
    """2 分钟窗口去重：防止面板里的旧消息因 OCR 抖动被反复当成新消息。
    窗口过期后同样的话仍能触发（Aki 隔几分钟再说一遍'哈哈'是新消息）。"""
    state.recent_seen = [(m, t) for m, t in state.recent_seen if now - t < 120]
    return any(_msg_similar(msg, m) for m, t in state.recent_seen)


_NAME_PREFIX_RE = re.compile(r'^\[[^\]]*\]\s*')


def _is_self_msg(msg: str) -> bool:
    m = re.match(r'^\[([^\]]*)\]', msg)
    return bool(m) and m.group(1) in SELF_NAMES


_PUNCT_RE = re.compile(r'[\s,.!?~·…，。！？～、；;:：]')


def _msg_similar(a: str, b: str, threshold=0.75) -> bool:
    """只比消息正文。带 [名字] 前缀比会把所有短消息判成同一条
    （"[Aki] 好"vs"[Aki] 走"相似度 0.87）——07-03 "消息发不出去"的元凶。
    标点先归一掉：OCR 最爱把"，"读成"."，4字消息一个标点错字就穿透 0.75。
    短正文（≤3字）相似度没有意义，改用全等。"""
    if not a or not b:
        return False
    a = _PUNCT_RE.sub('', _NAME_PREFIX_RE.sub('', a))
    b = _PUNCT_RE.sub('', _NAME_PREFIX_RE.sub('', b))
    if not a or not b:
        return a == b
    if min(len(a), len(b)) <= 3:
        return a == b
    return SequenceMatcher(None, a, b).ratio() > threshold


def read_chat_ocr(frame, ocr_engine, roi):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = roi
    img = frame[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # 首选灰度；效果够好就不再跑另外两个变体（省 2/3 的 OCR 开销，同机跑游戏时很关键）
    variants = [
        ("gray", gray_eq),
        ("color", img),
        ("th150", cv2.threshold(gray_eq, 150, 255, cv2.THRESH_BINARY)[1]),
    ]
    best, best_score = [], -1
    for name, im in variants:
        result, _ = ocr_engine(im)
        result = result or []
        if result:
            confs = [float(r[2]) for r in result]
            score = len(result) + sum(confs) / len(confs)
        else:
            score = 0
        if score > best_score:
            best_score, best = score, result
        if name == "gray" and len(result) >= 3 \
                and sum(confs) / len(confs) > 0.55:
            break
    if not best:
        return []
    return [{"text": str(r[1]), "confidence": float(r[2])} for r in best]

# ===================== 对话压缩（与 v6 相同） =====================

def trim(conv, client):
    if len(conv) <= SUMMARY_TRIGGER:
        return conv
    cut = len(conv) - SUMMARY_KEEP
    old_text = "\n".join(m["content"] for m in conv[:cut])
    try:
        s = client.chat.completions.create(
            model=SUMMARY_MODEL, max_tokens=200,
            messages=[{"role": "user",
                       "content": "用3-5句中文总结：\n" + old_text}]
        ).choices[0].message.content
        print(f"  [AI] 压缩{cut}条 -> {s[:60]}...")
        r = conv[cut:]
        if r and r[0]["role"] == "user":
            r[0]["content"] = f"[摘要] {s}\n\n{r[0]['content']}"
        else:
            r.insert(0, {"role": "user", "content": f"[摘要] {s}"})
        return r
    except Exception as e:
        print(f"  [AI] 压缩失败: {e}")
        r = conv[-MAX_MSGS:]
        while r and r[0]["role"] != "user":
            r.pop(0)
        return r

# ===================== WatchThread（替代 v6 的 Sense） =====================

def watch_loop(det: PanelDetector, state: SharedState,
               ocr_q: queue.Queue, action_q: queue.Queue):
    print("  [Watch] 线程启动")
    last_gray = None
    roi = det.cfg.roi_chat

    while not shutdown_event.is_set():
        t0 = time.time()
        try:
            snap = det.world.snapshot()

            if snap.screen == Screen.LOADING:
                last_gray = None
                time.sleep(WATCH_INTERVAL)
                continue

            # 弹窗 → 交给 Action（限频）
            with state.lock:
                busy = state.action_busy
                last_confirm = state.last_confirm_time
            if snap.confirm_dialog.value and not busy \
                    and t0 - last_confirm > CONFIRM_COOLDOWN:
                with state.lock:
                    state.last_confirm_time = t0
                try:
                    action_q.put_nowait(("confirm",))
                except queue.Full:
                    pass

            # Aki 伸手（牵手 F 图标）→ 按 F 接住。只认 f_hand 模板，
            # 点火图标(f_icon)不在此列——火由 AI 的[ACT]管
            if snap.f_prompt.value and snap.f_prompt_name == "f_hand":
                if busy:
                    # 看见了但动作线程正忙——这是 07-03"从没牵上"的重点嫌疑，
                    # 出声留证（限频，别刷屏）
                    with state.lock:
                        last_log = state.last_fhand_busy_log
                    if t0 - last_log > 3.0:
                        with state.lock:
                            state.last_fhand_busy_log = t0
                        print("  [Watch] 看到 Aki 伸手了，但动作线程正忙，接不了")
                else:
                    with state.lock:
                        last_fhand = state.last_fhand_time
                    if t0 - last_fhand > 6.0:
                        with state.lock:
                            state.last_fhand_time = t0
                        try:
                            action_q.put_nowait(("accept_f",))
                        except queue.Full:
                            pass

            # 聊天 OCR 调度
            with state.lock:
                last_ocr = state.last_chat_ocr_time
            if snap.chat_open.value:
                frame = det.latest_frame()
                need = False
                if frame is not None:
                    h, w = frame.shape[:2]
                    g = cv2.cvtColor(
                        frame[int(h*roi[1]):int(h*roi[3]),
                              int(w*roi[0]):int(w*roi[2])],
                        cv2.COLOR_BGR2GRAY)
                    g = cv2.GaussianBlur(g, (5, 5), 0)
                    if last_gray is None or g.shape != last_gray.shape:
                        need = True
                    else:
                        d = float(np.abs(g.astype(float)
                                         - last_gray.astype(float)).mean())
                        need = d > 4.5
                    last_gray = g
                if (need and t0 - last_ocr > OCR_COOLDOWN) \
                        or t0 - last_ocr > OCR_FALLBACK:
                    try:
                        ocr_q.put_nowait(("chat",))
                    except queue.Full:
                        pass
            else:
                # 面板关着时 ROI 里只有游戏世界，OCR 只会读出垃圾——不跑
                last_gray = None
                # 守门：面板确认关着（False，不是"看不清"的None）且世界空闲 → 按开
                with state.lock:
                    last_reopen = state.last_reopen_time
                    last_action = state.last_action_end
                if (snap.screen == Screen.IN_WORLD
                        and snap.chat_open.value is False
                        and not snap.confirm_dialog.value
                        and not busy
                        and action_q.empty()   # 队列里排着动作时别插队开面板
                                               # （关→守门开→动作又关 的乒乓，07-03 Aki 体感反馈）
                        and t0 - last_action > POST_ACTION_GRACE
                        and t0 - last_reopen > REOPEN_COOLDOWN):
                    with state.lock:
                        state.last_reopen_time = t0
                    try:
                        action_q.put_nowait(("reopen",))
                    except queue.Full:
                        pass

        except Exception as e:
            print(f"  [Watch] 异常: {e}")

        time.sleep(max(0, WATCH_INTERVAL - (time.time() - t0)))

    print("  [Watch] 线程退出")

# ===================== OcrThread =====================

def ocr_loop(det: PanelDetector, state: SharedState,
             ocr_q: queue.Queue, ai_q: queue.Queue):
    print("  [OCR] 线程启动")
    ocr_engine = make_ocr_engine()

    while not shutdown_event.is_set():
        try:
            kind = ocr_q.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            if kind[0] == "chat":
                _do_chat_ocr(det, state, ocr_engine, ai_q)
        except Exception as e:
            print(f"  [OCR] 异常: {e}")

    print("  [OCR] 线程退出")


def _do_chat_ocr(det, state, ocr_engine, ai_q):
    frame = det.latest_frame()
    if frame is None:
        return
    items = read_chat_ocr(frame, ocr_engine, det.cfg.roi_chat)

    with state.lock:
        state.last_chat_ocr_time = time.time()

    text = ocr_str(items)
    if DEBUG and text:
        print(f"  [OCR] {text.replace(chr(10), ' | ')[:160]}")

    now = time.time()
    with state.lock:
        state.sent_history = [(t, ts) for t, ts in state.sent_history
                              if now - ts < 600]
        own_texts = [t for t, ts in state.sent_history]
    msgs = extract(text, own_texts)
    if not msgs:
        return

    with state.lock:
        # "最新一条"只在别人的消息里选：precious 自己发的话 OCR 读回来
        # 绝不能当成触发源（07-03 无限回家循环的自产自销闭环）
        cur_latest = ""
        for m in reversed(msgs):
            if not _is_self_msg(m):
                cur_latest = m
                break
        cur_aki = ""
        for m in reversed(msgs):
            if m.startswith(f"[{AKI}]"):
                cur_aki = m
                break

        now = time.time()
        new_aki = bool(cur_aki) and not _seen_before(state, cur_aki, now)
        new_other = bool(cur_latest) and not _seen_before(state, cur_latest, now)
        # 本轮读到的全部消息都记进窗口——只记"最新"会被 OCR 抖动骗过：
        # 某条旧消息偶尔漏读，下一轮又冒出来就成了"新消息"
        for m in msgs:
            if not _seen_before(state, m, now):
                state.recent_seen.append((m, now))
        if not new_aki and not new_other:
            return
        priority = new_aki

    try:
        ai_q.put_nowait((msgs, priority))
    except queue.Full:
        try:
            ai_q.get_nowait()
        except queue.Empty:
            pass
        try:
            ai_q.put_nowait((msgs, priority))
        except queue.Full:
            pass

# ===================== AiThread（与 v6 相同） =====================

def ai_loop(state: SharedState, ai_q: queue.Queue,
            action_q: queue.Queue, client):
    print("  [AI] 线程启动")

    while not shutdown_event.is_set():
        try:
            msgs, priority = ai_q.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            content = "\n".join(msgs[-10:])
            if priority:
                content += "\n\n（秋刚说了新消息，优先回她）"
            else:
                content += "\n\n（上面没有秋的新消息。路人说的话可以不理。）"

            ts = time.strftime("%H:%M:%S")
            print(f"\n[{ts}] {' | '.join(msgs[-3:])[:80]}")

            with state.lock:
                state.conversation.append({"role": "user", "content": content})
                state.conversation = trim(state.conversation, client)
                conv_snapshot = (
                    [{"role": "system", "content": SYSTEM}]
                    + list(state.conversation)
                )

            reply = client.chat.completions.create(
                model=MODEL, max_tokens=300,
                messages=conv_snapshot,
            ).choices[0].message.content

            with state.lock:
                state.conversation.append(
                    {"role": "assistant", "content": reply})

            if "[IDLE]" in reply:
                print("  [AI] (idle)")
            else:
                print(f"  [AI] -> {reply[:120]}")
                action_q.put(("execute", reply))

        except Exception as e:
            print(f"  [AI] 异常: {e}")

    print("  [AI] 线程退出")

# ===================== ActionThread =====================

def action_loop(det: PanelDetector, state: SharedState,
                action_q: queue.Queue):
    print("  [Action] 线程启动")
    sess = _make_session()

    while not shutdown_event.is_set():
        try:
            cmd = action_q.get(timeout=1.0)
        except queue.Empty:
            continue

        with state.lock:
            state.action_busy = True
        try:
            if cmd[0] == "execute":
                _execute_reply(det, state, sess, cmd[1])
            elif cmd[0] == "confirm":
                _confirm_action(det, state, sess)
            elif cmd[0] == "go_home":
                _go_home_action(det, state, sess)
            elif cmd[0] == "accept_f":
                print("  [Action] Aki 伸手了，按 F 牵住")
                _close_panel(det, sess)
                _key(sess, "f", 120)
                time.sleep(0.8)
                snap = det.world.snapshot()
                if snap.f_prompt.value and snap.f_prompt_name == "f_hand":
                    print("  [Action] 图标还在，再按一次 F")
                    _key(sess, "f", 120)
            elif cmd[0] == "reopen":
                try:
                    # 按 C 前留案发现场：雨林式误判是"状态冻结"（面板开着
                    # 却一直判关），不翻转就没有 flip 存证，只有这里逮得到
                    det.dump_evidence("reopen")
                    if _open_panel(det, sess):
                        print("  [Action] 面板已按开（守门）")
                except RuntimeError as e:
                    if "前台" not in str(e):
                        raise
                    # 游戏不在前台时守门被拒是常态（Aki 在看日志），不刷屏
        except Exception as e:
            print(f"  [Action] 异常: {e}")
        finally:
            with state.lock:
                state.action_busy = False
                state.last_action_end = time.time()

    print("  [Action] 线程退出")

# ===================== 主函数 =====================

def main():
    client = openai.OpenAI(
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    print("光遇循环调度 v7.0 (PanelDetector + WorldState)")
    print("=" * 40)

    # 1. 定位游戏窗口（优先客户区，MCP 兜底）
    region = find_game_client_region(DetectorConfig.window_titles)
    if region:
        follow = True
        print(f"窗口(客户区): {region}")
    else:
        follow = False
        sess = _make_session()
        st = json.loads(mcp("status", session=sess))
        win = st.get("window")
        if not win:
            print("找不到光遇窗口，先开游戏")
            return
        region = {"left": win["left"], "top": win["top"],
                  "width": max(1, win["width"]), "height": max(1, win["height"])}
        print(f"窗口(MCP整窗，可能含标题栏): {region}")

    # 2. MCP 连通性
    try:
        st = json.loads(mcp("status"))
        print(f"MCP:  后端={st.get('input_backend')}")
    except Exception as e:
        print(f"MCP 连不上: {e}")
        print("先跑: python sky-mcp-server.py --http --port 9900 --token 1234")
        return

    # 3. 启动检测器（自采集 + 自带OCR线程；debug=True 时 chat 翻转自动存证）
    # 和游戏同机跑，参数放保守：截屏 ~7fps、全帧OCR 6秒一次
    cv2.setNumThreads(2)
    det = PanelDetector(region=region, follow_window=follow,
                        cfg=DetectorConfig(debug=DEBUG,
                                           sense_interval=0.15,
                                           ocr_full_interval=10.0))
    det.start()

    def on_change(name, old, new, world):
        ts = time.strftime("%H:%M:%S")
        extra = ""
        if name == "f_prompt" and new:
            extra = f" ({world.f_prompt_name})"   # f_icon=点火 / f_hand=牵手
        print(f"  [World] {name}: {old} -> {new}{extra}")
    det.world.on_change = on_change

    state = SharedState()
    ocr_q = queue.Queue(maxsize=2)
    ai_q = queue.Queue(maxsize=1)
    action_q = queue.Queue(maxsize=8)

    threads = [
        threading.Thread(target=watch_loop, args=(det, state, ocr_q, action_q),
                         name="Watch", daemon=True),
        threading.Thread(target=ocr_loop, args=(det, state, ocr_q, ai_q),
                         name="OCR", daemon=True),
        threading.Thread(target=ai_loop, args=(state, ai_q, action_q, client),
                         name="AI", daemon=True),
        threading.Thread(target=action_loop, args=(det, state, action_q),
                         name="Action", daemon=True),
    ]
    for t in threads:
        t.start()

    print("Ctrl+C 退出")
    print("=" * 40)

    try:
        while True:
            time.sleep(2.0)
            if DEBUG:
                print("  " + det.world.snapshot().describe())
    except KeyboardInterrupt:
        print("\n退出中...")
        shutdown_event.set()
        for t in threads:
            t.join(timeout=3.0)
        det.stop()
        print("已退出")


if __name__ == "__main__":
    main()
