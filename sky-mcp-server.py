#!/usr/bin/env python3
"""
Sky MCP Server for PC Sky.

This server is for the Windows/PC version of Sky running on the same computer.
It exposes MCP tools for keyboard control, chat, screenshots, and OCR.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import io
import json
import os
import secrets
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

SERVER_INFO = {"name": "sky-mcp-server", "version": "0.2.0-gpu-merge"}
PROTOCOL_VERSION = "2025-03-26"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class SkyError(RuntimeError):
    pass


def log(message: str) -> None:
    sys.stderr.write(f"[sky-mcp] {message}\n")
    sys.stderr.flush()


def load_optional(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 9800
    token: str | None = None
    allow_unsafe_http: bool = False
    window_title: str | None = None
    monitor: int = 1
    input_backend: str = "auto"
    screenshot_scale: float = 1.0
    screenshot_max_width: int = 1920
    screenshot_max_height: int = 1080
    serial_port: str | None = None


class PcSkyController:
    def __init__(self, config: ServerConfig):
        self.config = config
        self._pyautogui = None
        self._pydirectinput = None
        self._mss = None
        self._image_cls = None
        self._pyperclip = None
        self._ocr_name = "none"
        self._ocr_device = "unknown"
        self._ocr_engine = None
        self._gamepad = None
        self._vgamepad = None
        self._arduino = None
        self._serial_lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        return {
            "mode": "pc",
            "platform": sys.platform,
            "input_backend": self._select_input_backend_name(),
            "ocr": self._detect_ocr_name(),
            "ocr_device": self._ocr_device,
            "screenshot_max": {
                "width": self.config.screenshot_max_width,
                "height": self.config.screenshot_max_height,
            },
            "window": self.find_window(),
        }

    def ensure_capture_deps(self) -> None:
        if self._mss is None:
            self._mss = load_optional("mss")
        if self._image_cls is None:
            pil = load_optional("PIL.Image")
            self._image_cls = pil
        if self._mss is None:
            raise SkyError("Missing dependency: mss. Run: pip install -r requirements.txt")
        if self._image_cls is None:
            raise SkyError("Missing dependency: Pillow. Run: pip install -r requirements.txt")

    def _select_input_backend_name(self) -> str:
        return self._resolve_input_backend_name()

    def _resolve_input_backend_name(self, backend: str | None = None) -> str:
        backend = (backend or self.config.input_backend or "auto").lower().strip()
        if backend != "auto":
            if backend not in {"pyautogui", "pydirectinput", "gamepad", "arduino"}:
                raise SkyError(f"Unsupported input backend: {backend}")
            return backend
        if sys.platform.startswith("win") and load_optional("pydirectinput") is not None:
            return "pydirectinput"
        return "pyautogui"

    def _input_module(self, backend: str | None = None):
        backend = self._resolve_input_backend_name(backend)
        if backend == "pydirectinput":
            if self._pydirectinput is None:
                self._pydirectinput = load_optional("pydirectinput")
            if self._pydirectinput is None:
                raise SkyError("Missing dependency: pydirectinput. Run: pip install -r requirements.txt")
            self._pydirectinput.PAUSE = 0.01
            return self._pydirectinput
        if self._pyautogui is None:
            self._pyautogui = load_optional("pyautogui")
        if self._pyautogui is None:
            raise SkyError("Missing dependency: pyautogui. Run: pip install -r requirements.txt")
        self._pyautogui.FAILSAFE = False
        self._pyautogui.PAUSE = 0.01
        return self._pyautogui

    def _ensure_gamepad(self):
        if self._gamepad is not None:
            return self._gamepad
        if self._vgamepad is None:
            self._vgamepad = load_optional("vgamepad")
        if self._vgamepad is None:
            raise SkyError("Missing dependency: vgamepad. Run: pip install vgamepad")
        self._gamepad = self._vgamepad.VX360Gamepad()
        log("Virtual Xbox 360 gamepad connected (persistent)")
        return self._gamepad

    GAMEPAD_MAP = {
        "w":     ("stick", (0.0, 1.0)),
        "s":     ("stick", (0.0, -1.0)),
        "a":     ("stick", (-1.0, 0.0)),
        "d":     ("stick", (1.0, 0.0)),
        "space": ("btn", "XUSB_GAMEPAD_A"),
        "f":     ("btn", "XUSB_GAMEPAD_X"),
        "q":     ("btn", "XUSB_GAMEPAD_B"),
        "tab":   ("btn", "XUSB_GAMEPAD_Y"),
        "e":     ("btn", "XUSB_GAMEPAD_RIGHT_SHOULDER"),
        "shift": ("btn", "XUSB_GAMEPAD_LEFT_SHOULDER"),
        "shift-space": ("btn_combo", ("XUSB_GAMEPAD_LEFT_SHOULDER", "XUSB_GAMEPAD_A")),
    }

    def _gamepad_press(self, key: str, duration_ms: int = 80) -> str:
        pad = self._ensure_gamepad()
        vg = self._vgamepad
        mapping = self.GAMEPAD_MAP.get(key)
        if mapping is None:
            raise SkyError(f"No gamepad mapping for key: {key}. Available: {list(self.GAMEPAD_MAP.keys())}")
        duration = max(0, int(duration_ms)) / 1000
        kind = mapping[0]
        value = mapping[1]

        if kind == "stick":
            x, y = value
            pad.left_joystick_float(x_value_float=x, y_value_float=y)
            pad.update()
            time.sleep(duration)
            pad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            pad.update()
        elif kind == "btn":
            btn = getattr(vg.XUSB_BUTTON, value, None)
            if btn is None:
                raise SkyError(f"Unknown gamepad button: {value}")
            pad.press_button(button=btn)
            pad.update()
            time.sleep(duration)
            pad.release_button(button=btn)
            pad.update()
        elif kind == "btn_combo":
            btns = [getattr(vg.XUSB_BUTTON, b) for b in value]
            for b in btns:
                pad.press_button(button=b)
            pad.update()
            time.sleep(duration)
            for b in btns:
                pad.release_button(button=b)
            pad.update()

        return f"gamepad {key} for {int(duration_ms)}ms"

    def _ensure_arduino(self):
        if self._arduino is not None:
            return self._arduino
        port = self.config.serial_port
        if not port:
            raise SkyError("No --serial-port specified for arduino backend")
        import serial
        self._arduino = serial.Serial(port, 115200, timeout=5)
        ready = self._arduino.readline().decode().strip()
        log(f"Arduino connected on {port}: {ready}")
        # v2 固件支持 RELEASE（松开所有键）：每次连上先清一遍悬键，
        # 让"出错断开→重连"成为真正的自愈。v1 旧固件回 ERR，无害。
        try:
            self._arduino.write(b"RELEASE\n")
            self._arduino.readline()
        except Exception:
            pass
        return self._arduino

    # 游戏窗口标题（精确前缀，不用泛匹配——"Sky"会误中 cmd 标题里的 sky-mcp-server）
    GAME_TITLE_PREFIXES = ("光·遇", "Sky: Children of the Light")

    def _game_is_foreground(self) -> bool:
        """开发板是真键盘，按键会进当前前台窗口。游戏不在前台时必须拒绝按键，
        否则会把 C / Ctrl+V 按进 Aki 的桌面（2026-07-03 两次'整机卡死'的元凶）。"""
        if sys.platform != "win32":
            return True
        try:
            import ctypes
            u32 = ctypes.windll.user32
            buf = ctypes.create_unicode_buffer(256)
            u32.GetWindowTextW(u32.GetForegroundWindow(), buf, 256)
            title = buf.value
            return any(title.startswith(p) for p in self.GAME_TITLE_PREFIXES)
        except Exception:
            return True  # 检查本身失败时不拦截，避免误伤

    def _drop_arduino(self) -> None:
        """断开串口。下次指令重开串口时 DTR 会触发板子复位 → 固件重启自动松开所有键。"""
        try:
            if self._arduino is not None:
                self._arduino.close()
        except Exception:
            pass
        self._arduino = None

    def _arduino_cmd(self, cmd: str) -> None:
        # ThreadingHTTPServer 是并发的：不加锁时两个请求的字节会在串口交错，
        # 板子解析出乱码指令，最坏情况把 Ctrl 等键永久卡死（键盘鼠标全瘫）。
        # 任何异常/坏应答一律断开串口，靠重连复位自愈。
        if not self._game_is_foreground():
            raise SkyError("游戏不在前台，拒绝按键（安全保护：防止按进桌面/其他窗口）")
        with self._serial_lock:
            try:
                s = self._ensure_arduino()
                s.write(cmd.encode())
                resp = s.readline().decode().strip()
            except SkyError:
                raise
            except Exception:
                self._drop_arduino()
                raise
            if resp != "OK":
                self._drop_arduino()
                raise SkyError(f"Arduino error: {resp!r}（已断开串口，下次调用自动复位重连）")

    def _arduino_press(self, keycode: int, duration_ms: int = 80) -> None:
        self._arduino_cmd(f"PRESS {keycode} {duration_ms}\n")

    def _arduino_hotkey(self, key1: int, key2: int, duration_ms: int = 50) -> None:
        self._arduino_cmd(f"HOTKEY {key1} {key2} {duration_ms}\n")

    ARDUINO_KEY_MAP = {
    "enter": 176,
    "escape": 177,
    "tab": 179,
    "space": 32,
    "w": 119, "a": 97, "s": 115, "d": 100,
    "f": 102, "q": 113, "e": 101, "c": 99,
    "shift": 129,
    "1": 49, "2": 50, "3": 51, "4": 52,
    "5": 53, "6": 54, "7": 55, "8": 56,
    "9": 57, "0": 48,
    # 方向键（Arduino Keyboard 库键码）。固件 PRESS 透传任意键码，无需重烧
    "right": 215, "left": 216, "down": 217, "up": 218,
}

    def _clipboard(self):
        if self._pyperclip is None:
            self._pyperclip = load_optional("pyperclip")
        if self._pyperclip is None:
            raise SkyError("Missing dependency: pyperclip. Run: pip install -r requirements.txt")
        return self._pyperclip

    def _hotkey(self, *keys: str, duration_ms: int = 30, backend: str | None = None) -> None:
        self._press_combo([normalize_key(key) for key in keys], duration_ms, backend=backend)

    def _mouse_module(self):
        if self._pyautogui is None:
            self._pyautogui = load_optional("pyautogui")
        if self._pyautogui is None:
            raise SkyError("Missing dependency: pyautogui. Run: pip install -r requirements.txt")
        self._pyautogui.FAILSAFE = False
        self._pyautogui.PAUSE = 0.01
        return self._pyautogui

    def _press_combo(self, keys: list[str], duration_ms: int = 30, backend: str | None = None) -> None:
        if not keys:
            return
        inp = self._input_module(backend)
        duration = max(0, int(duration_ms)) / 1000
        modifiers = keys[:-1]
        main = keys[-1]
        for mod in modifiers:
            inp.keyDown(mod)
        try:
            inp.keyDown(main)
            time.sleep(duration)
            inp.keyUp(main)
        finally:
            for mod in reversed(modifiers):
                try:
                    inp.keyUp(mod)
                except Exception:
                    pass

    def _tap_key(self, key: str, duration_ms: int = 35, backend: str | None = None) -> None:
        key = normalize_key(key)
        if "+" in key or "-" in key:
            parts = [normalize_key(p) for p in key.replace("+", "-").split("-") if p]
            self._press_combo(parts, duration_ms, backend=backend)
            return
        self._press_combo([key], duration_ms, backend=backend)

    def _paste_text(self, message: str, backend: str | None = None) -> None:
        clip = self._clipboard()
        clip.copy(message)
        time.sleep(0.05)
        resolved = self._resolve_input_backend_name(backend)
        if resolved == "arduino":
            self._arduino_hotkey(128, 118, 50)
        elif sys.platform == "darwin":
            self._hotkey("command", "v", backend=backend)
        else:
            self._hotkey("ctrl", "v", backend=backend)
        time.sleep(0.08)

    def _foreground_window(self) -> dict[str, Any] | None:
        if not sys.platform.startswith("win"):
            return None
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return None
            buf = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd, buf, len(buf))
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            return {"hwnd": int(hwnd), "title": buf.value, "pid": int(pid.value)}
        except Exception:
            return None

    def _force_foreground_window(self, hwnd: int) -> None:
        if not sys.platform.startswith("win"):
            return

        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        sw_restore = 9
        hwnd_topmost = -1
        hwnd_notopmost = -2
        swp_no_move = 0x0002
        swp_no_size = 0x0001
        swp_show_window = 0x0040
        vk_menu = 0x12
        keyeventf_keyup = 0x0002

        user32.ShowWindow(hwnd, sw_restore)
        user32.SetWindowPos(hwnd, hwnd_topmost, 0, 0, 0, 0, swp_no_move | swp_no_size | swp_show_window)
        user32.SetWindowPos(hwnd, hwnd_notopmost, 0, 0, 0, 0, swp_no_move | swp_no_size | swp_show_window)
        user32.BringWindowToTop(hwnd)

        user32.keybd_event(vk_menu, 0, 0, 0)
        user32.SetForegroundWindow(hwnd)
        user32.keybd_event(vk_menu, 0, keyeventf_keyup, 0)
        time.sleep(0.05)

        foreground = self._foreground_window()
        if foreground and foreground.get("hwnd") == hwnd:
            return

        try:
            foreground_hwnd = user32.GetForegroundWindow()
            foreground_pid = wintypes.DWORD()
            target_pid = wintypes.DWORD()
            foreground_thread = user32.GetWindowThreadProcessId(foreground_hwnd, ctypes.byref(foreground_pid))
            target_thread = user32.GetWindowThreadProcessId(hwnd, ctypes.byref(target_pid))
            current_thread = kernel32.GetCurrentThreadId()

            for thread_id in {foreground_thread, target_thread}:
                if thread_id:
                    user32.AttachThreadInput(current_thread, thread_id, True)
            try:
                user32.ShowWindow(hwnd, sw_restore)
                user32.BringWindowToTop(hwnd)
                user32.SetActiveWindow(hwnd)
                user32.SetFocus(hwnd)
                user32.SetForegroundWindow(hwnd)
            finally:
                for thread_id in {foreground_thread, target_thread}:
                    if thread_id:
                        user32.AttachThreadInput(current_thread, thread_id, False)
        except Exception:
            pass

    def _window_handle(self, win) -> int | None:
        hwnd = getattr(win, "_hWnd", None) or getattr(win, "hWnd", None)
        return int(hwnd) if hwnd else None

    def _click_window_center(self, win_info: dict[str, Any]) -> None:
        mouse = self._mouse_module()
        x = int(win_info["left"] + win_info["width"] / 2)
        y = int(win_info["top"] + win_info["height"] / 2)
        mouse.click(x, y)

    def find_window(self) -> dict[str, Any] | None:
        titles = [self.config.window_title] if self.config.window_title else [
            "光·遇",
            "光遇",
            "Sky",
            "Sky: Children of the Light",
            "Sky Children of the Light",
            "光遇",
        ]
        try:
            import pygetwindow as gw
        except Exception:
            return None

        for title in [t for t in titles if t]:
            for win in gw.getWindowsWithTitle(title):
                if win.isMinimized:
                    continue
                return {
                    "title": win.title,
                    "left": int(win.left),
                    "top": int(win.top),
                    "width": int(win.width),
                    "height": int(win.height),
                }
        return None

    def focus_game(self) -> dict[str, Any]:
        try:
            import pygetwindow as gw
        except Exception as exc:
            raise SkyError(f"pygetwindow is required to focus the Sky window: {exc}") from exc

        win_info = self.find_window()
        if not win_info:
            raise SkyError("Sky window not found. Start Sky first, or pass --window-title.")
        matches = gw.getWindowsWithTitle(win_info["title"])
        if not matches:
            raise SkyError("Sky window disappeared before focus.")
        win = matches[0]
        hwnd = self._window_handle(win)
        if win.isMinimized:
            win.restore()
        try:
            win.activate()
        except Exception as exc:
            if not (sys.platform.startswith("win") and hwnd):
                raise SkyError(f"Failed to activate Sky window: {exc}") from exc
            try:
                import ctypes
                user32 = ctypes.windll.user32
                user32.ShowWindow(hwnd, 9)
                user32.SetForegroundWindow(hwnd)
            except Exception as fallback_exc:
                raise SkyError(f"Failed to activate Sky window: {fallback_exc}") from fallback_exc
        if sys.platform.startswith("win") and hwnd:
            self._force_foreground_window(hwnd)
        time.sleep(0.15)
        foreground = self._foreground_window()
        if sys.platform.startswith("win") and hwnd and (not foreground or foreground.get("hwnd") != hwnd):
            self._click_window_center(win_info)
            time.sleep(0.15)
            foreground = self._foreground_window()
        if sys.platform.startswith("win") and hwnd and foreground and foreground.get("hwnd") != hwnd:
            raise SkyError(f"Sky window found but not focused. Foreground window is: {foreground.get('title')}")
        return {**win_info, "focused": True, "foreground": foreground}

    def ensure_game_foreground(self) -> dict[str, Any]:
        win_info = self.find_window()
        if not win_info:
            raise SkyError("Sky window not found. Start Sky first, or pass --window-title.")
        foreground = self._foreground_window()
        if not sys.platform.startswith("win"):
            return {**win_info, "focused": True, "foreground": foreground}
        try:
            import pygetwindow as gw
        except Exception as exc:
            raise SkyError(f"pygetwindow is required to verify the Sky window: {exc}") from exc
        matches = gw.getWindowsWithTitle(win_info["title"])
        hwnd = self._window_handle(matches[0]) if matches else None
        if hwnd and foreground and foreground.get("hwnd") != hwnd:
            raise SkyError(f"Sky window is not foreground. Foreground window is: {foreground.get('title')}")
        return {**win_info, "focused": True, "foreground": foreground}

    def press_key(self, key: str, duration_ms: int = 80, backend: str | None = None) -> str:
        self.focus_game()
        key = normalize_key(key)
        resolved = self._resolve_input_backend_name(backend)
        if resolved == "gamepad":
            return self._gamepad_press(key, duration_ms)
        if resolved == "arduino":
            kc = self.ARDUINO_KEY_MAP.get(key)
            if kc is None:
                raise SkyError(f"No arduino mapping for key: {key}. Available: {list(self.ARDUINO_KEY_MAP.keys())}")
            self._arduino_press(kc, duration_ms)
            return f"arduino {key} for {int(duration_ms)}ms"
        self._tap_key(key, duration_ms, backend=backend)
        return f"pressed {key} for {int(duration_ms)}ms via {resolved}"

    def open_chat(self, key: str = "enter", duration_ms: int = 35, backend: str | None = None) -> str:
        self.focus_game()
        key = normalize_key(key)
        resolved = self._resolve_input_backend_name(backend)
        if resolved == "arduino":
            kc = self.ARDUINO_KEY_MAP.get(key, 176)
            self._arduino_press(kc, duration_ms)
            return f"arduino tapped {key} for {int(duration_ms)}ms"
        self._tap_key(key, duration_ms, backend=backend)
        return f"tapped {key} for {int(duration_ms)}ms via {resolved}"

    def type_text(
        self,
        message: str,
        send: bool = True,
        enter_tap_ms: int = 35,
        backend: str | None = None,
        require_foreground: bool = True,
    ) -> str:
        if not isinstance(message, str) or not message.strip():
            raise SkyError("message must be a non-empty string")
        if len(message) > 240:
            raise SkyError("message is too long; keep it under 240 characters")

        if require_foreground:
            self.ensure_game_foreground()
        self._paste_text(message, backend=backend)
        if send:
            resolved = self._resolve_input_backend_name(backend)
            if resolved == "arduino":
                self._arduino_press(176, enter_tap_ms)
            else:
                self._tap_key("enter", enter_tap_ms, backend=backend)
        action = "sent" if send else "typed"
        return f"{action} text: {message} via {self._resolve_input_backend_name(backend)}"

    def send_chat(
        self,
        message: str,
        open_key: str = "enter",
        open_delay_ms: int = 180,
        assume_open: bool = False,
        send: bool = True,
        enter_tap_ms: int = 35,
        backend: str | None = None,
    ) -> str:
        if not isinstance(message, str) or not message.strip():
            raise SkyError("message must be a non-empty string")
        if len(message) > 240:
            raise SkyError("message is too long; keep it under 240 characters")

        if assume_open:
            self.ensure_game_foreground()
        else:
            self.focus_game()
            resolved = self._resolve_input_backend_name(backend)
            if resolved == "arduino":
                kc = self.ARDUINO_KEY_MAP.get(normalize_key(open_key), 176)
                self._arduino_press(kc, enter_tap_ms)
            else:
                self._tap_key(open_key, enter_tap_ms, backend=backend)
            time.sleep(max(0, open_delay_ms) / 1000)
        self._paste_text(message, backend=backend)
        if send:
            resolved_send = self._resolve_input_backend_name(backend)
            if resolved_send == "arduino":
                self._arduino_press(176, enter_tap_ms)
            else:
                self._tap_key("enter", enter_tap_ms, backend=backend)
        action = "sent" if send else "typed"
        return f"{action} chat: {message} via {self._resolve_input_backend_name(backend)}"

    def _limit_image_size(self, image):
        max_width = max(1, int(self.config.screenshot_max_width))
        max_height = max(1, int(self.config.screenshot_max_height))
        scale = min(max_width / image.width, max_height / image.height, 1)
        if scale >= 1:
            return image
        resampling = getattr(self._image_cls, "Resampling", self._image_cls).LANCZOS
        return image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            resampling,
        )

    def screenshot_image(self):
        self.ensure_capture_deps()
        with self._mss.mss() as screen:
            win = self.find_window()
            if win:
                region = {
                    "left": win["left"],
                    "top": win["top"],
                    "width": max(1, win["width"]),
                    "height": max(1, win["height"]),
                }
            else:
                monitors = screen.monitors
                index = min(max(1, self.config.monitor), len(monitors) - 1)
                region = monitors[index]
            shot = screen.grab(region)
            image = self._image_cls.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        if self.config.screenshot_scale != 1.0:
            scale = min(max(self.config.screenshot_scale, 0.2), 4.0)
            resampling = getattr(self._image_cls, "Resampling", self._image_cls).LANCZOS
            image = image.resize((int(image.width * scale), int(image.height * scale)), resampling)
        return self._limit_image_size(image)

    def screenshot_base64(self) -> str:
        image = self.screenshot_image()
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def read_screen(self) -> dict[str, Any]:
        image = self.screenshot_image()
        with tempfile.NamedTemporaryFile(prefix="sky-mcp-", suffix=".png", delete=False) as tmp:
            path = tmp.name
        try:
            image.save(path)
            return {
                "ocr": self._detect_ocr_name(),
                "ocr_device": self._ocr_device,
                "image_size": {"width": image.width, "height": image.height},
                "texts": self._run_ocr(path),
            }
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _detect_ocr_name(self) -> str:
        if self._ocr_engine is not None:
            return self._ocr_name
        try:
            os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR
            try:
                import paddle
                if paddle.device.is_compiled_with_cuda():
                    paddle.set_device("gpu:0")
                self._ocr_device = paddle.get_device()
            except Exception:
                self._ocr_device = "unknown"

            self._ocr_engine = PaddleOCR(use_angle_cls=False, lang="ch", det_limit_side_len=1920)

            self._ocr_name = f"paddleocr-mobile-{self._ocr_device}"
            return self._ocr_name
        except Exception:
            pass
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._ocr_engine = pytesseract
            self._ocr_name = "tesseract"
            return self._ocr_name
        except Exception:
            pass
        self._ocr_name = "none"
        return self._ocr_name

    def _run_ocr(self, image_path: str) -> list[dict[str, Any]]:
        engine = self._detect_ocr_name()
        if engine.startswith("paddleocr"):
            return self._run_paddle_ocr(image_path)
        if engine == "tesseract":
            return self._run_tesseract_ocr(image_path)
        return [{"text": "No OCR engine installed. Install PaddleOCR or Tesseract.", "confidence": 0, "x": 0, "y": 0}]

    def _run_paddle_ocr(self, image_path: str) -> list[dict[str, Any]]:
        # 放大+增强对比度再OCR
        img = self._image_cls.open(image_path)
        img = img.resize((img.width * 2, img.height * 2), getattr(self._image_cls, "Resampling", self._image_cls).LANCZOS)
        try:
            from PIL import ImageEnhance
            img = ImageEnhance.Contrast(img).enhance(1.5)
            img = ImageEnhance.Sharpness(img).enhance(1.5)
        except Exception:
            pass
        enhanced_path = image_path + ".enhanced.png"
        img.save(enhanced_path)
        try:
            try:
                result = self._ocr_engine.ocr(enhanced_path, cls=True)
            except Exception:
                result = self._ocr_engine.ocr(enhanced_path)
        finally:
            try:
                os.unlink(enhanced_path)
            except OSError:
                pass
        if not result:
            return []

        texts: list[dict[str, Any]] = []
        if isinstance(result[0], dict):
            for page in result:
                rec_texts = page.get("rec_texts") or []
                rec_scores = page.get("rec_scores") or []
                rec_boxes = page.get("rec_polys") or page.get("dt_polys") or []
                for index, text in enumerate(rec_texts):
                    points = rec_boxes[index] if index < len(rec_boxes) else []
                    if hasattr(points, "tolist"):
                        points = points.tolist()
                    xs = [p[0] for p in points] if points else [0]
                    ys = [p[1] for p in points] if points else [0]
                    texts.append({
                        "text": str(text),
                        "confidence": float(rec_scores[index]) if index < len(rec_scores) else 0,
                        "x": int(min(xs) / 2),
                        "y": int(min(ys) / 2),
                    })
            return texts

        for line in result[0] or []:
            box, (text, confidence) = line[0], line[1]
            texts.append({
                "text": str(text),
                "confidence": float(confidence),
                "x": int(min(p[0] for p in box) / 2),
                "y": int(min(p[1] for p in box) / 2),
            })
        return texts

    def _run_tesseract_ocr(self, image_path: str) -> list[dict[str, Any]]:
        img = self._image_cls.open(image_path)
        # 放大2倍
        img = img.resize((img.width * 2, img.height * 2), getattr(self._image_cls, "Resampling", self._image_cls).LANCZOS)
        # 增强对比度
        try:
            from PIL import ImageEnhance
            img = ImageEnhance.Contrast(img).enhance(2.0)
            img = ImageEnhance.Sharpness(img).enhance(2.0)
        except Exception:
            pass
        # 保存处理后的图片给tesseract
        enhanced_path = image_path + ".enhanced.png"
        img.save(enhanced_path)
        try:
            text = self._ocr_engine.image_to_string(self._image_cls.open(enhanced_path), lang="chi_sim+eng")
        finally:
            try:
                os.unlink(enhanced_path)
            except OSError:
                pass
        return [
            {"text": line.strip(), "confidence": 0.5, "x": 0, "y": 0}
            for line in text.splitlines()
            if line.strip()
        ]


def normalize_key(key: str) -> str:
    key = str(key).lower().strip()
    aliases = {
        "return": "enter",
        "esc": "escape",
        "spacebar": "space",
        "cmd": "command",
        "win": "windows",
        "ctrl": "ctrl",
        "control": "ctrl",
        "option": "alt",
    }
    return aliases.get(key, key)


def text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


TOOLS = [
    {
        "name": "status",
        "description": "Check dependencies, selected input backend, OCR engine, and Sky window detection.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "focus_game",
        "description": "Bring the Sky PC window to the foreground.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "press_key",
        "description": "Press a keyboard key in Sky PC. Use WASD for movement, space for jump/fly, F for interaction, Q for honk, Tab for flight mode.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key name such as w, a, s, d, space, f, q, e, tab, enter, or shift-space."},
                "duration_ms": {"type": "integer", "description": "Hold duration in milliseconds.", "default": 80},
                "backend": {"type": "string", "description": "Input backend override: auto, pydirectinput, pyautogui, gamepad, or arduino.", "default": "auto"},
            },
            "required": ["key"],
        },
    },
    {
        "name": "open_chat",
        "description": "Short-tap the Sky chat key. Use this to test whether Enter opens chat; keep duration short so Enter does not become voice/chat-hold behavior.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Chat key, usually enter.", "default": "enter"},
                "duration_ms": {"type": "integer", "description": "Tap duration in milliseconds.", "default": 35},
                "backend": {"type": "string", "description": "Input backend override: auto, pydirectinput, pyautogui, gamepad, or arduino.", "default": "auto"},
            },
        },
    },
    {
        "name": "send_chat",
        "description": "Open Sky chat, paste a message via clipboard, and optionally send it. Supports Chinese and emoji.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Chat message under 240 characters."},
                "open_key": {"type": "string", "description": "Key used to open chat.", "default": "enter"},
                "open_delay_ms": {"type": "integer", "description": "Delay after opening chat before pasting.", "default": 180},
                "assume_open": {"type": "boolean", "description": "Set true when the chat input is already open; no focus click or open key is sent.", "default": False},
                "send": {"type": "boolean", "description": "Set false to paste without pressing Enter.", "default": True},
                "enter_tap_ms": {"type": "integer", "description": "Short Enter tap duration.", "default": 35},
                "backend": {"type": "string", "description": "Input backend override: auto, pydirectinput, pyautogui, gamepad, or arduino.", "default": "auto"},
            },
            "required": ["message"],
        },
    },
    {
        "name": "type_text",
        "description": "Paste text into an already-open Sky chat input and optionally send. Use this when the user manually opened the input box.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Chat message under 240 characters."},
                "send": {"type": "boolean", "description": "Set false to paste without pressing Enter.", "default": True},
                "enter_tap_ms": {"type": "integer", "description": "Short Enter tap duration.", "default": 35},
                "backend": {"type": "string", "description": "Input backend override: auto, pydirectinput, pyautogui, gamepad, or arduino.", "default": "auto"},
                "require_foreground": {"type": "boolean", "description": "Require Sky to already be the foreground window before pasting.", "default": True},
            },
            "required": ["message"],
        },
    },
    {
        "name": "read_screen",
        "description": "Take a screenshot of the Sky window and OCR visible text.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "take_screenshot",
        "description": "Take a screenshot of the Sky window and return it as a PNG image.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class McpServer:
    def __init__(self, controller: PcSkyController):
        self.controller = controller

    def handle_tool_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "status":
            return {"content": text_content(json.dumps(self.controller.status(), ensure_ascii=False, indent=2))}
        if name == "focus_game":
            return {"content": text_content(json.dumps(self.controller.focus_game(), ensure_ascii=False, indent=2))}
        if name == "press_key":
            result = self.controller.press_key(
                arguments["key"],
                arguments.get("duration_ms", 80),
                arguments.get("backend"),
            )
            return {"content": text_content(result)}
        if name == "open_chat":
            result = self.controller.open_chat(
                arguments.get("key", "enter"),
                arguments.get("duration_ms", 35),
                arguments.get("backend"),
            )
            return {"content": text_content(result)}
        if name == "send_chat":
            result = self.controller.send_chat(
                arguments["message"],
                arguments.get("open_key", "enter"),
                arguments.get("open_delay_ms", 180),
                arguments.get("assume_open", False),
                arguments.get("send", True),
                arguments.get("enter_tap_ms", 35),
                arguments.get("backend"),
            )
            return {"content": text_content(result)}
        if name == "type_text":
            result = self.controller.type_text(
                arguments["message"],
                arguments.get("send", True),
                arguments.get("enter_tap_ms", 35),
                arguments.get("backend"),
                arguments.get("require_foreground", True),
            )
            return {"content": text_content(result)}
        if name == "read_screen":
            result = self.controller.read_screen()
            try:
                shot = self.controller.screenshot_base64()
                return {"content": [
                    {"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)},
                    {"type": "image", "data": shot, "mimeType": "image/png"},
                ]}
            except Exception:
                return {"content": text_content(json.dumps(result, ensure_ascii=False, indent=2))}

        if name == "take_screenshot":
            data = self.controller.screenshot_base64()
            return {"content": [{"type": "image", "data": data, "mimeType": "image/png"}]}
        raise SkyError(f"Unknown tool: {name}")

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        msg_id = message.get("id")
        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": SERVER_INFO,
                    },
                }
            if method == "notifications/initialized":
                return None
            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
            if method == "tools/call":
                params = message.get("params") or {}
                result = self.handle_tool_call(params.get("name", ""), params.get("arguments") or {})
                result.setdefault("isError", False)
                return {"jsonrpc": "2.0", "id": msg_id, "result": result}
            if method == "ping":
                return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": text_content(f"Error: {exc}"), "isError": True},
            }


def run_stdio(server: McpServer) -> None:
    log("starting stdio transport")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = server.handle(message)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def require_authorized(handler: BaseHTTPRequestHandler, config: ServerConfig) -> bool:
    if not config.token:
        return True
    auth = handler.headers.get("Authorization", "")
    expected = f"Bearer {config.token}"
    if auth == expected:
        return True
    handler.send_response(401)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(json.dumps({"error": "missing or invalid bearer token"}).encode("utf-8"))
    return False


def run_http(server: McpServer, config: ServerConfig) -> None:
    if config.host not in LOCAL_HOSTS and not config.token:
        raise SystemExit("Refusing to bind a remote HTTP server without --token.")
    server.controller._detect_ocr_name()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            body = json.dumps({
                "status": "ok",
                "server": SERVER_INFO,
                "ocr": server.controller._detect_ocr_name(),
                "ocr_device": server.controller._ocr_device,
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if not require_authorized(self, config):
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                message = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return
            response = server.handle(message)
            if response is None:
                self.send_response(204)
                self.end_headers()
                return
            payload = json.dumps(response, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args):
            log(fmt % args)

    httpd = ThreadingHTTPServer((config.host, config.port), Handler)
    log(f"starting HTTP transport at http://{config.host}:{config.port}")
    if config.token:
        log("HTTP bearer token is enabled")
    httpd.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP server for controlling PC Sky.")
    parser.add_argument("--http", nargs="?", const="true", default=None, help="Run HTTP JSON-RPC transport. Optional legacy form: --http 9800.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host. Use 0.0.0.0 for LAN clients such as Polaris.")
    parser.add_argument("--port", type=int, default=9800, help="HTTP port.")
    parser.add_argument("--token", default=os.environ.get("SKY_MCP_TOKEN"), help="Bearer token for HTTP POST requests.")
    parser.add_argument("--print-token", action="store_true", help="Generate and print a random token, then exit.")
    parser.add_argument("--window-title", default=os.environ.get("SKY_WINDOW_TITLE"), help="Window title substring for Sky.")
    parser.add_argument("--monitor", type=int, default=int(os.environ.get("SKY_MONITOR", "1")), help="Monitor index fallback for screenshots.")
    parser.add_argument("--input-backend", choices=["auto", "pyautogui", "pydirectinput", "gamepad", "arduino"], default=os.environ.get("SKY_INPUT_BACKEND", "auto"))
    parser.add_argument("--screenshot-scale", type=float, default=float(os.environ.get("SKY_SCREENSHOT_SCALE", "1.0")))
    parser.add_argument("--screenshot-max-width", type=int, default=int(os.environ.get("SKY_SCREENSHOT_MAX_WIDTH", "1920")))
    parser.add_argument("--screenshot-max-height", type=int, default=int(os.environ.get("SKY_SCREENSHOT_MAX_HEIGHT", "1080")))
    parser.add_argument("--serial-port", default=os.environ.get("SKY_SERIAL_PORT"), help="COM port for Arduino keyboard, e.g. COM9.")
    args = parser.parse_args()
    if args.http and args.http != "true":
        args.port = int(args.http)
    args.http = bool(args.http)
    return args


def main() -> None:
    args = parse_args()
    if args.print_token:
        print(secrets.token_urlsafe(24))
        return
    config = ServerConfig(
        host=args.host,
        port=args.port,
        token=args.token,
        allow_unsafe_http=False,
        window_title=args.window_title,
        monitor=args.monitor,
        input_backend=args.input_backend,
        screenshot_scale=args.screenshot_scale,
        screenshot_max_width=args.screenshot_max_width,
        screenshot_max_height=args.screenshot_max_height,
        serial_port=args.serial_port,
    )
    controller = PcSkyController(config)
    server = McpServer(controller)
    if args.http:
        run_http(server, config)
    else:
        run_stdio(server)


if __name__ == "__main__":
    main()
