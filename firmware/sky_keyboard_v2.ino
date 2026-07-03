// sky keyboard firmware v2 (2026-07-03)
// 协议与 v1 兼容（PRESS/HOTKEY/PING），sky-mcp-server 无需改协议。
// v1 的三个隐患（2026-07-03 两次整机键鼠卡死的根源）：
//   1. 按住时长 ms 无上限 —— 串口乱码可拼出几分钟的时长，板子按住 Ctrl 死等
//   2. 无"松开所有键"指令 —— 卡住后除了断电没有解药
//   3. 接收缓冲无上限 —— 乱码无限堆积
// v2 修复：时长封顶 2 秒；新增 RELEASE 指令；开机先 releaseAll；缓冲限长。

#include <Keyboard.h>

String buf = "";
bool overflow = false;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }
  delay(1000);
  Keyboard.begin();
  Keyboard.releaseAll();          // 复位/重插后，确保没有键悬着
  Serial.println("READY v2");
}

void handleCmd(String cmd) {
  cmd.trim();

  if (cmd == "PING") {
    Serial.println("PONG");
    return;
  }

  if (cmd == "RELEASE") {         // 急救指令：松开一切
    Keyboard.releaseAll();
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("PRESS ")) {
    int sp = cmd.indexOf(' ', 6);
    if (sp < 0) { Serial.println("ERR"); return; }
    int kc = cmd.substring(6, sp).toInt();
    long ms = cmd.substring(sp + 1).toInt();
    if (kc <= 0) { Serial.println("ERR badkey"); return; }
    ms = constrain(ms, 1, 2000);  // 乱码也压不死键盘
    Keyboard.press(kc);
    delay(ms);
    Keyboard.releaseAll();        // release(kc) 升级为 releaseAll：兜住一切解析意外
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("HOTKEY ")) {
    String p = cmd.substring(7);
    int s1 = p.indexOf(' ');
    int s2 = p.indexOf(' ', s1 + 1);
    if (s1 < 0 || s2 < 0) { Serial.println("ERR"); return; }
    int k1 = p.substring(0, s1).toInt();
    int k2 = p.substring(s1 + 1, s2).toInt();
    long ms = p.substring(s2 + 1).toInt();
    if (k1 <= 0 || k2 <= 0) { Serial.println("ERR badkey"); return; }
    ms = constrain(ms, 1, 2000);
    Keyboard.press(k1);
    delay(10);
    Keyboard.press(k2);
    delay(ms);
    Keyboard.releaseAll();
    Serial.println("OK");
    return;
  }

  Serial.println("ERR unknown");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      if (overflow) {
        Serial.println("ERR overflow");  // 超长=乱码，整行丢弃
      } else {
        handleCmd(buf);
      }
      buf = "";
      overflow = false;
    } else if (buf.length() < 64) {
      buf += c;
    } else {
      overflow = true;
    }
  }
}
