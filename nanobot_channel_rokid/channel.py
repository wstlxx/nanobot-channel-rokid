import asyncio
import json
import urllib.parse
import logging
from typing import Any

import websockets
from pydantic import Field

from nanobot.channels.base import BaseChannel
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Base

logger = logging.getLogger(__name__)

class RokidConfig(Base):
    """Rokid channel configuration."""
    enabled: bool = False
    ws_url: str = "wss://..."  # 替换为真实的 Rokid WS Gateway URL
    link_code: str = ""
    link_secret: str = ""
    allow_from: list[str] = Field(default_factory=list)
    streaming: bool = True     # 开启 nanobot 的流式输出支持

class RokidChannel(BaseChannel):
    name = "rokid"
    display_name = "Rokid Glasses"

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = RokidConfig(**config)
        super().__init__(config, bus)
        self.ws = None
        self._reconnect_base_delay = 1.0
        self._max_reconnect_delay = 30.0

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return RokidConfig().model_dump(by_alias=True)

    def _build_ws_url(self) -> str:
        url = urllib.parse.urlparse(self.config.ws_url)
        query = urllib.parse.parse_qs(url.query)
        query['linkCode'] = [self.config.link_code]
        query['linkSecret'] = [self.config.link_secret]
        new_query = urllib.parse.urlencode(query, doseq=True)
        return urllib.parse.urlunparse(
            (url.scheme, url.netloc, url.path, url.params, new_query, url.fragment)
        )

    async def start(self) -> None:
        """必须永久阻塞。管理 WebSocket 连接和重连机制。"""
        self._running = True
        ws_url = self._build_ws_url()
        reconnect_attempt = 0

        while self._running:
            try:
                logger.info(f"[rokid] Connecting to {ws_url.split('?')[0]} (attempt {reconnect_attempt + 1})")
                async with websockets.connect(ws_url) as ws:
                    self.ws = ws
                    reconnect_attempt = 0
                    logger.info("[rokid] Connected successfully.")
                    
                    # 发送连接状态帧 (与 openclaw ws-bridge-service 对齐)
                    await ws.send(json.dumps({"type": "status", "connected": True}))
                    
                    # 持续监听消息
                    async for message in ws:
                        await self._on_message(message)
                        
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"[rokid] Connection closed: {e.code} - {e.reason}")
            except Exception as e:
                logger.error(f"[rokid] WebSocket error: {e}")

            if not self._running:
                break

            # 指数退避重连
            delay = min(self._reconnect_base_delay * (2 ** reconnect_attempt), self._max_reconnect_delay)
            reconnect_attempt += 1
            logger.info(f"[rokid] Reconnecting in {delay} seconds...")
            await asyncio.sleep(delay)

    async def stop(self) -> None:
        """干净地关闭"""
        self._running = False
        if self.ws and not self.ws.closed:
            await self.ws.close(code=1000, reason="Plugin stopping")

    async def _on_message(self, raw_message: str) -> None:
        """解析来自 Rokid 的消息并路由给 Nanobot Agent"""
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.warning(f"[rokid] Invalid JSON received: {raw_message[:100]}")
            return

        if data.get("type") == "cancel":
            # 暂不支持硬取消，nanobot 的 pipeline 会继续处理
            return

        # 兼容两种字段名 (messages/requestId 或 message/message_id)
        messages = data.get("messages") or data.get("message", [])
        request_id = data.get("requestId") or data.get("message_id")

        if not messages or not request_id:
            logger.warning(f"[rokid] Unrecognized message format: {raw_message[:100]}")
            return

        text_parts = []
        media_urls = []

        for m in messages:
            if m.get("type") == "text" and m.get("text"):
                text_parts.append(m["text"])
            elif m.get("type") == "image" and m.get("image_url"):
                media_urls.append(m["image_url"])

        combined_text = "\n".join(text_parts)
        
        # 将消息发布到 Agent 总线
        # 我们使用 request_id 作为 chat_id，确保回复能对应回这条请求
        await self._handle_message(
            sender_id=self.config.link_code, 
            chat_id=request_id,
            content=combined_text,
            media=media_urls  # 注意：如果 nanobot agent 无法直接处理 URL media，你可能需要在这里下载为本地路径
        )

    # ==========================================
    # 下行消息处理 (Agent -> Rokid)
    # ==========================================

    async def _send_ws_frame(self, frame: dict) -> None:
        if self.ws and not self.ws.closed:
            await self.ws.send(json.dumps(frame))
        else:
            logger.warning("[rokid] Cannot send message, WebSocket is not connected.")

    async def send(self, msg: OutboundMessage) -> None:
        """一次性发送（如果未开启流式）"""
        # 如果截获到了设备工具调用格式（见下方说明），则发送 tool_call 帧
        if await self._handle_device_tool_intercept(msg):
            return

        await self._send_ws_frame({
            "event": "done",
            "data": {
                "role": "agent",
                "message_id": msg.chat_id,
                "agent_id": "nanobot",
                "answer_stream": msg.content,
                "is_finish": True,
                "type": "answer"
            }
        })

    async def send_delta(self, msg: OutboundMessage) -> None:
        """流式输出，逐块发送给设备，对应 sendStreamChunk 和 sendDone"""
        if await self._handle_device_tool_intercept(msg):
            return

        delta = getattr(msg, "delta", "")
        stream_end = getattr(msg, "_stream_end", False)

        if delta:
            await self._send_ws_frame({
                "event": "message",
                "data": {
                    "role": "agent",
                    "message_id": msg.chat_id,
                    "agent_id": "nanobot",
                    "answer_stream": delta,
                    "is_finish": False,
                    "type": "answer"
                }
            })

        if stream_end:
            await self._send_ws_frame({
                "event": "done",
                "data": {
                    "role": "agent",
                    "message_id": msg.chat_id,
                    "agent_id": "nanobot",
                    "answer_stream": "",
                    "is_finish": True,
                    "type": "answer"
                }
            })

    async def _handle_device_tool_intercept(self, msg: OutboundMessage) -> bool:
        """
        拦截设备指令：
        Nanobot 和 Openclaw 处理工具的逻辑不同。Openclaw 在网关层注册了工具并截获。
        在 Nanobot 中，你可以让 Agent 遇到需要调用设备功能时，输出特定的 JSON 或 XML 格式（例如 {"command": "take_photo"}）
        这个方法检测到这类特殊文本后，将其转换为设备期望的 tool_call 帧并吞掉该文本。
        """
        if not msg.content.strip():
            return False
            
        try:
            # 这是一个简单的启发式检测：如果输出是纯 JSON 且包含 command 字段
            if msg.content.strip().startswith("{") and msg.content.strip().endswith("}"):
                data = json.loads(msg.content)
                if "command" in data:
                    await self._send_ws_frame({
                        "event": "done",
                        "data": {
                            "role": "agent",
                            "message_id": msg.chat_id,
                            "agent_id": "nanobot",
                            "is_finish": True,
                            "type": "tool_call",
                            "tool_call": data
                        }
                    })
                    return True
        except json.JSONDecodeError:
            pass
            
        return False