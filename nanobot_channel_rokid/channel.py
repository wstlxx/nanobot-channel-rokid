import asyncio
import json
import urllib.parse
import logging
from typing import Any

import websockets
import websockets.exceptions
from pydantic import Field

from nanobot.channels.base import BaseChannel
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Base

logger = logging.getLogger(__name__)

class RokidConfig(Base):
    """Rokid channel configuration."""
    enabled: bool = False
    ws_url: str = "wss://rcs.rokid.com/claw/ws/link"
    link_code: str = ""
    link_secret: str = ""
    allow_from: list[str] = Field(default_factory=list)
    streaming: bool = True

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
                    
                    # 发送连接状态帧
                    await self._send_ws_frame({"type": "status", "connected": True})
                    
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
        if self.ws:
            try:
                await self.ws.close(code=1000, reason="Plugin stopping")
            except Exception:
                pass

    async def _on_message(self, raw_message: str) -> None:
        """解析来自 Rokid 的消息并路由给 Nanobot Agent"""
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.warning(f"[rokid] Invalid JSON received: {raw_message[:100]}")
            return

        if data.get("type") == "cancel":
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
        
        await self._handle_message(
            sender_id=self.config.link_code, 
            chat_id=request_id,
            content=combined_text,
            media=media_urls 
        )

    async def _send_ws_frame(self, frame: dict) -> None:
        """安全地发送 WebSocket 帧"""
        if not self.ws:
            logger.warning("[rokid] Cannot send message, WebSocket is not initialized.")
            return
            
        try:
            await self.ws.send(json.dumps(frame))
        except websockets.exceptions.ConnectionClosed:
            logger.warning("[rokid] Cannot send message, WebSocket connection is closed.")
        except Exception as e:
            logger.error(f"[rokid] Error sending message: {e}")

    async def send(self, msg: OutboundMessage) -> None:
        """一次性发送（如果未开启流式）"""
        if await self._handle_device_tool_intercept(msg.chat_id, msg.content):
            return

        # 检查是否还有后续消息（比如调用工具后）
        is_resuming = msg.metadata.get("_resuming", False)

        await self._send_ws_frame({
            "event": "done" if not is_resuming else "message",
            "data": {
                "role": "agent",
                "message_id": msg.chat_id,
                "agent_id": "nanobot",
                "answer_stream": msg.content,
                "is_finish": not is_resuming,  # 如果还要继续，就告诉 Rokid 不要结束
                "type": "answer"
            }
        })

    async def send_delta(self, chat_id: str, content: str, metadata: dict) -> None:
        """流式输出，逐块发送给设备"""
        if await self._handle_device_tool_intercept(chat_id, content):
            return

        stream_end = metadata.get("_stream_end", False)
        # 检查多步逻辑中是否还会继续
        is_resuming = metadata.get("_resuming", False)

        if content:
            await self._send_ws_frame({
                "event": "message",
                "data": {
                    "role": "agent",
                    "message_id": chat_id,
                    "agent_id": "nanobot",
                    "answer_stream": content,
                    "is_finish": False,
                    "type": "answer"
                }
            })

        # 只有在流结束，且明确没有后续工具调用的情况下，才正式通知 Rokid 关断通道
        if stream_end and not is_resuming:
            await self._send_ws_frame({
                "event": "done",
                "data": {
                    "role": "agent",
                    "message_id": chat_id,
                    "agent_id": "nanobot",
                    "answer_stream": "",
                    "is_finish": True,
                    "type": "answer"
                }
            })

    async def _handle_device_tool_intercept(self, chat_id: str, content: str) -> bool:
        """拦截设备指令（如拍照、导航等 JSON 命令）"""
        if not content or not content.strip():
            return False
            
        try:
            text = content.strip()
            if text.startswith("{") and text.endswith("}"):
                data = json.loads(text)
                if "command" in data:
                    await self._send_ws_frame({
                        "event": "done",
                        "data": {
                            "role": "agent",
                            "message_id": chat_id,
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