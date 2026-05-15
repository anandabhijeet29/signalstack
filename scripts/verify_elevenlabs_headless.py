"""Verify ElevenLabs Conversational AI headless text injection.

This script proves the core architecture works before building production code:
  1. Creates a Conversation session with a custom AudioInterface
  2. Injects text via send_user_message() (no microphone)
  3. Captures agent response text via callback_agent_response
  4. Captures PCM audio via AudioInterface.output()

Run this BEFORE running signalstack debate to confirm your API key and agent IDs work.

Usage:
    ELEVENLABS_API_KEY=your-key ELEVENLABS_AGENT_ID=your-agent-id \\
        python scripts/verify_elevenlabs_headless.py

Expected output:
    Session started. Waiting for WebSocket...
    WebSocket connected. Injecting text...
    Agent responded: <agent's text response>
    Audio captured: <N> bytes
    SUCCESS — headless text injection works.

If you see "RuntimeError: Session not started or websocket not connected":
    The WebSocket took longer than expected to connect. Increase WAIT_TIMEOUT.
"""

import os
import queue
import sys
import time
import threading
from typing import Callable, List, Optional

# ---------------------------------------------------------------------------
# Minimal AudioInterface implementation (no dependencies beyond elevenlabs)
# ---------------------------------------------------------------------------

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs.conversational_ai.conversation import AudioInterface, Conversation
except ImportError:
    print("Error: elevenlabs package not found. Run: pip install 'elevenlabs>=2.0'")
    sys.exit(1)


class HeadlessAudioInterface(AudioInterface):
    """Captures audio output from ElevenLabs. Ignores microphone input."""

    def __init__(self) -> None:
        self._audio_chunks: List[bytes] = []

    def start(self, input_callback: Callable[[bytes], None]) -> None:
        pass  # no mic input — text comes via send_user_message()

    def stop(self) -> None:
        pass

    def output(self, audio: bytes) -> None:
        self._audio_chunks.append(audio)

    def interrupt(self) -> None:
        self._audio_chunks.clear()

    def total_bytes(self) -> int:
        return sum(len(c) for c in self._audio_chunks)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

WAIT_TIMEOUT = 10.0   # seconds to wait for WebSocket to connect
RESPONSE_TIMEOUT = 30.0  # seconds to wait for agent response

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_AGENT_ID = os.environ.get("ELEVENLABS_AGENT_ID", "")

if not ELEVENLABS_API_KEY:
    print("Error: ELEVENLABS_API_KEY not set.")
    sys.exit(1)

if not ELEVENLABS_AGENT_ID:
    print("Error: ELEVENLABS_AGENT_ID not set.")
    sys.exit(1)


def main() -> None:
    el = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    iface = HeadlessAudioInterface()

    response_event = threading.Event()
    agent_responses: List[str] = []

    def on_response(text: str) -> None:
        agent_responses.append(text)
        print(f"Agent responded: {text}")
        response_event.set()

    conv = Conversation(
        el,
        agent_id=ELEVENLABS_AGENT_ID,
        requires_auth=True,
        audio_interface=iface,
        callback_agent_response=on_response,
    )

    print("Session starting...")
    conv.start_session()

    # Wait for WebSocket to connect
    print("Waiting for WebSocket connection...")
    start = time.monotonic()
    while getattr(conv, "_ws", None) is None:
        if time.monotonic() - start > WAIT_TIMEOUT:
            conv.end_session()
            print(f"FAIL — WebSocket did not connect within {WAIT_TIMEOUT}s.")
            sys.exit(1)
        time.sleep(0.05)

    print("WebSocket connected. Injecting text message...")
    TEST_MESSAGE = "Say exactly: headless injection verified."
    conv.send_user_message(TEST_MESSAGE)

    # Wait for agent response
    print(f"Waiting up to {RESPONSE_TIMEOUT}s for agent response...")
    if not response_event.wait(timeout=RESPONSE_TIMEOUT):
        conv.end_session()
        print(f"FAIL — No response within {RESPONSE_TIMEOUT}s.")
        sys.exit(1)

    # Wait a moment for audio chunks to arrive after text callback
    time.sleep(1.0)

    print(f"Audio captured: {iface.total_bytes()} bytes")

    conv.end_session()
    conv.wait_for_session_end()

    if agent_responses:
        print("\nSUCCESS — headless text injection works.")
        print(f"Agent ID: {ELEVENLABS_AGENT_ID}")
        print(f"Response: {agent_responses[-1]}")
        print(f"Audio: {iface.total_bytes()} bytes PCM 16-bit 16kHz")
    else:
        print("FAIL — Session ended without agent response.")
        sys.exit(1)


if __name__ == "__main__":
    main()
