"""DebateOrchestrator: agent-to-agent ElevenLabs Conversational AI debate.

Architecture:
    Two ElevenLabs ConvAI sessions (one per persona) are opened at the start and
    kept alive across all turns — no per-turn session setup cost.

    Turn-taking loop:
        1. Inject last speaker's text into next agent via send_user_message()
        2. Wait for callback_agent_response to fire (threading.Event)
        3. Stream audio live via AudioInterface.output() → sounddevice
        4. Repeat until budget exhausted

    Audio:
        - ElevenLabs ConvAI outputs PCM 16-bit mono at 16kHz
        - Streamed live to speakers via sounddevice.OutputStream
        - All chunks accumulated in memory; written to MP3 via pydub at end

    Fallback (--no-conversational-ai):
        Anthropic messages.create for text generation + elevenlabs.generate() for TTS.
        Still uses ElevenLabs platform — just separates text and audio synthesis.

    SDK version: elevenlabs>=2.0 (tested with 2.47.0)
    Key API: Conversation.send_user_message(text), AudioInterface.output(bytes),
             AudioInterface.interrupt(), callback_agent_response
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import sounddevice as sd
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# ElevenLabs ConvAI PCM format: 16-bit mono at 16kHz
_SAMPLE_RATE = 16_000
_SAMPLE_WIDTH = 2  # bytes (16-bit)


@dataclass
class DebateBudget:
    """Tracks debate turn budget."""

    max_turns: int = 6
    turns_used: int = 0

    @property
    def exhausted(self) -> bool:
        return self.turns_used >= self.max_turns


@dataclass
class TurnRecord:
    """Result of a single agent turn."""

    speaker: str  # "skeptic" or "optimist"
    text: str
    audio_bytes: bytes


class DebateAudioInterface:
    """Custom ElevenLabs AudioInterface for headless debate orchestration.

    Replaces the default microphone/speaker interface:
      - start(): does nothing — input comes via send_user_message(), not mic
      - output(): captures PCM chunks to buffer + real-time queue
      - interrupt(): drains the queue when ElevenLabs signals an interruption
      - stop(): no-op cleanup

    PCM format: 16-bit mono at 16kHz (ElevenLabs ConvAI default).
    """

    def __init__(self) -> None:
        self._audio_buffer: List[bytes] = []
        self._audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self._transcript: List[str] = []

    # --- AudioInterface ABC methods ---

    def start(self, input_callback: Callable[[bytes], None]) -> None:
        """Called once before the session starts. We don't use mic input."""
        pass  # Text is injected via send_user_message(), not mic audio

    def stop(self) -> None:
        """Called once after the session ends."""
        pass

    def output(self, audio: bytes) -> None:
        """Receive a PCM audio chunk from the agent. Must return quickly."""
        self._audio_buffer.append(audio)
        try:
            self._audio_queue.put_nowait(audio)
        except queue.Full:
            # Drop rather than block the ElevenLabs callback thread
            pass

    def interrupt(self) -> None:
        """ElevenLabs signals the agent was interrupted — drain buffered audio."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    # --- Orchestration helpers ---

    def reset_for_turn(self) -> None:
        """Clear per-turn state before injecting a new message."""
        self._audio_buffer = []
        self._transcript = []
        self.interrupt()  # drain queue

    def get_transcript(self) -> str:
        return " ".join(self._transcript)

    def drain_audio_chunks(self) -> bytes:
        """Return all accumulated audio as raw bytes and discard the queue.

        _audio_buffer and _audio_queue receive the same PCM chunks. We play
        from the returned bytes, so the queue must be emptied to prevent the
        caller from double-playing the same audio.
        """
        self.interrupt()  # discard queue contents (same data as _audio_buffer)
        return b"".join(self._audio_buffer)


class DebateOrchestrator:
    """Orchestrates a two-agent ElevenLabs voice debate.

    Primary path (default):
        Two Conversation sessions with persistent WebSocket connections.
        send_user_message() injects the previous speaker's text.
        threading.Event waits for callback_agent_response before proceeding.

    Fallback path (use_conversational_ai=False):
        Anthropic generates text, elevenlabs.generate() synthesizes audio.
        Activated via --no-conversational-ai CLI flag.

    Usage:
        orchestrator = DebateOrchestrator(
            elevenlabs_api_key=...,
            skeptic_agent_id=...,
            optimist_agent_id=...,
        )
        turns = orchestrator.run_debate(scaffold, asymmetric_context)
        orchestrator.save_mp3("debate_2026_05_16.mp3")
    """

    def __init__(
        self,
        elevenlabs_api_key: str,
        skeptic_agent_id: str,
        optimist_agent_id: str,
        budget: Optional[DebateBudget] = None,
        use_conversational_ai: bool = True,
    ) -> None:
        self.api_key = elevenlabs_api_key
        self.skeptic_agent_id = skeptic_agent_id
        self.optimist_agent_id = optimist_agent_id
        self.budget = budget or DebateBudget()
        self.use_conversational_ai = use_conversational_ai
        self._all_audio_chunks: List[bytes] = []

    def run_debate(
        self,
        scaffold: Dict[str, Any],
        context: "AsymmetricContext",  # noqa: F821 — imported at call site
    ) -> List[TurnRecord]:
        """Run the debate. Returns list of TurnRecord (speaker, text, audio).

        Raises:
            ValueError: if ElevenAgents fails and use_conversational_ai=True.
            RuntimeError: if required env vars are missing.
        """
        if self.use_conversational_ai:
            return self._run_elevenlabs_debate(scaffold, context)
        else:
            return self._run_tts_fallback(scaffold, context)

    # ------------------------------------------------------------------
    # Primary path: ElevenLabs Conversational AI
    # ------------------------------------------------------------------

    def _run_elevenlabs_debate(
        self,
        scaffold: Dict[str, Any],
        context: "AsymmetricContext",
    ) -> List[TurnRecord]:
        from elevenlabs.client import ElevenLabs
        from elevenlabs.conversational_ai.conversation import Conversation

        el = ElevenLabs(api_key=self.api_key)

        # Per-agent response synchronization
        skeptic_event = threading.Event()
        optimist_event = threading.Event()
        skeptic_responses: List[str] = []
        optimist_responses: List[str] = []

        skeptic_iface = DebateAudioInterface()
        optimist_iface = DebateAudioInterface()

        def on_skeptic_response(text: str) -> None:
            skeptic_responses.append(text)
            skeptic_event.set()

        def on_optimist_response(text: str) -> None:
            optimist_responses.append(text)
            optimist_event.set()

        # Override agent config so each ConvAI session uses our debate system prompt
        # and doesn't send an initial greeting (which would fire the callback before
        # we inject our first message, causing the event to fire on the wrong text).
        def _agent_override(system_prompt: str) -> Dict[str, Any]:
            return {
                "agent": {
                    "prompt": {"prompt": system_prompt},
                    "first_message": "",  # silence the greeting
                }
            }

        try:
            skeptic_conv = Conversation(
                el,
                agent_id=self.skeptic_agent_id,
                requires_auth=True,
                audio_interface=skeptic_iface,
                callback_agent_response=on_skeptic_response,
                override_agent_config=_agent_override(context.skeptic_system_prompt),
            )
            optimist_conv = Conversation(
                el,
                agent_id=self.optimist_agent_id,
                requires_auth=True,
                audio_interface=optimist_iface,
                callback_agent_response=on_optimist_response,
                override_agent_config=_agent_override(context.optimist_system_prompt),
            )
        except TypeError:
            # Older SDK versions don't support override_agent_config — fall back without it.
            # System prompts won't be injected; agents will use their dashboard configuration.
            logger.warning(
                "ElevenLabs SDK does not support override_agent_config — "
                "agents will use their dashboard system prompt. "
                "Upgrade elevenlabs to >=2.0 for full debate context injection."
            )
            skeptic_conv = Conversation(
                el,
                agent_id=self.skeptic_agent_id,
                requires_auth=True,
                audio_interface=skeptic_iface,
                callback_agent_response=on_skeptic_response,
            )
            optimist_conv = Conversation(
                el,
                agent_id=self.optimist_agent_id,
                requires_auth=True,
                audio_interface=optimist_iface,
                callback_agent_response=on_optimist_response,
            )

        turns: List[TurnRecord] = []
        first_topic = scaffold["topics"][0]
        last_text = first_topic.get("skeptic_angle", first_topic.get("claim", "Let's debate AI."))
        agents = [
            ("skeptic", skeptic_conv, skeptic_iface, skeptic_event, skeptic_responses),
            ("optimist", optimist_conv, optimist_iface, optimist_event, optimist_responses),
        ]
        speaker_idx = 0

        try:
            skeptic_conv.start_session()
            optimist_conv.start_session()

            # Wait for both WebSocket connections to establish
            _wait_for_ws(skeptic_conv, label="skeptic")
            _wait_for_ws(optimist_conv, label="optimist")

            # Flush any greeting callbacks that fired before our first send_user_message.
            # Even with first_message="", some SDK versions still fire an empty callback.
            time.sleep(1.0)
            skeptic_event.clear()
            optimist_event.clear()
            skeptic_responses.clear()
            optimist_responses.clear()
            skeptic_iface.reset_for_turn()
            optimist_iface.reset_for_turn()

            while not self.budget.exhausted:
                name, conv, iface, event, responses = agents[speaker_idx % 2]

                iface.reset_for_turn()
                event.clear()
                responses.clear()

                logger.info("[%s] Speaking on: %s", name, last_text[:80])
                conv.send_user_message(last_text)

                # Wait for the agent's text response (up to 30s)
                if not event.wait(timeout=30):
                    logger.warning("[%s] No response after 30s — ending debate", name)
                    break

                agent_text = responses[-1] if responses else ""
                if not agent_text.strip():
                    logger.warning("[%s] Empty response — ending debate early", name)
                    break

                # Wait for trailing PCM chunks to arrive after the transcript callback fires.
                time.sleep(0.5)
                audio_bytes = iface.drain_audio_chunks()  # also empties the queue
                if audio_bytes:
                    self._stream_audio_from_bytes(audio_bytes)
                    self._all_audio_chunks.append(audio_bytes)

                turns.append(TurnRecord(speaker=name, text=agent_text, audio_bytes=audio_bytes))
                self.budget.turns_used += 1
                last_text = agent_text
                speaker_idx += 1

        except Exception as e:
            raise ValueError(
                f"ElevenAgents session failed: {e}\n"
                "Use --no-conversational-ai for the TTS fallback path."
            ) from e
        finally:
            skeptic_conv.end_session()
            optimist_conv.end_session()
            skeptic_conv.wait_for_session_end()
            optimist_conv.wait_for_session_end()

        return turns

    # ------------------------------------------------------------------
    # Fallback path: OpenAI text + ElevenLabs TTS
    # ------------------------------------------------------------------

    def _run_tts_fallback(
        self,
        scaffold: Dict[str, Any],
        context: "AsymmetricContext",
    ) -> List[TurnRecord]:
        """Fallback: Anthropic generates turn text, ElevenLabs TTS synthesizes audio.

        Activated via --no-conversational-ai. Each turn is a stateless Anthropic call
        (system prompt = persona context, user message = previous speaker's text).
        TTS uses the voice_id from debate_personas.yaml, not the ConvAI agent ID.
        """
        from anthropic import Anthropic
        from elevenlabs.client import ElevenLabs
        from signalstack.agents.debate_context import load_persona

        el = ElevenLabs(api_key=self.api_key)
        anthropic_client = Anthropic()

        # Resolve voice IDs from persona config (not ConvAI agent IDs).
        # ELEVENLABS_SKEPTIC_VOICE_ID / ELEVENLABS_OPTIMIST_VOICE_ID override the yaml defaults.
        try:
            skeptic_voice = os.getenv(
                "ELEVENLABS_SKEPTIC_VOICE_ID",
                load_persona("skeptic").get("default_voice_id", "EXAVITQu4vr4xnSDxMaL"),
            )
            optimist_voice = os.getenv(
                "ELEVENLABS_OPTIMIST_VOICE_ID",
                load_persona("optimist").get("default_voice_id", "TX3LPaxmHKxFdv7VOQHJ"),
            )
        except Exception:
            skeptic_voice = "EXAVITQu4vr4xnSDxMaL"
            optimist_voice = "TX3LPaxmHKxFdv7VOQHJ"

        turns: List[TurnRecord] = []
        first_topic = scaffold["topics"][0]
        debate_title = scaffold.get("title", "AI Weekly Debate")
        claim = first_topic.get("claim", "the week's biggest AI story")

        personas = [
            ("skeptic", context.skeptic_system_prompt, skeptic_voice),
            ("optimist", context.optimist_system_prompt, optimist_voice),
        ]

        model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

        def _tts(text: str, voice_id: str) -> bytes:
            try:
                return b"".join(
                    el.text_to_speech.convert(
                        text=text,
                        voice_id=voice_id,
                        output_format="pcm_16000",
                    )
                )
            except Exception as e:
                logger.warning("ElevenLabs TTS failed: %s — skipping audio", e)
                return b""

        def _speak(speaker: str, text: str, voice_id: str) -> None:
            audio = _tts(text, voice_id)
            if audio:
                self._stream_audio_from_bytes(audio)
                self._all_audio_chunks.append(audio)
            turns.append(TurnRecord(speaker=speaker, text=text, audio_bytes=audio))

        def _llm(system_prompt: str, user_content: str, max_tokens: int = 200) -> str:
            resp = anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            return resp.content[0].text.strip()

        # --- Intro narration (neutral, read by skeptic voice) ---
        intro_text = (
            f"Welcome to {debate_title}. "
            f"This week's question: {claim}. "
            "Two perspectives — a skeptic and an optimist — each with different evidence. Let's get into it."
        )
        _speak("intro", intro_text, skeptic_voice)

        # --- Main debate turns ---
        last_text = first_topic.get("skeptic_angle", claim)
        speaker_idx = 0

        while not self.budget.exhausted:
            name, system_prompt, voice_id = personas[speaker_idx % 2]
            other_name = personas[(speaker_idx + 1) % 2][0]

            prompt = last_text if speaker_idx == 0 else f"{other_name} said: {last_text}"

            try:
                text = _llm(system_prompt, prompt)
            except Exception as e:
                logger.warning("Anthropic call failed for %s: %s", name, e)
                break

            if not text:
                logger.warning("%s returned empty response — ending debate", name)
                break

            _speak(name, text, voice_id)
            self.budget.turns_used += 1
            last_text = text
            speaker_idx += 1

        # --- Closing statements (one per persona, not counted against budget) ---
        closing_prompt = (
            "The debate is wrapping up. Give your closing statement in 2 sentences: "
            "your key takeaway and what you think this means going forward."
        )
        for name, system_prompt, voice_id in personas:
            try:
                closing = _llm(system_prompt, closing_prompt, max_tokens=150)
                if closing:
                    _speak(name, closing, voice_id)
            except Exception as e:
                logger.warning("Closing statement failed for %s: %s", name, e)

        return turns

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def save_mp3(self, output_path: str) -> Optional[str]:
        """Write all accumulated PCM audio to a file.

        Tries MP3 first (requires ffmpeg). Falls back to WAV if ffmpeg is
        not installed — renames the output path extension to .wav automatically.

        Returns:
            The actual output path on success, None if no audio was captured.
        """
        if not self._all_audio_chunks:
            logger.warning("save_mp3: no audio captured — nothing to save")
            return None

        raw_pcm = b"".join(self._all_audio_chunks)
        segment = AudioSegment(
            data=raw_pcm,
            sample_width=_SAMPLE_WIDTH,
            frame_rate=_SAMPLE_RATE,
            channels=1,
        )
        try:
            segment.export(output_path, format="mp3")
            logger.info("Saved debate MP3 → %s (%d KB)", output_path, len(raw_pcm) // 1024)
            return output_path
        except FileNotFoundError:
            # ffmpeg not installed — fall back to WAV (no external dependency)
            wav_path = str(Path(output_path).with_suffix(".wav"))
            logger.warning("ffmpeg not found — saving as WAV instead: %s", wav_path)
            segment.export(wav_path, format="wav")
            logger.info("Saved debate WAV → %s (%d KB)", wav_path, len(raw_pcm) // 1024)
            return wav_path

    def _stream_audio(self, audio_queue: queue.Queue, buffered: bytes) -> None:
        """Stream audio to sounddevice in real time from the queue.

        First plays any chunks already buffered, then drains remaining from queue.
        """
        with sd.OutputStream(
            samplerate=_SAMPLE_RATE,
            channels=1,
            dtype="int16",
        ) as stream:
            # Play the already-buffered chunks
            if buffered:
                arr = np.frombuffer(buffered, dtype=np.int16)
                stream.write(arr)
            # Drain any remaining chunks from queue
            while True:
                try:
                    chunk = audio_queue.get(timeout=1.0)
                    arr = np.frombuffer(chunk, dtype=np.int16)
                    stream.write(arr)
                except queue.Empty:
                    break

    def _stream_audio_from_bytes(self, audio_bytes: bytes) -> None:
        """Stream complete PCM audio bytes to sounddevice (for TTS fallback)."""
        arr = np.frombuffer(audio_bytes, dtype=np.int16)
        sd.play(arr, samplerate=_SAMPLE_RATE)
        sd.wait()


# ------------------------------------------------------------------
# Text-only debate (no ElevenLabs required)
# ------------------------------------------------------------------


def run_text_debate(
    summaries: List[Dict],
    trace: Any = None,
    rounds: int = 3,
) -> Optional[str]:
    """Pure-Anthropic turn-taking debate. Returns a markdown transcript.

    No ElevenLabs or audio dependencies. Uses Haiku for both personas.
    The debate runs ``rounds`` full cycles (skeptic + optimist per cycle).
    """
    from anthropic import Anthropic
    from signalstack.agents.debate_context import (
        build_asymmetric_context,
        build_scaffold,
        load_persona,
    )

    try:
        client = Anthropic()
    except Exception as exc:
        logger.warning("Anthropic client unavailable for debate: %s", exc)
        return None

    try:
        skeptic_persona = load_persona("skeptic")
        optimist_persona = load_persona("optimist")
    except Exception as exc:
        logger.warning("Failed to load personas: %s — using defaults", exc)
        skeptic_persona = {"name": "The Skeptic", "description": "A skeptical analyst."}
        optimist_persona = {"name": "The Optimist", "description": "An optimistic analyst."}

    scaffold = build_scaffold(trace, summaries)
    context = build_asymmetric_context(trace, summaries, skeptic_persona, optimist_persona)

    first_topic = scaffold["topics"][0]
    opening = first_topic.get("skeptic_angle") or first_topic.get("claim", "What matters most in AI this week?")
    title = scaffold.get("title", "AI Weekly Debate")

    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    personas = [
        (skeptic_persona.get("name", "The Skeptic"), context.skeptic_system_prompt),
        (optimist_persona.get("name", "The Optimist"), context.optimist_system_prompt),
    ]

    transcript: List[tuple] = []
    last_text = opening

    for turn in range(rounds * 2):
        name, system_prompt = personas[turn % 2]
        other_name = personas[(turn + 1) % 2][0]
        prompt = last_text if turn == 0 else f"{other_name} said: {last_text}"

        try:
            resp = client.messages.create(
                model=model,
                max_tokens=250,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            last_text = resp.content[0].text.strip()
            transcript.append((name, last_text))
            logger.debug("Debate turn %d (%s): %d chars", turn + 1, name, len(last_text))
        except Exception as exc:
            logger.warning("Debate turn %d failed for %s: %s", turn + 1, name, exc)
            break

    if not transcript:
        return None

    lines = [f"## Debate: {title}", ""]
    for speaker, text in transcript:
        lines.append(f"**{speaker}:** {text}")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------


def _wait_for_ws(conv: Any, label: str = "", timeout: float = 10.0) -> None:
    """Block until Conversation._ws is set (WebSocket connected) or timeout.

    Raises:
        RuntimeError: if the connection isn't established within timeout seconds.
    """
    start = time.monotonic()
    while getattr(conv, "_ws", None) is None:
        if time.monotonic() - start > timeout:
            raise RuntimeError(
                f"ElevenLabs WebSocket ({label}) failed to connect within {timeout}s. "
                "Check ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID."
            )
        time.sleep(0.05)
