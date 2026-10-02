"""Stage 4 — minimal Tkinter viewer.

Window layout:
    +------------------------------------------+
    |                                          |
    |        rendered current slide image      |
    |        (with animated highlight box)     |
    |                                          |
    +------------------------------------------+
    | active: s6_e2 Matcher    score 0.86      |
    +------------------------------------------+
    | > spoken phrase here_______________      |
    +------------------------------------------+

Keys:
    Return        — submit the phrase in the entry
    PageDown / →  — next slide
    PageUp   / ←  — previous slide
    Escape        — clear the active highlight
    Ctrl+Q        — quit

GUI cannot be auto-verified in headless environments. Run interactively:
    python3 -m src.viewer --index out/index.json --slides out/slides
"""

from __future__ import annotations

import argparse
import json
import queue
import time
from pathlib import Path
from typing import Optional

from .encoder import load_default_encoder
from .geom import Bbox, FitTransform, ease_in_out_cubic, inches_to_image_px, lerp_bbox
from .matcher import Index, Matcher, PolicyConfig


# ---- highlight animation parameters ----
ANIM_MS = 100      # highlight glide duration (snappier; was 180)
ANIM_FRAMES = 10
# two-tone box: a dark edge under a bright amber line, so the highlight reads on
# both light and dark slide themes (a dark-only box vanished on dark decks)
HIGHLIGHT_OUTER_COLOUR = "#1f1f1f"
HIGHLIGHT_OUTER_WIDTH = 6
HIGHLIGHT_INNER_COLOUR = "#ffb000"
HIGHLIGHT_INNER_WIDTH = 3
HIGHLIGHT_PAD = 4  # px outside the element so the box frames the text instead of covering it


class ViewerApp:
    def __init__(
        self,
        index_path: Path,
        slides_dir: Path,
        threshold: float = PolicyConfig.threshold,
        gap: float = PolicyConfig.gap,
        stickiness: int = PolicyConfig.stickiness,
        start_slide: int = 1,
        mic: bool = False,
        wav: Optional[Path] = None,
        grammar_bias: bool = False,
        match_throttle_ms: int = 90,
        device: int | str | None = None,
    ) -> None:
        import tkinter as tk
        from PIL import Image, ImageTk

        self._tk = tk
        self._Image = Image
        self._ImageTk = ImageTk

        self.slides_dir = slides_dir
        self._index_path = index_path
        self._mic = mic
        self._wav = wav
        self._device = device
        self._grammar_bias = grammar_bias
        self._match_throttle_ms = match_throttle_ms
        self._last_match_at = 0.0
        self._stt_queue: "queue.Queue[str]" = queue.Queue()
        self._transcriber = None  # set in _start_audio
        self.raw_index = json.loads(index_path.read_text())
        self.slide_w_in = self.raw_index["slide_size"]["w"]
        self.slide_h_in = self.raw_index["slide_size"]["h"]

        # encoder + matcher (heavy — load before window so first phrase is fast)
        self.encoder = load_default_encoder()
        print(f"loaded encoder ({self.encoder.model_name})")
        self.index = Index(index_path, self.encoder)
        self.matcher = Matcher(self.index, self.encoder)
        # The headless decision core (policy + buffer + deck alias normalizer),
        # shared with the PNG/video simulators so all three ground identically.
        from .live_controller import LiveController
        self._controller = LiveController(
            self.matcher, self.raw_index, slide_no=start_slide,
            policy_config=PolicyConfig(threshold=threshold, gap=gap, stickiness=stickiness),
        )
        self.policy = self._controller.policy  # alias: reset()/step()/active_id still work

        # element lookup: id → (slide_no, bbox_inches, text, type)
        self._element_by_id: dict[str, dict] = {}
        for slide in self.raw_index["slides"]:
            for el in slide["elements"]:
                bb = el["bbox"]
                lb = el.get("axis_label_bbox")
                self._element_by_id[el["id"]] = {
                    "slide": slide["slide_number"],
                    "bbox_in": (bb["x"], bb["y"], bb["w"], bb["h"]),
                    "axis_label_in": (lb["x"], lb["y"], lb["w"], lb["h"]) if lb else None,
                    "text": el.get("text"),
                    "type": el["type"],
                }

        self.slide_no = start_slide
        self._image_tk = None             # current PhotoImage (kept alive)
        self._image_pil = None
        self._current_canvas_bbox: Optional[Bbox] = None  # rendered highlight bbox
        self._target_canvas_bbox: Optional[Bbox] = None
        self._active_bbox_in: Optional[tuple] = None       # last targeted bbox (line or union)
        self._active_element_id: Optional[str] = None      # to detect element switches
        self._anim_start_bbox: Optional[Bbox] = None
        self._anim_step = 0
        self._anim_job = None
        self._fit: FitTransform = FitTransform(scale=1.0, offset_x=0, offset_y=0, draw_w=0, draw_h=0)
        self._image_id: Optional[int] = None
        self._highlight_outer_id: Optional[int] = None
        self._axis_label_id: Optional[int] = None        # companion box on the axis label
        self._resize_job = None                          # pending settle-redraw
        self._last_draw_size: Optional[tuple[int, int]] = None  # last drawn canvas size

        # ---- build UI ----
        self.root = tk.Tk()
        self.root.title("Smart Pointer")
        self.root.geometry("1280x800")
        self.root.configure(bg="#101010")

        self.canvas = tk.Canvas(self.root, bg="#101010", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="slide 1 — type a phrase below")
        status = tk.Label(self.root, textvariable=self.status_var,
                          bg="#1f1f1f", fg="#dddddd",
                          font=("Helvetica", 11), anchor="w", padx=10, pady=4)
        status.pack(fill="x", side="bottom")

        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(self.root, textvariable=self.entry_var,
                              font=("Helvetica", 13), bg="#202020", fg="#ffffff",
                              insertbackground="#ffffff", relief="flat")
        self.entry.pack(fill="x", side="bottom", ipady=8)

        # ---- bindings ----
        self.entry.bind("<Return>", self._on_speak)
        self.root.bind("<Next>", lambda _e: self._change_slide(+1))
        self.root.bind("<Prior>", lambda _e: self._change_slide(-1))
        self.root.bind("<Right>", lambda _e: self._change_slide(+1))
        self.root.bind("<Left>", lambda _e: self._change_slide(-1))
        self.root.bind("<Escape>", lambda _e: self._clear_highlight())
        self.root.bind("<Control-q>", lambda _e: self.root.destroy())
        self.canvas.bind("<Configure>", self._on_configure)

        self._load_slide_image()
        self.entry.focus_set()

        if self._mic or self._wav is not None:
            self._start_audio()
            # poll the STT queue 20x/sec
            self.root.after(50, self._drain_stt_queue)

    # ---------- slide handling ----------
    def _load_slide_image(self) -> None:
        path = self.slides_dir / f"slide_{self.slide_no:03d}.png"
        if not path.exists():
            self.status_var.set(f"missing image: {path}")
            self._image_pil = None
            self._image_tk = None
            return
        self._image_pil = self._Image.open(str(path))
        self._redraw()

    def _change_slide(self, delta: int) -> None:
        self._set_slide(self.slide_no + delta)

    def _set_slide(self, slide_no: int) -> None:
        if slide_no < 1 or slide_no > len(self.raw_index["slides"]):
            return
        self.slide_no = slide_no
        self.policy.reset()
        self._cancel_anim()
        self._current_canvas_bbox = None
        self._target_canvas_bbox = None
        self._load_slide_image()
        self._update_status()

    # ---------- typed phrase handling ----------
    def _on_speak(self, _evt=None) -> None:
        text = self.entry_var.get().strip()
        if not text:
            return
        self.entry_var.set("")
        res, active_id, why = self.policy.step(text, self.slide_no)
        line = f"phrase={text!r}  best={res.element_id or '-'} score={res.score:.3f} → {why}"
        if active_id:
            el = self._element_by_id.get(active_id, {})
            line += f"   active={active_id} {el.get('type','')} {(el.get('text') or '')[:40]!r}"
            self._set_target_from_element(active_id, line_bbox=self._controller.unit_bbox(active_id))
        else:
            line += "   active=none"
            self._set_target_canvas_bbox(None)
        self.status_var.set(line)

    # ---------- highlight machinery ----------
    def _set_target_from_element(self, element_id: str,
                                 line_bbox: dict | None = None) -> None:
        el = self._element_by_id[element_id]
        if not self._image_pil:
            return
        # Per-event rule, identical to the simulators: draw the focused wrapped
        # line when the controller picked one, else the union box. The controller
        # already HOLDS the line across momentary misses, so a None here means
        # "no line chosen for this element yet" (union), not "snap back". The
        # event-less resize redraw passes no line_bbox and reuses the current rect.
        if line_bbox is not None:
            bbox_in = (line_bbox["x"], line_bbox["y"], line_bbox["w"], line_bbox["h"])
        elif element_id == self._active_element_id and self._active_bbox_in is not None:
            bbox_in = self._active_bbox_in
        else:
            bbox_in = el["bbox_in"]
        self._active_bbox_in = bbox_in
        self._active_element_id = element_id
        target_img_px = inches_to_image_px(
            bbox_in, self.slide_w_in, self.slide_h_in,
            self._image_pil.width, self._image_pil.height,
        )
        target_canvas = self._fit.image_bbox_to_canvas(target_img_px)
        self._set_target_canvas_bbox(target_canvas)

    def _set_target_canvas_bbox(self, target: Optional[Bbox]) -> None:
        self._cancel_anim()
        if target is None:
            self._target_canvas_bbox = None
            self._current_canvas_bbox = None
            self._active_bbox_in = None
            self._active_element_id = None
            self._draw_highlight(None)
            return
        if self._current_canvas_bbox is None:
            # appearing from nowhere — pop in at the target without animating
            self._current_canvas_bbox = target
            self._target_canvas_bbox = target
            self._draw_highlight(target)
            return
        self._anim_start_bbox = self._current_canvas_bbox
        self._target_canvas_bbox = target
        self._anim_step = 0
        self._tick_anim()

    def _tick_anim(self) -> None:
        if self._anim_start_bbox is None or self._target_canvas_bbox is None:
            return
        self._anim_step += 1
        t = self._anim_step / ANIM_FRAMES
        eased = ease_in_out_cubic(t)
        self._current_canvas_bbox = lerp_bbox(self._anim_start_bbox, self._target_canvas_bbox, eased)
        self._draw_highlight(self._current_canvas_bbox)
        if self._anim_step >= ANIM_FRAMES:
            self._current_canvas_bbox = self._target_canvas_bbox
            self._anim_job = None
            return
        self._anim_job = self.root.after(int(ANIM_MS / ANIM_FRAMES), self._tick_anim)

    def _cancel_anim(self) -> None:
        if self._anim_job is not None:
            try:
                self.root.after_cancel(self._anim_job)
            except Exception:
                pass
            self._anim_job = None

    def _clear_highlight(self) -> None:
        self.policy.reset()
        self._set_target_canvas_bbox(None)
        self._update_status()

    # ---------- rendering ----------
    def _on_configure(self, _event) -> None:
        """Debounced resize handler.

        Spurious <Configure> events (same size) are dropped. While the user is
        actively dragging, redraw with the cheap BILINEAR filter for
        responsiveness; once resizing settles (~80 ms idle), do one crisp
        LANCZOS pass.
        """
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        if (cw, ch) == self._last_draw_size:
            return
        self._redraw(resample=self._Image.BILINEAR)
        if self._resize_job is not None:
            try:
                self.root.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.root.after(80, self._settle_redraw)

    def _settle_redraw(self) -> None:
        self._resize_job = None
        self._redraw(resample=self._Image.LANCZOS)

    def _redraw(self, resample=None) -> None:
        if self._image_pil is None:
            return
        if resample is None:
            resample = self._Image.LANCZOS
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        self._fit = FitTransform.fit(self._image_pil.width, self._image_pil.height, cw, ch)
        if self._fit.draw_w <= 0 or self._fit.draw_h <= 0:
            return
        self._last_draw_size = (cw, ch)
        resized = self._image_pil.resize(
            (max(int(self._fit.draw_w), 1), max(int(self._fit.draw_h), 1)),
            resample,
        )
        self._image_tk = self._ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self._image_id = self.canvas.create_image(
            self._fit.offset_x, self._fit.offset_y, anchor="nw", image=self._image_tk,
        )
        self._highlight_outer_id = None
        # if the active element is set, recompute its canvas bbox at the new scale
        if self.policy.active_id:
            self._set_target_from_element(self.policy.active_id)
            # and skip animation on resize
            self._current_canvas_bbox = self._target_canvas_bbox
            self._draw_highlight(self._current_canvas_bbox)

    def _draw_highlight(self, bbox: Optional[Bbox]) -> None:
        if self._highlight_outer_id is not None:
            self.canvas.delete(self._highlight_outer_id)
            self._highlight_outer_id = None
        self.canvas.delete("hl_inner")
        if self._axis_label_id is not None:
            self.canvas.delete(self._axis_label_id)
            self._axis_label_id = None
        if bbox is None:
            return
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return
        x, y, w, h = x - HIGHLIGHT_PAD, y - HIGHLIGHT_PAD, w + 2 * HIGHLIGHT_PAD, h + 2 * HIGHLIGHT_PAD
        self._highlight_outer_id = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            outline=HIGHLIGHT_OUTER_COLOUR, width=HIGHLIGHT_OUTER_WIDTH,
        )
        self.canvas.create_rectangle(
            x, y, x + w, y + h, tags="hl_inner",
            outline=HIGHLIGHT_INNER_COLOUR, width=HIGHLIGHT_INNER_WIDTH,
        )
        # companion box on the matching x-axis label (chart points), so the data
        # point and its axis value read as related — static, same colour.
        self._draw_axis_label()

    def _draw_axis_label(self) -> None:
        active = self.policy.active_id
        el = self._element_by_id.get(active) if active else None
        label_in = el.get("axis_label_in") if el else None
        if not label_in or not self._image_pil:
            return
        img_px = inches_to_image_px(
            label_in, self.slide_w_in, self.slide_h_in,
            self._image_pil.width, self._image_pil.height,
        )
        x, y, w, h = self._fit.image_bbox_to_canvas(img_px)
        if w <= 0 or h <= 0:
            return
        self._axis_label_id = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            outline=HIGHLIGHT_INNER_COLOUR, width=HIGHLIGHT_INNER_WIDTH,
        )

    def _update_status(self) -> None:
        total = len(self.raw_index["slides"])
        line = f"slide {self.slide_no}/{total}"
        if self.policy.active_id:
            el = self._element_by_id.get(self.policy.active_id, {})
            line += f"   active={self.policy.active_id} {(el.get('text') or '')[:40]!r}"
        else:
            line += "   active=none"
        self.status_var.set(line)

    # ---------- audio + STT ----------
    def _ensure_engine(self, sample_rate: int = 16000):
        from .stt import load_default_stt_engine, hotwords_from_index
        if getattr(self, "_engine", None) is not None:
            return self._engine
        print("loading STT model…")
        self._engine = load_default_stt_engine(sample_rate=sample_rate, grammar=self._grammar_bias)
        self._hotwords = hotwords_from_index(self.raw_index)
        return self._engine

    def _start_audio(self) -> None:
        self._audio_restarted = False
        self._build_and_start_transcriber()

    def _build_and_start_transcriber(self) -> None:
        from .audio import FileSource, MicSource
        from .live import LiveTranscriber

        if self._wav is not None:
            source = FileSource(str(self._wav))
        else:
            source = MicSource(sample_rate=16000, device=self._device)
        engine = self._ensure_engine(source.sample_rate)
        # the controller owns the rolling buffer the audio thread feeds
        buffer = self._controller.buffer
        self._buffer = buffer
        self._transcriber = LiveTranscriber(source, engine, buffer, hotwords=getattr(self, "_hotwords", None))
        self._transcriber.start_in_thread(self._on_stt_update, on_error=self._on_audio_error)

    def _on_audio_error(self, exc: Exception) -> None:
        """Surface a dead STT/audio thread on the status line (was a silent
        freeze) and attempt ONE mic restart. Runs on the worker thread, so it
        marshals back to the Tk main loop via `after`."""
        def handle():
            msg = f"{type(exc).__name__}: {exc}"
            # only mic can meaningfully restart; replay/file ending is terminal
            if self._wav is None and not getattr(self, "_audio_restarted", False):
                self._audio_restarted = True
                self.status_var.set(f"audio error ({msg}) — restarting mic…")
                old = getattr(self, "_transcriber", None)
                if old is not None:
                    try:
                        old.stop()
                    except Exception:
                        pass
                try:
                    self._build_and_start_transcriber()
                except Exception as e2:
                    self.status_var.set(f"mic restart failed: {e2}")
            else:
                self.status_var.set(f"audio stopped: {msg}")
        try:
            self.root.after(0, handle)
        except Exception:
            pass

    def start_playlist(self, items: list[dict], inter_slide_pause_s: float = 0.6) -> None:
        """Walk a list of {'slide': N, 'wav': path} entries; play each WAV
        out the speakers while feeding STT in lockstep. Switches slides
        between entries. Called by `src/simulate.py`.
        """
        import threading

        from .audio import PlaybackSource
        from .live import LiveTranscriber
        from .stt import RollingBuffer

        self._ensure_engine()
        self._playlist_items = list(items)
        self._inter_slide_pause_s = inter_slide_pause_s

        def worker():
            for item in self._playlist_items:
                if not self.root.winfo_exists():
                    return
                self.root.after(0, lambda sn=item["slide"]: self._set_slide(sn))
                time.sleep(0.3)  # let the slide render
                source = PlaybackSource(str(item["wav"]))
                # fresh buffer per slide; point the shared controller at it so
                # _run_match (which reads controller.buffer) tracks this slide
                buffer = RollingBuffer(window_seconds=4.0)
                self._controller.buffer = buffer
                self._buffer = buffer
                transcriber = LiveTranscriber(source, self._engine, buffer,
                                              hotwords=getattr(self, "_hotwords", None))
                self._transcriber = transcriber
                transcriber.run(on_update=self._on_stt_update)
                # let the highlight settle, then move on
                time.sleep(self._inter_slide_pause_s)
            self.root.after(0, lambda: self.status_var.set("simulation complete — Esc/Ctrl-Q"))

        t = threading.Thread(target=worker, daemon=True, name="PlaylistWorker")
        t.start()
        self.root.after(50, self._drain_stt_queue)

    def _on_stt_update(self, buffer_text: str, partial) -> None:
        # called from the audio thread → just hand off to the Tk thread
        self._stt_queue.put(buffer_text)

    def _drain_stt_queue(self) -> None:
        latest: Optional[str] = None
        try:
            while True:
                latest = self._stt_queue.get_nowait()
        except queue.Empty:
            pass
        if latest:
            now = time.monotonic() * 1000
            if now - self._last_match_at >= self._match_throttle_ms:
                self._last_match_at = now
                self._run_match(latest)
        self.root.after(50, self._drain_stt_queue)

    def _run_match(self, text: str) -> None:
        # The controller windows the buffer tail and runs the shared policy; it
        # tracks the speaker's *current* phrase, not the whole growing utterance.
        self._controller.slide_no = self.slide_no
        ev = self._controller.feed(text)
        if ev is None:
            return
        line = f"heard={ev.tail!r}  best={ev.element_id or '-'} score={ev.score:.3f} → {ev.why}"
        if ev.active_id:
            el = self._element_by_id.get(ev.active_id, {})
            tag = f" L{ev.line_index}" if ev.line_index is not None else ""
            line += f"   active={ev.active_id}{tag} {(el.get('text') or '')[:40]!r}"
            self._set_target_from_element(ev.active_id, line_bbox=ev.bbox)
        else:
            line += "   active=none"
            self._set_target_canvas_bbox(None)
        self.status_var.set(line)

    # ---------- entry point ----------
    def run(self) -> None:
        self._update_status()
        try:
            self.root.mainloop()
        finally:
            if self._transcriber is not None:
                self._transcriber.stop()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Smart Pointer — Stage 4 viewer (Tkinter)")
    ap.add_argument("--index", default=None,
                    help="index json (default: best available — scoped > refined > base)")
    ap.add_argument("--slides", default="out/slides")
    ap.add_argument("--slide", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=PolicyConfig.threshold)
    ap.add_argument("--gap", type=float, default=PolicyConfig.gap)
    ap.add_argument("--stick", type=int, default=PolicyConfig.stickiness)
    ap.add_argument("--mic", action="store_true", help="capture live audio from the default mic via STT")
    ap.add_argument("--wav", type=Path, default=None, help="replay a 16-bit mono WAV through STT")
    ap.add_argument("--grammar-bias", action="store_true",
                    help="constrain the recognizer to deck vocabulary only (Vosk fallback engine; use when speaker stays strictly on-topic)")
    ap.add_argument("--device", default=None,
                    help="mic input device (index or name substring); default system input")
    ap.add_argument("--list-devices", action="store_true",
                    help="list available audio input devices and exit")
    args = ap.parse_args(argv)

    if args.list_devices:
        from .audio import list_input_devices
        devs = list_input_devices()
        if not devs:
            print("no audio input devices found")
        for d in devs:
            print(f"  [{d['index']}] {d['name']}  "
                  f"({d['channels']} ch, {d['default_samplerate']} Hz)")
        return 0

    # --device accepts an integer index or a name substring
    device: int | str | None = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    from .matcher import best_index
    index_path = best_index(args.index)
    slides_dir = Path(args.slides)
    if not index_path.exists():
        print(f"index not found: {index_path}")
        return 2
    if not slides_dir.exists():
        print(f"slides dir not found: {slides_dir}; run `python3 -m src.render <deck>` first")
        return 2

    app = ViewerApp(
        index_path=index_path,
        slides_dir=slides_dir,
        threshold=args.threshold, gap=args.gap, stickiness=args.stick,
        start_slide=args.slide,
        mic=args.mic, wav=args.wav, grammar_bias=args.grammar_bias,
        device=device,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
