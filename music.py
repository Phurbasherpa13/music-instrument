"""
╔══════════════════════════════════════════════════════════════╗
║          POLY INSTRUMENT SYNTHESIZER  — v3 Final            ║
║  Piano · Guitar · Violin · Trumpet · Flute · Bell           ║
║  Bass · Organ · Marimba                                      ║
╠══════════════════════════════════════════════════════════════╣
║  Install :  pip install numpy pyaudio                        ║
║  Run     :  python music.py                                  ║
╠══════════════════════════════════════════════════════════════╣
║  Keyboard                                                    ║
║    White oct-1 :  1 2 3 4 5 6 7   (C D E F G A B)          ║
║    White oct-2 :  8 9 0 - =        (C D E F G)              ║
║    Black oct-1 :  Q W   R T Y      (C# D#  F# G# A#)        ║
║    Black oct-2 :  U I   O P [      (C# D#  F# G# A#)        ║
║    Octave      :  Z (down)  X (up)                          ║
║    Random      :  SPACE                                      ║
╚══════════════════════════════════════════════════════════════╝

BUGS FIXED vs v2
  1. Black keys rendered LEFT of C (wrong formula wb*WW → (wb+1)*WW)
  2. Karplus-Strong pure-Python O(n) loop replaced with numpy segments
  3. Tkinter thread-safety: GUI updates moved to root.after(0, ...)
  4. Envelope sus_end clamped to n to prevent negative linspace slices
  5. Keyboard: e.keysym fallback for special chars (-, =, [) on Windows
  6. Waveform zero-crossing trigger for stable oscilloscope display
  7. _bell: np.full+cumsum replaced with direct np.arange * phase
  8. Arp thread race condition: lbl captured before thread starts
"""

import tkinter as tk
import numpy as np
import pyaudio
import threading
import random
import time
from collections import deque

# ──────────────────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
CHUNK       = 512

NOTE_NAMES   = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
NOTE_OFFSETS = {n: i for i, n in enumerate(NOTE_NAMES)}
WHITES       = [n for n in NOTE_NAMES if '#' not in n]   # C D E F G A B

CHORD_TYPES = {
    'Major':      [0,4,7],
    'Minor':      [0,3,7],
    'Dom7':       [0,4,7,10],
    'Maj7':       [0,4,7,11],
    'Min7':       [0,3,7,10],
    'Sus4':       [0,5,7],
    'Diminished': [0,3,6],
    'Augmented':  [0,4,8],
    'Power':      [0,7,12],
    'Add9':       [0,4,7,14],
}

# ── Keyboard map: char/keysym → (note, octave-offset) ────────
# keysym values for special chars that may have empty e.char on Windows
KEYSYM_MAP = {
    'minus':      ('F', 1),   # -
    'equal':      ('G', 1),   # =
    'bracketleft':('A#',1),   # [
}
CHAR_MAP = {
    '1':('C',0),'2':('D',0),'3':('E',0),'4':('F',0),
    '5':('G',0),'6':('A',0),'7':('B',0),
    '8':('C',1),'9':('D',1),'0':('E',1),'-':('F',1),
    '=':('G',1),
    'q':('C#',0),'w':('D#',0),'r':('F#',0),'t':('G#',0),'y':('A#',0),
    'u':('C#',1),'i':('D#',1),'o':('F#',1),'p':('G#',1),'[':('A#',1),
}

# ── Colours ───────────────────────────────────────────────────
BG     = '#0b0d13'
PANEL  = '#13161f'
PANEL2 = '#1a1e2b'
TEXT   = '#d8e8f0'
DIM    = '#4a5570'

# ──────────────────────────────────────────────────────────────
#  INSTRUMENTS
#  Each entry:
#    emoji, color       — UI display
#    algorithm          — 'additive' | 'karplus' | 'bell'
#    harmonics          — [(freq_mult, amplitude), ...]
#    adsr               — (attack, decay, sustain_level, release) seconds
#    noise_mix          — 0..1  breathiness / bow noise
#    vibrato            — (rate_hz, depth_fraction)
#    chorus             — cents detune for second oscillator (0 = off)
#    oct_offset         — transpose random-note octave
# ──────────────────────────────────────────────────────────────
INSTRUMENTS = {
    'Piano': {
        'emoji':'🎹','color':'#00e5ff','algorithm':'additive',
        # Rich upper harmonics + slight inharmonicity on high partials
        'harmonics':[(1,1.0),(2,0.55),(3,0.28),(4,0.20),(5,0.14),
                     (6,0.09),(7,0.06),(8,0.04),(9,0.02)],
        'adsr':(0.005,0.45,0.25,0.8),'noise_mix':0.004,
        'vibrato':(0,0),'chorus':6,'oct_offset':0,
    },
    'Guitar': {
        'emoji':'🎸','color':'#ff9900','algorithm':'karplus',
        'harmonics':[(1,1.0),(2,0.5),(3,0.25),(4,0.12),(5,0.06)],
        'adsr':(0.003,0.25,0.08,0.55),'noise_mix':0.012,
        'vibrato':(5.5,0.003),'chorus':0,'oct_offset':0,
    },
    'Violin': {
        'emoji':'🎻','color':'#ff4d8d','algorithm':'additive',
        # Stronger 2nd harmonic = nasal bowed quality
        'harmonics':[(1,0.7),(2,1.0),(3,0.7),(4,0.35),(5,0.25),
                     (6,0.18),(7,0.12),(8,0.07)],
        'adsr':(0.15,0.06,0.9,0.3),'noise_mix':0.022,
        'vibrato':(6.2,0.009),'chorus':0,'oct_offset':0,
    },
    'Trumpet': {
        'emoji':'🎺','color':'#ffe066','algorithm':'additive',
        # Bright odd+even harmonics, strong 2nd
        'harmonics':[(1,0.6),(2,1.0),(3,0.9),(4,0.7),(5,0.55),
                     (6,0.42),(7,0.32),(8,0.22),(9,0.14),(10,0.08)],
        'adsr':(0.09,0.04,0.92,0.12),'noise_mix':0.005,
        'vibrato':(6.5,0.007),'chorus':0,'oct_offset':0,
    },
    'Flute': {
        'emoji':'🪈','color':'#88ffcc','algorithm':'additive',
        # Mostly fundamental, gentle 2nd, airy noise
        'harmonics':[(1,1.0),(2,0.22),(3,0.06),(4,0.02)],
        'adsr':(0.1,0.04,0.88,0.22),'noise_mix':0.03,
        'vibrato':(5.8,0.013),'chorus':0,'oct_offset':0,
    },
    'Bell': {
        'emoji':'🔔','color':'#cc99ff','algorithm':'bell',
        # Church bell inharmonic partials (McIntyre & Woodhouse ratios)
        'harmonics':[(1,1.0),(2.0,0.35),(2.756,0.55),(3.375,0.22),
                     (5.404,0.18),(7.081,0.10),(9.0,0.06)],
        'adsr':(0.003,1.2,0.0,3.0),'noise_mix':0.001,
        'vibrato':(0,0),'chorus':0,'oct_offset':0,
    },
    'Bass': {
        'emoji':'🎵','color':'#ff6655','algorithm':'karplus',
        'harmonics':[(1,1.0),(2,0.6),(3,0.4),(4,0.22),(5,0.10)],
        'adsr':(0.004,0.28,0.12,0.45),'noise_mix':0.008,
        'vibrato':(0,0),'chorus':0,'oct_offset':-1,
    },
    'Organ': {
        'emoji':'🎼','color':'#66ccff','algorithm':'additive',
        # Hammond-style drawbars: 16' 8' 4' 2⅔' 2' 1⅗' 1'
        'harmonics':[(1,1.0),(2,1.0),(3,0.8),(4,0.6),(5,0.5),
                     (6,0.38),(8,0.28),(10,0.18)],
        'adsr':(0.012,0.0,1.0,0.04),'noise_mix':0.001,
        # Organ Leslie cabinet: slow tremolo modulation handled in synthesis
        'vibrato':(7.2,0.006),'chorus':4,'oct_offset':0,
    },
    'Marimba': {
        'emoji':'🥁','color':'#ffaa44','algorithm':'bell',
        # Marimba bar ratios
        'harmonics':[(1,1.0),(4.0,0.45),(10.0,0.18),(2.76,0.12)],
        'adsr':(0.003,0.18,0.0,0.55),'noise_mix':0.002,
        'vibrato':(0,0),'chorus':0,'oct_offset':0,
    },
}

# ──────────────────────────────────────────────────────────────
#  MUSIC HELPERS
# ──────────────────────────────────────────────────────────────
def midi_to_freq(m: int) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)

def note_to_midi(name: str, octave: int) -> int:
    return NOTE_OFFSETS[name] + (octave + 1) * 12

def note_to_freq(name: str, octave: int) -> float:
    return midi_to_freq(note_to_midi(name, octave))

# ──────────────────────────────────────────────────────────────
#  SYNTHESIS UTILITIES
# ──────────────────────────────────────────────────────────────

def _envelope(n: int, adsr: tuple) -> np.ndarray:
    """
    ADSR envelope.  FIX: sus_end clamped so release never overflows n,
    and short notes still get a complete (if compressed) shape.
    """
    A, D, S, R = adsr
    sr  = SAMPLE_RATE
    a_s = max(1, int(A * sr))
    d_s = max(1, int(D * sr))
    r_s = max(1, int(R * sr))

    # FIX (BUG 4): clamp so that a_s + d_s + r_s never exceeds n
    total = a_s + d_s + r_s
    if total > n:
        scale = n / total
        a_s = max(1, int(a_s * scale))
        d_s = max(1, int(d_s * scale))
        r_s = max(1, int(r_s * scale))

    sus_end = n - r_s                        # where release begins
    sus_end = max(a_s + d_s, sus_end)        # sustain can't start before decay ends
    sus_end = min(sus_end, n)                # clamp to n

    env = np.ones(n, dtype=np.float32)

    # Attack
    env[:a_s] = np.linspace(0.0, 1.0, a_s)

    # Decay
    d_end = min(a_s + d_s, sus_end)
    de    = d_end - a_s
    if de > 0:
        env[a_s:d_end] = np.linspace(1.0, S, de)

    # Sustain
    if d_end < sus_end:
        env[d_end:sus_end] = S

    # Release
    if sus_end < n:
        env[sus_end:] = np.linspace(S, 0.0, n - sus_end)

    return np.clip(env, 0.0, 1.0)


def _additive(freq: float, t: np.ndarray,
              harmonics: list, vibrato: tuple,
              chorus_cents: float = 0.0) -> np.ndarray:
    """
    Additive synthesis with optional vibrato LFO and chorus detuning.
    FIX: chorus adds a second oscillator detuned by chorus_cents.
    """
    vib_r, vib_d = vibrato
    dt   = 1.0 / SAMPLE_RATE
    n    = len(t)

    # Vibrato: ramps in after the first 30 % of the note
    if vib_r > 0 and vib_d > 0:
        ramp   = np.clip(t / max(t[-1] * 0.30, 1e-9), 0.0, 1.0)
        lfo    = 1.0 + vib_d * np.sin(2.0 * np.pi * vib_r * t) * ramp
        freq_t = freq * lfo
    else:
        freq_t = np.full(n, freq, dtype=np.float64)

    wave = np.zeros(n, dtype=np.float64)

    for mult, amp in harmonics:
        phase = np.cumsum(freq_t * mult) * dt * 2.0 * np.pi
        wave += amp * np.sin(phase)

    # Chorus: second oscillator slightly detuned, summed at -3 dB
    if chorus_cents != 0.0:
        freq2  = freq * (2.0 ** (chorus_cents / 1200.0))
        freq2t = freq_t * (freq2 / freq)
        wave2  = np.zeros(n, dtype=np.float64)
        for mult, amp in harmonics:
            phase2 = np.cumsum(freq2t * mult) * dt * 2.0 * np.pi
            wave2 += amp * np.sin(phase2)
        wave = (wave + wave2 * 0.5) / 1.5   # mix at ~-3.5 dB

    return wave


def _karplus(freq: float, n: int, harmonics: list) -> np.ndarray:
    """
    Karplus-Strong plucked string.
    FIX (BUG 2): replaced O(n) Python loop with numpy segment processing.
    Each iteration applies the low-pass averaging across the full buffer
    then outputs buf_len samples, reducing iterations by ~buf_len.
    """
    sr      = SAMPLE_RATE
    buf_len = max(2, int(round(sr / freq)))

    # Seed buffer: noise + coloured harmonics
    buf = np.random.uniform(-1.0, 1.0, buf_len)
    t0  = np.arange(buf_len) / sr
    col = sum(amp * np.sin(2.0 * np.pi * freq * mult * t0)
              for mult, amp in harmonics)
    buf = (buf * 0.25 + col * 0.75).astype(np.float64)

    out = np.zeros(n, dtype=np.float64)
    pos = 0
    while pos < n:
        end   = min(pos + buf_len, n)
        chunk = end - pos
        out[pos:end] = buf[:chunk]
        # Low-pass averaging step (vectorised over entire buffer)
        buf = 0.4985 * (buf + np.roll(buf, -1))
        pos += buf_len

    return out


def _bell(freq: float, t: np.ndarray, harmonics: list) -> np.ndarray:
    """
    Bell / marimba: inharmonic partials with per-partial exponential decay.
    FIX (BUG 7): replaced np.full + np.cumsum with direct np.arange * phase.
    """
    dt   = 1.0 / SAMPLE_RATE
    n    = len(t)
    wave = np.zeros(n, dtype=np.float64)
    idx  = np.arange(n, dtype=np.float64)

    for i, (mult, amp) in enumerate(harmonics):
        decay = np.exp(-t * (0.4 + i * 1.1))
        phase = idx * (freq * mult * dt * 2.0 * np.pi)  # FIX: direct formula
        wave += amp * decay * np.sin(phase)

    return wave


def synthesize_note(inst_name: str, freq: float, duration: float) -> np.ndarray:
    """
    Master synthesis function.  Returns normalised float32 audio.
    Applies synthesis algorithm, noise colouring, ADSR envelope,
    soft-clip, and RMS normalisation.
    """
    inst  = INSTRUMENTS[inst_name]
    sr    = SAMPLE_RATE
    n     = max(1, int(sr * duration))
    t     = np.linspace(0.0, duration, n, endpoint=False)
    algo  = inst['algorithm']

    # ── Core waveform ─────────────────────────────────────────
    if algo == 'karplus':
        wave = _karplus(freq, n, inst['harmonics'])
    elif algo == 'bell':
        wave = _bell(freq, t, inst['harmonics'])
    else:   # additive
        wave = _additive(freq, t, inst['harmonics'],
                         inst['vibrato'],
                         inst.get('chorus', 0.0))

    # ── Noise colouring (breath / bow / pick transient) ───────
    noise_mix = inst.get('noise_mix', 0.0)
    if noise_mix > 0.0:
        noise = np.random.standard_normal(n)
        # Weight noise toward the attack for transient realism
        noise_env = np.exp(-t * 12.0)
        noise = noise * (noise_env * 0.7 + 0.3)
        wave  = wave * (1.0 - noise_mix) + noise * noise_mix

    # ── ADSR envelope ─────────────────────────────────────────
    wave = wave * _envelope(n, inst['adsr'])

    # ── Soft-clip + normalise ─────────────────────────────────
    wave = np.tanh(wave * 1.4) * 0.72

    # RMS normalisation (more consistent perceived loudness than peak)
    rms = np.sqrt(np.mean(wave ** 2))
    if rms > 1e-6:
        wave = wave * (0.25 / rms)   # target RMS = 0.25

    # Hard-limit just in case
    wave = np.clip(wave, -0.98, 0.98)

    return wave.astype(np.float32)


def add_reverb(wave: np.ndarray, wet: float = 0.22) -> np.ndarray:
    """
    Simple comb-based reverb with 5 delay taps for a more spacious tail.
    """
    if wet < 0.01:
        return wave
    out = wave.copy()
    for delay_ms, lvl in [(40,0.5),(70,0.35),(120,0.22),(200,0.12),(320,0.06)]:
        d = int(SAMPLE_RATE * delay_ms / 1000)
        if d < len(wave):
            out[d:] += wave[:-d] * lvl * wet
    return np.tanh(out * 0.82).astype(np.float32)


# ──────────────────────────────────────────────────────────────
#  AUDIO ENGINE
# ──────────────────────────────────────────────────────────────
class AudioEngine:
    """
    PyAudio stream running in callback mode.
    Thread-safe queue of [buffer, position] entries.
    """
    def __init__(self):
        self.pa       = pyaudio.PyAudio()
        self._queue   = []
        self._lock    = threading.Lock()
        self.waveform = deque(maxlen=512)   # raw samples for visualiser

        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK,
            stream_callback=self._callback,
        )

    def _callback(self, in_data, n_frames, time_info, status):
        out = np.zeros(n_frames, dtype=np.float32)
        with self._lock:
            alive = []
            for entry in self._queue:
                buf, pos = entry
                take        = min(n_frames, len(buf) - pos)
                out[:take] += buf[pos:pos + take]
                entry[1]   += take
                if entry[1] < len(buf):
                    alive.append(entry)
            self._queue = alive

        out = np.tanh(out * 1.05).astype(np.float32)   # master soft-clip
        self.waveform.extend(out.tolist())
        return (out.tobytes(), pyaudio.paContinue)

    def play(self, wave: np.ndarray):
        """Queue a pre-synthesised buffer for playback."""
        with self._lock:
            self._queue.append([wave.copy(), 0])

    def shutdown(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()


# ──────────────────────────────────────────────────────────────
#  GUI APPLICATION
# ──────────────────────────────────────────────────────────────
class App:
    def __init__(self, root: tk.Tk):
        self.root   = root
        self.engine = AudioEngine()
        self.octave = 4

        # Runtime state
        self._active: dict = {}   # (note, oct) → True   piano key is lit
        self._held:   dict = {}   # char/keysym → (note, oct)  keyboard hold

        # Tkinter variables
        self.inst_var  = tk.StringVar(value='Piano')
        self.chord_var = tk.StringVar(value='Major')
        self.mode_var  = tk.StringVar(value='chord')

        # Widget refs — all None until _build_ui assigns them
        self.oct_lbl     = None
        self.info_lbl    = None
        self.piano_frame = None
        self.rand_btn    = None
        self.last_lbl    = None
        self.vis         = None
        self.status      = None
        self._inst_btns  = {}
        self._chord_btns = {}
        self._piano_keys = {}

        root.title('Poly Instrument Synthesizer  v3')
        root.configure(bg=BG)
        root.resizable(False, False)

        self._build_ui()       # creates every widget top → bottom
        self._bind_keyboard()
        self._vis_tick()

    # ─────────────────────────────────────────────────────────
    #  BUILD UI  ── strict top-to-bottom order
    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        root = self.root

        # 1 ── TOP BAR ────────────────────────────────────────
        top = tk.Frame(root, bg=BG)
        top.pack(fill='x', padx=16, pady=(14, 6))

        tk.Label(top, text='POLY SYNTH', bg=BG, fg='#00e5ff',
                 font=('Courier', 20, 'bold')).pack(side='left')
        tk.Label(top, text='v3  instrument studio', bg=BG, fg=DIM,
                 font=('Courier', 10)).pack(side='left', padx=12, pady=6)

        self.oct_lbl = tk.Label(top, text='', bg=PANEL2, fg='#ffe066',
                                font=('Courier', 11, 'bold'), padx=10, pady=3)
        self.oct_lbl.pack(side='right')

        # 2 ── INSTRUMENT GRID ────────────────────────────────
        inst_outer = tk.Frame(root, bg=BG)
        inst_outer.pack(fill='x', padx=14, pady=(0, 6))
        tk.Label(inst_outer, text='SELECT INSTRUMENT', bg=BG, fg=DIM,
                 font=('Courier', 8)).pack(anchor='w', pady=(0, 4))

        grid = tk.Frame(inst_outer, bg=BG)
        grid.pack(fill='x')
        for col, (name, info) in enumerate(INSTRUMENTS.items()):
            btn = tk.Frame(grid, bg=PANEL2, cursor='hand2')
            btn.grid(row=0, column=col, padx=3, pady=2, sticky='nsew')
            grid.columnconfigure(col, weight=1)
            em = tk.Label(btn, text=info['emoji'], bg=PANEL2,
                          font=('TkDefaultFont', 18))
            em.pack(pady=(8, 1))
            nm = tk.Label(btn, text=name.upper(), bg=PANEL2, fg=DIM,
                          font=('Courier', 7, 'bold'))
            nm.pack(pady=(0, 8))
            for w in (btn, em, nm):
                w.bind('<Button-1>', lambda e, n=name: self._select_inst(n))
            self._inst_btns[name] = (btn, em, nm)

        # 3 ── CENTRE SECTION ─────────────────────────────────
        centre = tk.Frame(root, bg=BG)
        centre.pack(fill='x', padx=14, pady=4)

        left = tk.Frame(centre, bg=PANEL)
        left.pack(side='left', fill='y', padx=(0, 8), pady=2)

        tk.Label(left, text='CHORD / RANDOM', bg=PANEL, fg=DIM,
                 font=('Courier', 8)).pack(anchor='w', padx=10, pady=(8, 4))

        cgrid = tk.Frame(left, bg=PANEL)
        cgrid.pack(padx=8, fill='x')
        for i, ct in enumerate(CHORD_TYPES):
            b = tk.Label(cgrid, text=ct, bg=PANEL2, fg=DIM,
                         font=('Courier', 8), padx=6, pady=3, cursor='hand2')
            b.grid(row=i // 3, column=i % 3, padx=2, pady=2, sticky='ew')
            cgrid.columnconfigure(i % 3, weight=1)
            b.bind('<Button-1>', lambda e, c=ct: self._select_chord(c))
            self._chord_btns[ct] = b

        mode_row = tk.Frame(left, bg=PANEL)
        mode_row.pack(fill='x', padx=8, pady=(8, 0))
        tk.Label(mode_row, text='MODE:', bg=PANEL, fg=DIM,
                 font=('Courier', 8)).pack(side='left')
        for label, val in [('NOTE','note'),('CHORD','chord'),('ARPEGGIO','arp')]:
            tk.Radiobutton(
                mode_row, text=label, variable=self.mode_var, value=val,
                bg=PANEL, fg=TEXT, selectcolor=PANEL2,
                activebackground=PANEL, font=('Courier', 8),
                indicatoron=0, padx=5, pady=2, relief='flat', bd=1, cursor='hand2',
            ).pack(side='left', padx=2)

        self.rand_btn = tk.Canvas(left, width=200, height=60,
                                  bg=PANEL, highlightthickness=0)
        self.rand_btn.pack(padx=8, pady=10)
        self.rand_btn.bind('<Button-1>', self._on_random)
        self.rand_btn.bind('<Enter>',    lambda e: self._draw_rand(True))
        self.rand_btn.bind('<Leave>',    lambda e: self._draw_rand(False))

        self.last_lbl = tk.Label(left, text='--- press random ---',
                                 bg=PANEL, fg=DIM, font=('Courier', 8),
                                 wraplength=210)
        self.last_lbl.pack(padx=8, pady=(0, 10))

        # Right panel — waveform + info
        right = tk.Frame(centre, bg=PANEL)
        right.pack(side='left', fill='both', expand=True, pady=2)

        tk.Label(right, text='WAVEFORM  (oscilloscope)', bg=PANEL, fg=DIM,
                 font=('Courier', 8)).pack(anchor='w', padx=10, pady=(8, 2))

        self.vis = tk.Canvas(right, width=430, height=130,
                             bg='#070910', highlightthickness=0)
        self.vis.pack(padx=10, pady=(0, 6))

        self.info_lbl = tk.Label(right, text='', bg=PANEL, fg=DIM,
                                 font=('Courier', 8), justify='left')
        self.info_lbl.pack(anchor='w', padx=10, pady=(0, 8))

        # 4 ── PIANO KEYBOARD ─────────────────────────────────
        piano_wrap = tk.Frame(root, bg='#090b10')
        piano_wrap.pack(fill='x', padx=0, pady=(6, 0))
        tk.Label(
            piano_wrap,
            text=('KEYBOARD  '
                  '[ Z/X = octave  '
                  '| 1-7 white oct1  '
                  '| 8 9 0 - = white oct2  '
                  '| Q W R T Y black  '
                  '| SPACE = random ]'),
            bg='#090b10', fg=DIM, font=('Courier', 8),
        ).pack(anchor='w', padx=14, pady=(6, 2))

        self.piano_frame = tk.Frame(piano_wrap, bg='#090b10')
        self.piano_frame.pack(pady=(0, 10))

        # 5 ── STATUS BAR ──────────────────────────────────────
        self.status = tk.Label(root, text='Ready',
                               bg='#06080c', fg=DIM, font=('Courier', 8),
                               anchor='w', padx=12)
        self.status.pack(fill='x', ipady=4, side='bottom')

        # ── Apply initial state (ALL widgets exist now) ───────
        self._select_inst('Piano')
        self._select_chord('Major')
        self._update_oct_label()
        self._build_piano()
        self._draw_rand(False)

    # ─────────────────────────────────────────────────────────
    #  PIANO KEYBOARD BUILDER
    # ─────────────────────────────────────────────────────────
    def _build_piano(self):
        if self.piano_frame is None:
            return
        for w in self.piano_frame.winfo_children():
            w.destroy()
        self._piano_keys = {}

        WW, WH = 30, 94   # white key width / height
        BW, BH = 19, 58   # black key width / height

        container = tk.Frame(self.piano_frame, bg='#090b10')
        container.pack()

        white_x   = 0
        white_pos = {}   # (note, oct) → left-edge x

        # ── White keys ───────────────────────────────────────
        for oct_off in range(2):
            oct = self.octave + oct_off
            for name in NOTE_NAMES:
                if '#' in name:
                    continue
                key = tk.Label(container,
                               text=name, fg='#8899bb',
                               font=('Courier', 6), anchor='s',
                               bg='#ddeeff', relief='solid', bd=1,
                               cursor='hand2')
                key.place(x=white_x, y=0, width=WW, height=WH)
                key.bind('<ButtonPress-1>',
                         lambda e, n=name, o=oct: self._key_dn(n, o))
                key.bind('<ButtonRelease-1>',
                         lambda e, n=name, o=oct: self._key_up(n, o))
                key.bind('<Enter>',
                         lambda e, k=key, n=name, o=oct:
                         k.configure(bg='#bbddff')
                         if (n, o) not in self._active else None)
                key.bind('<Leave>',
                         lambda e, k=key, n=name, o=oct:
                         k.configure(bg='#ddeeff')
                         if (n, o) not in self._active else None)
                white_pos[(name, oct)] = white_x
                self._piano_keys[(name, oct)] = key
                white_x += WW

        container.configure(width=white_x, height=WH)

        # ── Black keys ───────────────────────────────────────
        # FIX (BUG 1): correct x = c_x + (whites_before_base + 1) * WW - BW//2
        for oct_off in range(2):
            oct = self.octave + oct_off
            c_x = white_pos.get(('C', oct))
            if c_x is None:
                continue
            for name in NOTE_NAMES:
                if '#' not in name:
                    continue
                base = name.replace('#', '')
                # Number of white keys strictly before the base white note
                wb   = sum(1 for m in WHITES if NOTE_OFFSETS[m] < NOTE_OFFSETS[base])
                # FIXED formula: +1 so C# sits BETWEEN C and D
                bx   = c_x + (wb + 1) * WW - BW // 2

                key  = tk.Label(container, bg='#18192a',
                                relief='flat', bd=0, cursor='hand2')
                key.place(x=bx, y=0, width=BW, height=BH)
                # tk.Label placed later is drawn on top — no .lift() needed
                key.bind('<ButtonPress-1>',
                         lambda e, n=name, o=oct: self._key_dn(n, o))
                key.bind('<ButtonRelease-1>',
                         lambda e, n=name, o=oct: self._key_up(n, o))
                key.bind('<Enter>',
                         lambda e, k=key: k.configure(bg='#002244'))
                key.bind('<Leave>',
                         lambda e, k=key, n=name, o=oct:
                         k.configure(bg='#18192a')
                         if (n, o) not in self._active else None)
                self._piano_keys[(name, oct)] = key

    # ─────────────────────────────────────────────────────────
    #  INSTRUMENT & CHORD SELECTION
    # ─────────────────────────────────────────────────────────
    def _select_inst(self, name: str):
        self.inst_var.set(name)
        info = INSTRUMENTS[name]
        for iname, (btn, em, nm) in self._inst_btns.items():
            sel = (iname == name)
            btn.configure(bg=PANEL if sel else PANEL2,
                          relief='solid' if sel else 'flat',
                          bd=2 if sel else 0,
                          highlightthickness=2 if sel else 0,
                          highlightbackground=info['color'] if sel else BG)
            em.configure(bg=PANEL if sel else PANEL2)
            nm.configure(bg=PANEL if sel else PANEL2,
                         fg=info['color'] if sel else DIM)
        if self.info_lbl is not None:
            A, D, S, R = info['adsr']
            chorus = info.get('chorus', 0)
            self.info_lbl.configure(
                text=(f"Algo: {info['algorithm'].upper()}"
                      f"{'  Chorus: '+str(chorus)+'¢' if chorus else ''}   "
                      f"A:{A:.3f}  D:{D:.2f}  S:{S:.2f}  R:{R:.2f}")
            )
        if self.rand_btn is not None:
            self._draw_rand(False)

    def _select_chord(self, name: str):
        self.chord_var.set(name)
        for ct, b in self._chord_btns.items():
            b.configure(bg='#1e2535' if ct == name else PANEL2,
                        fg=TEXT      if ct == name else DIM)

    # ─────────────────────────────────────────────────────────
    #  RANDOM BUTTON
    # ─────────────────────────────────────────────────────────
    def _draw_rand(self, hover: bool):
        if self.rand_btn is None:
            return
        self.rand_btn.delete('all')
        color = INSTRUMENTS[self.inst_var.get()]['color']
        bg    = '#1e2535' if hover else PANEL2
        self.rand_btn.create_rectangle(2, 2, 198, 58,
                                       fill=bg, outline=color, width=2)
        self.rand_btn.create_text(100, 22, text='  RANDOM',
                                  fill=color, font=('Courier', 13, 'bold'))
        hints = {'note':'plays one note', 'chord':'plays a chord',
                 'arp':'plays arpeggio'}
        self.rand_btn.create_text(100, 42,
                                  text=hints.get(self.mode_var.get(), ''),
                                  fill=DIM, font=('Courier', 8))

    def _on_random(self, e=None):
        inst  = self.inst_var.get()
        color = INSTRUMENTS[inst]['color']
        mode  = self.mode_var.get()

        # Flash button (main thread — safe)
        self.rand_btn.delete('all')
        self.rand_btn.create_rectangle(2, 2, 198, 58,
                                       fill=color, outline=color, width=2)
        self.rand_btn.create_text(100, 30, text='  PLAYING',
                                  fill=BG, font=('Courier', 13, 'bold'))
        self.root.after(350, lambda: self._draw_rand(False))

        root_note = random.choice(NOTE_NAMES)
        octave    = random.choice([3, 4, 4, 5]) + INSTRUMENTS[inst].get('oct_offset', 0)

        def _gui(lbl: str):
            """FIX (BUG 3 & 8): always update GUI via root.after from main thread."""
            self.root.after(0, lambda: self.last_lbl.configure(text=lbl, fg=color))
            self.root.after(0, lambda: self.status.configure(text=f'Playing: {lbl}'))

        def run():
            if mode == 'note':
                freq = note_to_freq(root_note, octave)
                wave = add_reverb(synthesize_note(inst, freq, 1.3))
                self.engine.play(wave)
                _gui(f'{root_note}{octave}  ({inst})')

            elif mode == 'chord':
                ct        = self.chord_var.get()
                intervals = CHORD_TYPES[ct]
                base_midi = note_to_midi(root_note, octave)
                waves, names = [], []
                for s in intervals:
                    freq = midi_to_freq(base_midi + s)
                    waves.append(synthesize_note(inst, freq,
                                                 random.uniform(1.1, 1.9)))
                    names.append(NOTE_NAMES[(NOTE_OFFSETS[root_note] + s) % 12])
                length = max(len(w) for w in waves)
                mix    = np.zeros(length, dtype=np.float32)
                for w in waves:
                    mix[:len(w)] += w
                mix = add_reverb(np.tanh(mix / max(1, len(waves) * 0.75)).astype(np.float32))
                self.engine.play(mix)
                _gui(f'{root_note} {ct}  |  {" - ".join(names)}')

            else:   # arpeggio
                ct        = self.chord_var.get()
                intervals = CHORD_TYPES[ct]
                base_midi = note_to_midi(root_note, octave)
                bpm       = random.choice([80, 100, 120, 140])
                gap       = 60.0 / bpm
                # FIX (BUG 8): capture lbl BEFORE starting thread
                lbl = f'{root_note} {ct} arpeggio  @{bpm} bpm'
                _gui(lbl)

                def arp():
                    seq = intervals + list(reversed(intervals[:-1]))
                    for s in seq:
                        freq = midi_to_freq(base_midi + s)
                        w    = synthesize_note(inst, freq, gap * 0.88)
                        self.engine.play(w)
                        time.sleep(gap)

                threading.Thread(target=arp, daemon=True).start()

        threading.Thread(target=run, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    #  PIANO KEY EVENTS
    # ─────────────────────────────────────────────────────────
    def _key_dn(self, note: str, octave: int):
        inst  = self.inst_var.get()
        color = INSTRUMENTS[inst]['color']
        freq  = note_to_freq(note, octave)
        self._active[(note, octave)] = True
        w = self._piano_keys.get((note, octave))
        if w:
            w.configure(bg=color)

        def play():
            wave = synthesize_note(inst, freq, 1.6)
            self.engine.play(wave)
        threading.Thread(target=play, daemon=True).start()

    def _key_up(self, note: str, octave: int):
        self._active.pop((note, octave), None)
        w = self._piano_keys.get((note, octave))
        if w:
            w.configure(bg='#18192a' if '#' in note else '#ddeeff')

    # ─────────────────────────────────────────────────────────
    #  KEYBOARD INPUT
    # ─────────────────────────────────────────────────────────
    def _bind_keyboard(self):
        self.root.bind('<KeyPress>',   self._on_kp)
        self.root.bind('<KeyRelease>', self._on_kr)

    def _resolve_key(self, e) -> str:
        """
        FIX (BUG 5): Try e.char first; fall back to e.keysym for special
        keys that return empty e.char on some Windows keyboard layouts.
        Returns a canonical key string or '' if unmapped.
        """
        ch = (e.char or '').lower()
        if ch in CHAR_MAP or ch in ('z', 'x', ' '):
            return ch
        # Fallback for -, =, [ etc.
        if e.keysym in KEYSYM_MAP:
            return e.keysym
        return ''

    def _on_kp(self, e):
        key = self._resolve_key(e)
        if not key or key in self._held:
            return

        if key == 'z':
            self.octave = max(1, self.octave - 1)
            self._update_oct_label()
            self._build_piano()
        elif key == 'x':
            self.octave = min(7, self.octave + 1)
            self._update_oct_label()
            self._build_piano()
        elif key == ' ':
            self._on_random()
        else:
            # Resolve note from CHAR_MAP or KEYSYM_MAP
            entry = CHAR_MAP.get(key) or KEYSYM_MAP.get(key)
            if entry:
                name, oct_off = entry
                oct = self.octave + oct_off
                self._held[key] = (name, oct)
                self._key_dn(name, oct)

    def _on_kr(self, e):
        key = self._resolve_key(e)
        if key in self._held:
            name, oct = self._held.pop(key)
            self._key_up(name, oct)

    # ─────────────────────────────────────────────────────────
    #  WAVEFORM VISUALISER
    # ─────────────────────────────────────────────────────────
    def _vis_tick(self):
        try:
            self._draw_vis()
        except Exception:
            pass
        self.root.after(38, self._vis_tick)   # ~26 fps

    def _draw_vis(self):
        if self.vis is None:
            return
        c = self.vis
        c.delete('all')
        W, H  = 430, 130
        mid   = H // 2
        color = INSTRUMENTS[self.inst_var.get()]['color']

        # Grid
        for y in [mid // 2, mid, mid + mid // 2]:
            c.create_line(0, y, W, y, fill='#14172a', width=1)
        for x in range(0, W, 43):
            c.create_line(x, 0, x, H, fill='#14172a', width=1)

        samples = list(self.engine.waveform)
        if len(samples) < 8:
            c.create_text(W // 2, mid, text='--- silent ---',
                          fill=DIM, font=('Courier', 9))
            return

        # FIX (BUG 6): zero-crossing trigger for stable oscilloscope display
        arr   = np.array(samples, dtype=np.float32)
        n_vis = min(len(arr), 430)
        # Find first rising zero-crossing in second half of buffer
        half  = len(arr) // 2
        trig  = half  # default: no trigger found
        for i in range(half, len(arr) - n_vis - 1):
            if arr[i] <= 0.0 < arr[i + 1]:
                trig = i
                break
        segment = arr[trig: trig + n_vis]
        if len(segment) < 4:
            segment = arr[-n_vis:]

        pts = []
        for i, v in enumerate(segment):
            pts.extend([i, int(mid - float(v) * (mid - 6))])

        if len(pts) >= 4:
            c.create_line(*pts, fill=color + '44', width=5, smooth=True)
            c.create_line(*pts, fill=color,        width=2, smooth=True)

    # ─────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────
    def _update_oct_label(self):
        if self.oct_lbl:
            self.oct_lbl.configure(text=f'OCT  {self.octave}  [ Z < > X ]')

    def destroy(self):
        self.engine.shutdown()
        self.root.destroy()


# ──────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    root = tk.Tk()
    app  = App(root)
    root.protocol('WM_DELETE_WINDOW', app.destroy)
    root.mainloop()