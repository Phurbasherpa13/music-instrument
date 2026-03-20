"""
POLY INSTRUMENT SYNTHESIZER
Piano · Guitar · Violin · Trumpet · Flute · Bell · Bass · Organ · Marimba

Requirements:  pip install numpy pyaudio

Controls:
  Click instrument buttons to select timbre
  Click RANDOM  -> plays random note / chord / arpeggio
  Click piano keys to play manually
  Keyboard: 1 2 3 4 5 6 7 (white oct1)   8 9 0 - = (white oct2)
  Q W R T Y (black oct1)         U I O P [ (black oct2)
  Z / X  -> Octave down / up
  SPACE  -> Random trigger
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

CHORD_TYPES = {
    'Major':      [0, 4, 7],
    'Minor':      [0, 3, 7],
    'Dom7':       [0, 4, 7, 10],
    'Maj7':       [0, 4, 7, 11],
    'Min7':       [0, 3, 7, 10],
    'Sus4':       [0, 5, 7],
    'Diminished': [0, 3, 6],
    'Augmented':  [0, 4, 8],
    'Power':      [0, 7, 12],
    'Add9':       [0, 4, 7, 14],
}

QWERTY_MAP = {
    # White keys — octave 0
    '1': ('C',  0), '2': ('D',  0), '3': ('E',  0), '4': ('F',  0),
    '5': ('G',  0), '6': ('A',  0), '7': ('B',  0),
    # White keys — octave 1
    '8': ('C',  1), '9': ('D',  1), '0': ('E',  1), '-': ('F',  1),
    '=': ('G',  1),
    # Black keys — octave 0  (q=C# w=D#  r=F# t=G# y=A#)
    'q': ('C#', 0), 'w': ('D#', 0), 'r': ('F#', 0),
    't': ('G#', 0), 'y': ('A#', 0),
    # Black keys — octave 1
    'u': ('C#', 1), 'i': ('D#', 1), 'o': ('F#', 1),
    'p': ('G#', 1), '[': ('A#', 1),
}

# Colours
BG     = '#0b0d13'
PANEL  = '#13161f'
PANEL2 = '#1a1e2b'
TEXT   = '#d8e8f0'
DIM    = '#4a5570'

# ──────────────────────────────────────────────────────────────
#  INSTRUMENTS
# ──────────────────────────────────────────────────────────────
INSTRUMENTS = {
    'Piano':   {
        'emoji': '🎹', 'color': '#00e5ff', 'algorithm': 'additive',
        'harmonics': [(1,1.0),(2,0.5),(3,0.25),(4,0.18),(5,0.12),(6,0.08),(7,0.05)],
        'adsr': (0.005, 0.4, 0.3, 0.6), 'noise_mix': 0.003,
        'vibrato': (0, 0), 'oct_offset': 0,
    },
    'Guitar':  {
        'emoji': '🎸', 'color': '#ff9900', 'algorithm': 'karplus',
        'harmonics': [(1,1.0),(2,0.4),(3,0.2),(4,0.1)],
        'adsr': (0.003, 0.3, 0.1, 0.5), 'noise_mix': 0.01,
        'vibrato': (5.5, 0.003), 'oct_offset': 0,
    },
    'Violin':  {
        'emoji': '🎻', 'color': '#ff4d8d', 'algorithm': 'additive',
        'harmonics': [(1,0.8),(2,1.0),(3,0.6),(4,0.3),(5,0.2),(6,0.15)],
        'adsr': (0.12, 0.08, 0.85, 0.25), 'noise_mix': 0.018,
        'vibrato': (6.0, 0.008), 'oct_offset': 0,
    },
    'Trumpet': {
        'emoji': '🎺', 'color': '#ffe066', 'algorithm': 'additive',
        'harmonics': [(1,0.7),(2,1.0),(3,0.8),(4,0.6),(5,0.5),(6,0.4),(7,0.3)],
        'adsr': (0.08, 0.05, 0.9, 0.15), 'noise_mix': 0.005,
        'vibrato': (6.5, 0.007), 'oct_offset': 0,
    },
    'Flute':   {
        'emoji': '🪈', 'color': '#88ffcc', 'algorithm': 'additive',
        'harmonics': [(1,1.0),(2,0.25),(3,0.08),(4,0.04)],
        'adsr': (0.08, 0.05, 0.9, 0.2), 'noise_mix': 0.025,
        'vibrato': (5.8, 0.012), 'oct_offset': 0,
    },
    'Bell':    {
        'emoji': '🔔', 'color': '#cc99ff', 'algorithm': 'bell',
        'harmonics': [(1,1.0),(2.756,0.5),(5.404,0.25),(3.375,0.18),(7.081,0.12)],
        'adsr': (0.003, 0.8, 0.0, 2.5), 'noise_mix': 0.001,
        'vibrato': (0, 0), 'oct_offset': 0,
    },
    'Bass':    {
        'emoji': '🎵', 'color': '#ff6655', 'algorithm': 'karplus',
        'harmonics': [(1,1.0),(2,0.5),(3,0.35),(4,0.2)],
        'adsr': (0.004, 0.3, 0.15, 0.4), 'noise_mix': 0.008,
        'vibrato': (0, 0), 'oct_offset': -1,
    },
    'Organ':   {
        'emoji': '🎼', 'color': '#66ccff', 'algorithm': 'additive',
        'harmonics': [(1,1.0),(2,1.0),(3,0.8),(4,0.6),(5,0.5),(6,0.4),(8,0.3)],
        'adsr': (0.01, 0.0, 1.0, 0.05), 'noise_mix': 0.001,
        'vibrato': (7.0, 0.005), 'oct_offset': 0,
    },
    'Marimba': {
        'emoji': '🥁', 'color': '#ffaa44', 'algorithm': 'bell',
        'harmonics': [(1,1.0),(3.932,0.35),(9.538,0.12),(2.0,0.08)],
        'adsr': (0.003, 0.15, 0.0, 0.5), 'noise_mix': 0.002,
        'vibrato': (0, 0), 'oct_offset': 0,
    },
}

# ──────────────────────────────────────────────────────────────
#  MUSIC HELPERS
# ──────────────────────────────────────────────────────────────
def midi_to_freq(m):
    return 440.0 * 2 ** ((m - 69) / 12)

def note_to_midi(name, octave):
    return NOTE_OFFSETS[name] + (octave + 1) * 12

def note_to_freq(name, octave):
    return midi_to_freq(note_to_midi(name, octave))

# ──────────────────────────────────────────────────────────────
#  SYNTHESIS
# ──────────────────────────────────────────────────────────────
def _envelope(n, adsr):
    A, D, S, R = adsr
    sr      = SAMPLE_RATE
    a_s     = max(1, int(A * sr))
    d_s     = max(1, int(D * sr))
    r_s     = max(1, int(R * sr))
    sus_end = max(a_s + d_s, n - r_s)
    env     = np.ones(n)
    env[:a_s] = np.linspace(0, 1, a_s)
    de = min(d_s, max(0, sus_end - a_s))
    if de > 0:
        env[a_s:a_s + de] = np.linspace(1, S, de)
        env[a_s + de:sus_end] = S
    if n > sus_end:
        env[sus_end:] = np.linspace(S, 0, n - sus_end)
    return np.clip(env, 0, 1)


def _additive(freq, t, harmonics, vibrato):
    vib_r, vib_d = vibrato
    dt   = 1.0 / SAMPLE_RATE
    wave = np.zeros(len(t))
    lfo  = 1.0
    if vib_r > 0 and vib_d > 0:
        ramp = np.clip(t / max(t[-1] * 0.3, 1e-6), 0, 1)
        lfo  = 1.0 + vib_d * np.sin(2 * np.pi * vib_r * t) * ramp
    freq_t = freq * lfo
    for mult, amp in harmonics:
        phase = np.cumsum(freq_t * mult) * dt * 2 * np.pi
        wave += amp * np.sin(phase)
    return wave


def _karplus(freq, n, harmonics):
    sr      = SAMPLE_RATE
    buf_len = max(2, int(sr / freq))
    buf     = np.random.uniform(-1, 1, buf_len)
    t0      = np.arange(buf_len) / sr
    col     = sum(amp * np.sin(2 * np.pi * freq * mult * t0)
                  for mult, amp in harmonics)
    buf     = buf * 0.3 + col * 0.7
    out     = np.zeros(n)
    for i in range(n):
        out[i] = buf[i % buf_len]
        j  = i % buf_len
        nj = (j + 1) % buf_len
        buf[j] = 0.498 * (buf[j] + buf[nj])
    return out


def _bell(freq, t, harmonics):
    dt   = 1.0 / SAMPLE_RATE
    wave = np.zeros(len(t))
    for i, (mult, amp) in enumerate(harmonics):
        decay = np.exp(-t * (0.5 + i * 1.2))
        phase = np.cumsum(np.full(len(t), freq * mult)) * dt * 2 * np.pi
        wave += amp * decay * np.sin(phase)
    return wave


def synthesize_note(inst_name, freq, duration):
    inst = INSTRUMENTS[inst_name]
    sr   = SAMPLE_RATE
    n    = int(sr * duration)
    t    = np.linspace(0, duration, n, endpoint=False)
    algo = inst['algorithm']

    if algo == 'karplus':
        wave = _karplus(freq, n, inst['harmonics'])
    elif algo == 'bell':
        wave = _bell(freq, t, inst['harmonics'])
    else:
        wave = _additive(freq, t, inst['harmonics'], inst['vibrato'])

    mix = inst.get('noise_mix', 0)
    if mix > 0:
        wave = wave * (1 - mix) + np.random.uniform(-1, 1, n) * mix

    wave = wave * _envelope(n, inst['adsr'])
    wave = np.tanh(wave * 1.5) * 0.7
    peak = np.max(np.abs(wave))
    if peak > 1e-6:
        wave /= peak
    return wave.astype(np.float32)


def add_reverb(wave, wet=0.18):
    if wet < 0.01:
        return wave
    out = wave.copy()
    for delay_ms, lvl in [(55, 0.4), (110, 0.2), (180, 0.1)]:
        d = int(SAMPLE_RATE * delay_ms / 1000)
        if d < len(wave):
            out[d:] += wave[:-d] * lvl * wet
    return np.tanh(out * 0.85).astype(np.float32)


# ──────────────────────────────────────────────────────────────
#  AUDIO ENGINE
# ──────────────────────────────────────────────────────────────
class AudioEngine:
    def __init__(self):
        self.pa       = pyaudio.PyAudio()
        self._queue   = []
        self._lock    = threading.Lock()
        self.waveform = deque(maxlen=256)
        self.stream   = self.pa.open(
            format=pyaudio.paFloat32, channels=1,
            rate=SAMPLE_RATE, output=True,
            frames_per_buffer=CHUNK,
            stream_callback=self._cb,
        )

    def _cb(self, in_data, n, time_info, status):
        out = np.zeros(n, dtype=np.float32)
        with self._lock:
            alive = []
            for entry in self._queue:
                buf, pos = entry
                take = min(n, len(buf) - pos)
                out[:take] += buf[pos:pos + take]
                entry[1]   += take
                if entry[1] < len(buf):
                    alive.append(entry)
            self._queue = alive
        out = np.tanh(out)
        self.waveform.extend(out[:32].tolist())
        return (out.tobytes(), pyaudio.paContinue)

    def play(self, wave):
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
    def __init__(self, root):
        self.root   = root
        self.engine = AudioEngine()
        self.octave = 4

        # State
        self._active = {}   # (note, oct) -> True  (piano key pressed)
        self._held   = {}   # qwerty char -> (note, oct)

        # Tkinter variables
        self.inst_var  = tk.StringVar(value='Piano')
        self.chord_var = tk.StringVar(value='Major')
        self.mode_var  = tk.StringVar(value='chord')

        # Widget references — initialised to None so we can guard
        # against accidental early calls before widgets are created
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

        root.title('Poly Instrument Synthesizer')
        root.configure(bg=BG)
        root.resizable(False, False)

        # Build everything in strict top-to-bottom order
        self._build_ui()
        self._bind_keyboard()
        self._vis_tick()

    # ─────────────────────────────────────────────────────────
    #  BUILD UI  (all widgets created before any state applied)
    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        root = self.root

        # ── 1. TOP BAR ───────────────────────────────────────
        top = tk.Frame(root, bg=BG)
        top.pack(fill='x', padx=16, pady=(14, 6))

        tk.Label(top, text='POLY SYNTH', bg=BG, fg='#00e5ff',
                 font=('Courier', 20, 'bold')).pack(side='left')
        tk.Label(top, text='instrument studio', bg=BG, fg=DIM,
                 font=('Courier', 10)).pack(side='left', padx=12, pady=6)

        # Oct label (no piano rebuild — just text)
        self.oct_lbl = tk.Label(top, text='', bg=PANEL2, fg='#ffe066',
                                font=('Courier', 11, 'bold'), padx=10, pady=3)
        self.oct_lbl.pack(side='right')

        # ── 2. INSTRUMENT GRID ───────────────────────────────
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

        # ── 3. CENTRE SECTION ────────────────────────────────
        centre = tk.Frame(root, bg=BG)
        centre.pack(fill='x', padx=14, pady=4)

        # Left panel  ── chord picker + random
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
        for label, val in [('NOTE', 'note'), ('CHORD', 'chord'), ('ARPEGGIO', 'arp')]:
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

        # Right panel  ── waveform + info
        right = tk.Frame(centre, bg=PANEL)
        right.pack(side='left', fill='both', expand=True, pady=2)

        tk.Label(right, text='WAVEFORM', bg=PANEL, fg=DIM,
                 font=('Courier', 8)).pack(anchor='w', padx=10, pady=(8, 2))

        self.vis = tk.Canvas(right, width=420, height=130,
                             bg='#0a0c12', highlightthickness=0)
        self.vis.pack(padx=10, pady=(0, 6))

        self.info_lbl = tk.Label(right, text='', bg=PANEL, fg=DIM,
                                 font=('Courier', 8), justify='left')
        self.info_lbl.pack(anchor='w', padx=10, pady=(0, 8))

        # ── 4. PIANO KEYBOARD ────────────────────────────────
        piano_wrap = tk.Frame(root, bg='#090b10')
        piano_wrap.pack(fill='x', padx=0, pady=(6, 0))
        tk.Label(
            piano_wrap,
            text='KEYBOARD  [ Z/X = octave  |  1-7 = white keys  |  Q W R T Y = black keys  |  SPACE = random ]',
            bg='#090b10', fg=DIM, font=('Courier', 8),
        ).pack(anchor='w', padx=14, pady=(6, 2))

        self.piano_frame = tk.Frame(piano_wrap, bg='#090b10')
        self.piano_frame.pack(pady=(0, 10))

        # ── 5. STATUS BAR ─────────────────────────────────────
        self.status = tk.Label(root, text='Ready',
                               bg='#06080c', fg=DIM, font=('Courier', 8),
                               anchor='w', padx=12)
        self.status.pack(fill='x', ipady=4, side='bottom')

        # ── Apply initial state NOW that every widget exists ──
        self._select_inst('Piano')
        self._select_chord('Major')
        self._update_oct_label()
        self._build_piano()
        self._draw_rand(False)

    # ─────────────────────────────────────────────────────────
    #  PIANO
    # ─────────────────────────────────────────────────────────
    def _build_piano(self):
        if self.piano_frame is None:
            return
        for w in self.piano_frame.winfo_children():
            w.destroy()
        self._piano_keys = {}

        WW, WH = 28, 90
        BW, BH = 18, 56

        container = tk.Frame(self.piano_frame, bg='#090b10')
        container.pack()

        white_x   = 0
        white_pos = {}

        # --- White keys: use tk.Label (avoids Canvas.lift() Windows bug) ---
        for oct_off in range(2):
            oct = self.octave + oct_off
            for name in NOTE_NAMES:
                if '#' in name:
                    continue
                key = tk.Label(container, text=name, fg='#99aabb',
                               font=('Courier', 6), anchor='s',
                               bg='#ddeeff', relief='solid', bd=1, cursor='hand2')
                key.place(x=white_x, y=0, width=WW, height=WH)
                key.bind('<ButtonPress-1>',
                         lambda e, n=name, o=oct: self._key_dn(n, o))
                key.bind('<ButtonRelease-1>',
                         lambda e, n=name, o=oct: self._key_up(n, o))
                key.bind('<Enter>',  lambda e, k=key, n=name, o=oct:
                         k.configure(bg='#bbddff') if (n, o) not in self._active else None)
                key.bind('<Leave>',  lambda e, k=key, n=name, o=oct:
                         k.configure(bg='#ddeeff') if (n, o) not in self._active else None)
                white_pos[(name, oct)] = white_x
                self._piano_keys[(name, oct)] = key
                white_x += WW

        container.configure(width=white_x, height=WH)

        # --- Black keys: tk.Label placed last so it renders on top naturally ---
        whites_order = [n for n in NOTE_NAMES if '#' not in n]

        for oct_off in range(2):
            oct = self.octave + oct_off
            c_x = white_pos.get(('C', oct))
            if c_x is None:
                continue
            for name in NOTE_NAMES:
                if '#' not in name:
                    continue
                base = name.replace('#', '')
                wb   = sum(1 for m in whites_order if NOTE_OFFSETS[m] < NOTE_OFFSETS[base])
                bx   = c_x + wb * WW - BW // 2
                key  = tk.Label(container, bg='#1a1a2e',
                                relief='flat', bd=0, cursor='hand2')
                key.place(x=bx, y=0, width=BW, height=BH)
                # No .lift() needed — Labels placed later appear on top automatically
                key.bind('<ButtonPress-1>',
                         lambda e, n=name, o=oct: self._key_dn(n, o))
                key.bind('<ButtonRelease-1>',
                         lambda e, n=name, o=oct: self._key_up(n, o))
                key.bind('<Enter>',  lambda e, k=key: k.configure(bg='#003355'))
                key.bind('<Leave>',  lambda e, k=key, n=name, o=oct:
                         k.configure(bg='#1a1a2e') if (n, o) not in self._active else None)
                self._piano_keys[(name, oct)] = key

    # ─────────────────────────────────────────────────────────
    #  SELECTION
    # ─────────────────────────────────────────────────────────
    def _select_inst(self, name):
        self.inst_var.set(name)
        info = INSTRUMENTS[name]
        for iname, (btn, em, nm) in self._inst_btns.items():
            sel = (iname == name)
            btn.configure(bg=PANEL   if sel else PANEL2,
                          relief='solid' if sel else 'flat', bd=2 if sel else 0,
                          highlightthickness=2 if sel else 0,
                          highlightbackground=info['color'] if sel else BG)
            em.configure(bg=PANEL if sel else PANEL2)
            nm.configure(bg=PANEL if sel else PANEL2,
                         fg=info['color'] if sel else DIM)
        if self.info_lbl is not None:
            A, D, S, R = info['adsr']
            self.info_lbl.configure(
                text=(f"Algo: {info['algorithm'].upper()}   "
                      f"A:{A:.3f}  D:{D:.2f}  S:{S:.2f}  R:{R:.2f}")
            )
        if self.rand_btn is not None:
            self._draw_rand(False)

    def _select_chord(self, name):
        self.chord_var.set(name)
        for ct, b in self._chord_btns.items():
            b.configure(bg='#1e2535' if ct == name else PANEL2,
                        fg=TEXT      if ct == name else DIM)

    # ─────────────────────────────────────────────────────────
    #  RANDOM
    # ─────────────────────────────────────────────────────────
    def _draw_rand(self, hover):
        if self.rand_btn is None:
            return
        self.rand_btn.delete('all')
        color = INSTRUMENTS[self.inst_var.get()]['color']
        bg    = '#1e2535' if hover else PANEL2
        self.rand_btn.create_rectangle(2, 2, 198, 58,
                                       fill=bg, outline=color, width=2)
        self.rand_btn.create_text(100, 22, text='  RANDOM',
                                  fill=color, font=('Courier', 13, 'bold'))
        hints = {'note': 'plays one note',
                 'chord': 'plays a chord',
                 'arp': 'plays arpeggio'}
        self.rand_btn.create_text(100, 42,
                                  text=hints.get(self.mode_var.get(), ''),
                                  fill=DIM, font=('Courier', 8))

    def _on_random(self, e=None):
        inst  = self.inst_var.get()
        color = INSTRUMENTS[inst]['color']
        mode  = self.mode_var.get()

        # Flash button
        self.rand_btn.delete('all')
        self.rand_btn.create_rectangle(2, 2, 198, 58,
                                       fill=color, outline=color, width=2)
        self.rand_btn.create_text(100, 30, text='  PLAYING',
                                  fill=BG, font=('Courier', 13, 'bold'))
        self.root.after(350, lambda: self._draw_rand(False))

        root_note = random.choice(NOTE_NAMES)
        oct_off   = INSTRUMENTS[inst].get('oct_offset', 0)
        octave    = random.choice([3, 4, 4, 5]) + oct_off

        def run():
            if mode == 'note':
                freq = note_to_freq(root_note, octave)
                wave = add_reverb(synthesize_note(inst, freq, 1.2))
                self.engine.play(wave)
                lbl  = f'{root_note}{octave}  ({inst})'

            elif mode == 'chord':
                ct        = self.chord_var.get()
                intervals = CHORD_TYPES[ct]
                base_midi = note_to_midi(root_note, octave)
                waves, names = [], []
                for s in intervals:
                    freq = midi_to_freq(base_midi + s)
                    waves.append(synthesize_note(inst, freq, random.uniform(1.0, 1.8)))
                    names.append(NOTE_NAMES[(NOTE_OFFSETS[root_note] + s) % 12])
                length = max(len(w) for w in waves)
                mix    = np.zeros(length, dtype=np.float32)
                for w in waves:
                    mix[:len(w)] += w
                mix = add_reverb(np.tanh(mix / max(1, len(waves) * 0.8)))
                self.engine.play(mix)
                lbl = f'{root_note} {ct}  |  {" - ".join(names)}'

            else:   # arpeggio
                ct        = self.chord_var.get()
                intervals = CHORD_TYPES[ct]
                base_midi = note_to_midi(root_note, octave)
                bpm       = random.choice([80, 100, 120, 140])
                gap       = 60.0 / bpm
                names     = []

                def arp():
                    seq = intervals + list(reversed(intervals[:-1]))
                    for s in seq:
                        freq = midi_to_freq(base_midi + s)
                        w    = synthesize_note(inst, freq, gap * 0.9)
                        self.engine.play(w)
                        names.append(NOTE_NAMES[(NOTE_OFFSETS[root_note] + s) % 12])
                        time.sleep(gap)

                threading.Thread(target=arp, daemon=True).start()
                lbl = f'{root_note} {ct} arpeggio @ {bpm} bpm'

            self.last_lbl.configure(text=lbl, fg=color)
            self.status.configure(text=f'Playing: {lbl}')

        threading.Thread(target=run, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    #  PIANO KEY EVENTS
    # ─────────────────────────────────────────────────────────
    def _key_dn(self, note, octave):
        inst  = self.inst_var.get()
        color = INSTRUMENTS[inst]['color']
        freq  = note_to_freq(note, octave)
        self._active[(note, octave)] = True
        w = self._piano_keys.get((note, octave))
        if w:
            w.configure(bg=color)

        def play():
            wave = synthesize_note(inst, freq, 1.5)
            self.engine.play(wave)
        threading.Thread(target=play, daemon=True).start()

    def _key_up(self, note, octave):
        self._active.pop((note, octave), None)
        w = self._piano_keys.get((note, octave))
        if w:
            w.configure(bg='#1a1a2e' if '#' in note else '#ddeeff')

    # ─────────────────────────────────────────────────────────
    #  KEYBOARD
    # ─────────────────────────────────────────────────────────
    def _bind_keyboard(self):
        self.root.bind('<KeyPress>',   self._on_kp)
        self.root.bind('<KeyRelease>', self._on_kr)

    def _on_kp(self, e):
        ch = e.char.lower() if e.char else ''
        if ch in self._held:
            return
        if ch == 'z':
            self.octave = max(1, self.octave - 1)
            self._update_oct_label()
            self._build_piano()
        elif ch == 'x':
            self.octave = min(7, self.octave + 1)
            self._update_oct_label()
            self._build_piano()
        elif ch == ' ':
            self._on_random()
        elif ch in QWERTY_MAP:
            name, oct_off = QWERTY_MAP[ch]
            oct = self.octave + oct_off
            self._held[ch] = (name, oct)
            self._key_dn(name, oct)

    def _on_kr(self, e):
        ch = e.char.lower() if e.char else ''
        if ch in self._held:
            name, oct = self._held.pop(ch)
            self._key_up(name, oct)

    # ─────────────────────────────────────────────────────────
    #  WAVEFORM VISUALISER
    # ─────────────────────────────────────────────────────────
    def _vis_tick(self):
        try:
            self._draw_vis()
        except Exception:
            pass
        self.root.after(40, self._vis_tick)

    def _draw_vis(self):
        if self.vis is None:
            return
        c = self.vis
        c.delete('all')
        W, H  = 420, 130
        mid   = H // 2
        color = INSTRUMENTS[self.inst_var.get()]['color']

        for y in [mid // 2, mid, mid + mid // 2]:
            c.create_line(0, y, W, y, fill='#181c28', width=1)
        for x in range(0, W, 40):
            c.create_line(x, 0, x, H, fill='#181c28', width=1)

        samples = list(self.engine.waveform)
        if len(samples) < 4:
            c.create_text(W // 2, mid, text='--- silent ---',
                          fill=DIM, font=('Courier', 9))
            return

        n   = min(len(samples), 256)
        pts = []
        for i, v in enumerate(samples[-n:]):
            pts.extend([int(i / n * W),
                        int(mid - v * (mid - 8))])
        if len(pts) >= 4:
            c.create_line(*pts, fill=color + '55', width=5, smooth=True)
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