"""Frame-by-frame animation of TCN and consensus on the BAR-ATH tactical ATD."""

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "starter"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from games import GAME, GAMES, require
from starter.utils import AWAY_COLOUR, BALL_COLOUR, HOME_COLOUR, draw_pitch

LABELS = ("H", "A", "D", "X")
LABEL_NAMES = {0: "HOME", 1: "AWAY", 2: "DEAD", 3: "UNKNOWN"}
LABEL_COLOURS = ("#d62728", "#1f77b4", "#7f7f7f", "#b0b0b0")
N_FRAMES = GAMES[GAME]["n_video_frames"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]
DEFAULT_TCN = ROOT / "evaluation" / "results" / "model_tcn_clean.csv"
DEFAULT_CONSENSUS = ROOT / "evaluation" / "results" / "model_consensus.csv"


@dataclass
class AtdWindow:
    start: int
    stop: int
    players: np.ndarray
    ball: np.ndarray
    team_of_slot: np.ndarray
    tcn: np.ndarray
    consensus: np.ndarray
    home_name: str
    away_name: str
    pitch_length: float = PITCH_LENGTH
    pitch_width: float = PITCH_WIDTH

    @property
    def frame_count(self):
        return self.stop - self.start

    @property
    def home_mask(self):
        return self.team_of_slot == 0

    @property
    def away_mask(self):
        return self.team_of_slot == 1


def _team_info(meta_path):
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if not teams:
        raise ValueError("no Team entries in {}".format(meta_path))
    names = [(team.findtext("Name") or "").strip() for team in teams]
    team_ids = [team.get("id") for team in teams]
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    mapping = np.array(
        [0 if slot == home_id else 1 for slot in slots], dtype=np.int8)
    home_name = names[0] or "Home"
    away_name = names[1] if len(names) > 1 and names[1] else "Away"
    return mapping, home_name, away_name


def _triplet(text):
    try:
        values = text.split(",")
        if len(values) < 2:
            return np.nan, np.nan, np.nan
        x = float(values[0])
        y = float(values[1])
        speed = float(values[2]) if len(values) > 2 else np.nan
        return x, y, speed
    except (TypeError, ValueError):
        return np.nan, np.nan, np.nan


def _read_labels(path):
    import pandas as pd

    table = pd.read_csv(path)
    table.columns = [str(column).strip().lower() for column in table.columns]
    if "label" not in table.columns:
        raise ValueError("no label column in {}".format(path))
    mapping = {"H": 0, "A": 1, "D": 2, "X": 3}
    labels = table["label"].astype(str).str.upper().map(mapping)
    if labels.isna().any():
        bad = sorted(set(table["label"].astype(str)) - set(mapping))
        raise ValueError("unknown labels in {}: {}".format(path, bad[:5]))
    values = labels.to_numpy(dtype=np.int8)
    if len(values) != N_FRAMES:
        raise ValueError(
            "{} has {} labels, expected {}".format(path, len(values), N_FRAMES))
    return values


def load_atd_window(start_frame=0, end_frame=N_FRAMES, data_dir=None,
                    source="tactical", tcn_path=None, consensus_path=None):
    """Parse only the requested video-frame range from the raw tactical ATD."""
    start = max(0, int(start_frame))
    stop = min(N_FRAMES, int(end_frame))
    if stop <= start:
        raise ValueError("end_frame must be greater than start_frame")

    raw_path, meta_path = require(
        ["atd_raw", "atd_meta"], data_dir=data_dir, source=source)
    team_of_slot, home_name, away_name = _team_info(meta_path)
    slots = len(team_of_slot)
    players = np.full((stop - start, slots, 3), np.nan, dtype=np.float32)
    ball = np.full((stop - start, 3), np.nan, dtype=np.float32)

    with open(raw_path, "r") as handle:
        for line in handle:
            first_sep = line.find(":")
            if first_sep < 0:
                continue
            frame = int(line[:first_sep]) - 1
            if frame < start:
                continue
            if frame >= stop:
                break
            last_sep = line.rfind(":")
            local = frame - start
            payload = line[first_sep + 1:last_sep]
            for slot, player in enumerate(payload.split(";")[:slots]):
                players[local, slot] = _triplet(player)
            ball[local] = _triplet(line[last_sep + 1:].strip())

    tcn = _read_labels(tcn_path or DEFAULT_TCN)
    consensus = _read_labels(consensus_path or DEFAULT_CONSENSUS)
    return AtdWindow(
        start=start,
        stop=stop,
        players=players,
        ball=ball,
        team_of_slot=team_of_slot,
        tcn=tcn,
        consensus=consensus,
        home_name=home_name,
        away_name=away_name,
    )


def _label_run(labels, frame):
    code = int(labels[frame])
    start = frame
    while start > 0 and int(labels[start - 1]) == code:
        start -= 1
    stop = frame + 1
    while stop < len(labels) and int(labels[stop]) == code:
        stop += 1
    return start, stop, code


def _player_xy(players, mask, local):
    points = players[local][mask]
    finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    return np.column_stack((
        points[finite, 0] * PITCH_LENGTH,
        points[finite, 1] * PITCH_WIDTH,
    ))


def _ball_xy(ball, local):
    point = ball[local]
    if not np.isfinite(point[0]) or not np.isfinite(point[1]):
        return np.empty((0, 2))
    return np.array([[point[0] * PITCH_LENGTH, point[1] * PITCH_WIDTH]])


def animate_barca(start_frame=11400, end_frame=11525, step=1, interval=40,
                  figsize=(15, 7)):
    """Animate the same BAR-ATH tracking with the TCN and consensus labels."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.lines import Line2D

    window = load_atd_window(start_frame, end_frame)
    frames = np.arange(window.start, window.stop, max(1, int(step)), dtype=int)
    if not len(frames):
        raise ValueError("empty frame range")

    figure = plt.figure(figsize=figsize)
    grid = figure.add_gridspec(1, 2, wspace=0.10)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    scatters = []
    titles = []
    for ax in axes:
        draw_pitch(ax, PITCH_LENGTH, PITCH_WIDTH)
        ax.set_xlabel("")
        home = ax.scatter(
            [], [], s=100, color=HOME_COLOUR, edgecolors="black",
            linewidths=0.5, zorder=3)
        away = ax.scatter(
            [], [], s=100, color=AWAY_COLOUR, edgecolors="black",
            linewidths=0.5, zorder=3)
        ball = ax.scatter(
            [], [], s=75, color=BALL_COLOUR, edgecolors="black",
            linewidths=1.2, zorder=4)
        scatters.append((home, away, ball))
        titles.append(ax.set_title(""))

    suptitle = figure.suptitle("")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HOME_COLOUR,
               markeredgecolor="black", markersize=9, label=window.home_name),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=AWAY_COLOUR,
               markeredgecolor="black", markersize=9, label=window.away_name),
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10)
    figure.subplots_adjust(
        left=0.04, right=0.98, top=0.84, bottom=0.10, wspace=0.10)

    def update(index):
        frame = int(frames[index])
        local = frame - window.start
        point = window.ball[local]
        has_ball = np.isfinite(point[0]) and np.isfinite(point[1])
        home_xy = _player_xy(window.players, window.home_mask, local)
        away_xy = _player_xy(window.players, window.away_mask, local)
        ball_xy = _ball_xy(window.ball, local)
        artists = []
        for ax, (home, away, ball), labels, name in zip(
                axes, scatters, (window.tcn, window.consensus),
                ("TCN", "CONSENSUS")):
            home.set_offsets(home_xy)
            away.set_offsets(away_xy)
            ball.set_offsets(ball_xy)
            run_start, run_stop, code = _label_run(labels, frame)
            suffix = "" if has_ball else "  -- ball not tracked"
            ax.set_title(
                "{}: {}  frames [{}, {}){}".format(
                    name, LABEL_NAMES[code], run_start, run_stop, suffix),
                color=LABEL_COLOURS[code], fontsize=12)
            artists.extend([home, away, ball])
        suptitle.set_text(
            "{}  vs  {}   |   frame {} / {}   |   {:02d}:{:05.2f}".format(
                window.home_name, window.away_name, frame, N_FRAMES - 1,
                frame // (25 * 60), (frame % (25 * 60)) / 25.0))
        return artists + [suptitle] + titles

    return FuncAnimation(
        figure, update, frames=len(frames), interval=interval, blit=False)
