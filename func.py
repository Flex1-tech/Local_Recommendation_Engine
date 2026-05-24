import sys
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog
import tkinter as tk
import fleep
from mutagen import File
import subprocess



def get_media_type(filepath):
    """
    Retourne: 'audio', 'video' ou None
    """
    try:
        with open(filepath, "rb") as file:
            info = fleep.get(file.read(128))

        types = info.type

        if not types:
            return None

        if "audio" in types:
            return "audio"
        elif "video" in types:
            return "video"

        return None

    except:
        return None

def is_valid_media(filepath):
    try:
        result = run_ffprobe([
            "ffprobe",
            "-v", "error",
            filepath
        ])

        return result.returncode == 0

    except subprocess.SubprocessError:
        return False


def run_ffprobe(cmd, timeout=5):
    kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "timeout": timeout
    }

    # Windows uniquement
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    return subprocess.run(cmd, **kwargs)

def is_audio_file(filepath):
    with open(filepath, "rb") as file:
        info = fleep.get(file.read(128))
    return "audio" in info.type or "video" in info.type  # Some audio files may be classified as video

def open_file():
    return filedialog.askopenfilenames(
        title="Select audio files",
        filetypes=[("All files", "*.*")]
    )

def show_toast(root, message, icon=None, duration=30000):

    error_icon = icon if icon else root.toast_icon

    toast = ctk.CTkToplevel(root)
    toast.overrideredirect(True)
    toast.attributes("-topmost", True)

    bg = root.cget("fg_color")
    if isinstance(bg, tuple):
        bg = bg[0]

    toast.configure(fg_color=bg)

    frame = ctk.CTkFrame(
        toast,
        corner_radius=10,
        border_width=2,
        border_color="#F36C19",
        fg_color=bg
    )
    frame.pack(fill="both", expand=True)

    label = ctk.CTkLabel(
        frame,
        text=message,
        image=error_icon,
        compound="left",
        text_color="#F36C19",
        font=("Arial", 12)
    )
    label.pack(padx=15, pady=10)

    def place_toast():
        toast.update_idletasks()

        x = root.winfo_rootx() + root.winfo_width() - toast.winfo_width() - 20
        y = root.winfo_rooty() + root.winfo_height() - toast.winfo_height() - 20

        toast.geometry(f"+{x}+{y}")

    toast.after(10, place_toast)

    toast.after(duration, toast.destroy)

