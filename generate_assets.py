import io, wave, struct, random, math, os
from pathlib import Path

# Generate applause.wav in audio/

def generate_applause_wav(path: Path, duration: float = 1.2, sample_rate: int = 22050):
    n_samples = int(duration * sample_rate)
    audio = []
    bursts = [random.randint(0, max(1, n_samples - int(0.05*sample_rate))) for _ in range(12)]
    burst_duration = int(0.015 * sample_rate)
    for i in range(n_samples):
        t = i / sample_rate
        base_env = math.exp(-2.0 * t)  # global decay
        value = random.uniform(-1, 1) * base_env * 0.25
        for b in bursts:
            if b <= i < b + burst_duration:
                local_env = 1.0 - (i - b) / max(1, burst_duration)
                value += random.uniform(-1, 1) * 0.9 * local_env
        value = max(-1.0, min(1.0, value))
        audio.append(int(value * 32767))
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(struct.pack('<h', s) for s in audio))
    path.write_bytes(buf.getvalue())

# Generate 4 JPG reaction placeholders in reactions/

def generate_reactions(folder: Path, size: int = 512):
    from PIL import Image, ImageDraw
    colors = [(220, 60, 60), (60, 140, 220), (240, 170, 50), (120, 200, 140)]
    for i in range(4):
        img = Image.new('RGB', (size, size), colors[i])
        d = ImageDraw.Draw(img)
        # draw simple sad face
        cx, cy, r = size//2, size//2, size//3
        # eyes
        d.ellipse((cx-80, cy-40, cx-40, cy), fill=(0,0,0))
        d.ellipse((cx+40, cy-40, cx+80, cy), fill=(0,0,0))
        # mouth (arc)
        d.arc((cx-100, cy+10, cx+100, cy+110), start=20, end=160, fill=(0,0,0), width=6)
        out = folder / f"reaction{i+1}.jpg"
        img.save(out, format='JPEG', quality=90)

if __name__ == '__main__':
    root = Path(__file__).parent
    audio_dir = root / 'audio'
    react_dir = root / 'reactions'
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(react_dir, exist_ok=True)
    generate_applause_wav(audio_dir / 'applause.wav')
    try:
        generate_reactions(react_dir)
    except ImportError:
        print('Pillow not installed; reactions placeholders were not generated.')
        print('Run: pip install pillow')
