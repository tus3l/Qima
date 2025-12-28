from pathlib import Path
from PIL import Image

def compress(folder: Path, max_width: int = 480, quality: int = 70):
    folder.mkdir(parents=True, exist_ok=True)
    for p in folder.glob('*.jpg'):
        try:
            img = Image.open(p).convert('RGB')
            w, h = img.size
            if w > max_width:
                new_h = int(h * (max_width / w))
                img = img.resize((max_width, new_h), Image.LANCZOS)
            out = p  # overwrite
            img.save(out, format='JPEG', quality=quality, optimize=True, progressive=True)
            print(f'Compressed: {p.name} -> {img.size}, quality={quality}')
        except Exception as e:
            print(f'Failed {p}: {e}')
    # write manifest.json with file names
    names = sorted([p.name for p in folder.glob('*.jpg')])
    (folder / 'manifest.json').write_text(__import__('json').dumps(names, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    root = Path(__file__).parent
    compress(root / 'reactions')
