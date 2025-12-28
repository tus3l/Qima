# Xmath – Trig Limits Trainer (Static)

A bilingual (AR/EN) single-page web app to practice trigonometric limits near x→0 with a clean, glassmorphism UI, professional math input, and a training mode.

## Features
- AR/EN toggle with smooth transition; full UI localization (titles, buttons, HUD, keypad, placeholders).
- Professional math input via MathLive: fractions, powers, roots, π, trig templates, and a mini-keypad.
- Tiny in-answer menu popover guiding users to advanced features (e.g., matrices future).
- Training mode: difficulty levels, streak/accuracy HUD, hints with Maclaurin expansions.
- Success applause SFX (with WebAudio fallback) and reaction images (optional).
- Dark/Light themes with glassy cards and subtle background blobs.

## Quick Run (Local)
```bash
# From the Xmath folder
python -m http.server -b 127.0.0.1 8000
# Open http://127.0.0.1:8000/index.html
```

## Deploy to GitHub Pages
1. Create a repo and push this folder (index.html, README.md, optional assets in `audio/` and `reactions/`).
2. GitHub → Settings → Pages → Build and deployment → Deploy from branch.
3. Branch: `main` (or `master`), Folder: `/root`.
4. The site will be served at `https://<user>.github.io/<repo>/`.

## Optional Assets
- `audio/clap.mp3` (and/or `audio/applause.wav`) — success sound.
- `reactions/manifest.json` + jpg images — training reaction images.
	- Without these, the app still works (image will simply not load).

## QA Checklist
- Language toggle (AR/EN):
	- Titles/descriptions, mode buttons, training toolbar, difficulty labels, keypad labels, math-field placeholder, HUD (points/streak/accuracy) all change.
- Answer input:
	- Fraction button inserts real fraction with visible placeholders; caret moves to numerator.
	- Manual virtual keyboard (no extra MathLive toggle icon).
	- Supports LaTeX fractions (\\frac/\\dfrac/\\tfrac), roots, powers, π, and trig.
- Training mode:
	- New Question works; Hint shows Maclaurin expansions; HUD updates.
- Visuals:
	- Dark/Light switch; cards stay within bounds; tip popover appears once above the in-answer menu.
- No console/network blockers: external CDNs (MathLive, MathJax) are reachable in production.

## Notes
- The Python `app.py` is separate (Streamlit version) and not required for the static page. If you use it, install: `pip install sympy antlr4-python3-runtime`.
- Pylance type-stub warnings for SymPy are expected and do not affect runtime.

## Customization
- Texts: edit the `STR` object in `index.html`.
- Keypad/menu: extend `handleInsMath()` and keypad buttons in `index.html`.
- Theme: tweak CSS variables near the top of `index.html`.
