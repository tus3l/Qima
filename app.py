import streamlit as st
import sympy as sp
try:
    # Streamlit component for visual math input (MathQuill/MathLive)
    from st_math_input import st_math_input  # type: ignore
except Exception:
    st_math_input = None
try:
    from sympy.parsing.latex import parse_latex
except Exception:
    parse_latex = None
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ==========================
# Page Config
# ==========================
st.set_page_config(page_title="Xmath: Trig Limits via Maclaurin", page_icon="🧮", layout="centered")

# ==========================
# Utility: Applause Generator (base64 WAV, no external files)
# ==========================
import io, wave, struct, random, math, base64

def generate_applause_wav_base64(duration: float = 1.2, sample_rate: int = 22050) -> str:
    n_samples = int(duration * sample_rate)
    audio = []
    # Create multiple short bursts to resemble claps
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
        # clip
        value = max(-1.0, min(1.0, value))
        audio.append(int(value * 32767))
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(struct.pack('<h', s) for s in audio))
    return base64.b64encode(buf.getvalue()).decode('ascii')

@st.cache_data(show_spinner=False)
def get_applause_b64() -> str:
    return generate_applause_wav_base64()

# ==========================
# Styling (Glassmorphism + Theme)
# ==========================
LIGHT_VARS = {
    '--bg': '#f5f7ff',
    '--text': '#0f172a',
    '--muted': '#475569',
    '--accent': '#6366f1',
    '--glass-bg': 'rgba(255, 255, 255, 0.45)',
    '--glass-border': 'rgba(255, 255, 255, 0.6)'
}
DARK_VARS = {
    '--bg': '#0b1220',
    '--text': '#e5e7eb',
    '--muted': '#94a3b8',
    '--accent': '#a78bfa',
    '--glass-bg': 'rgba(17, 24, 39, 0.45)',
    '--glass-border': 'rgba(148, 163, 184, 0.4)'
}

CSS_BASE = """
:root {{
  {vars}
}}

html, body, [data-testid="stAppViewContainer"] {{
  background: var(--bg) !important;
  color: var(--text) !important;
}}

/* Direction will be injected dynamically */

.glass-card {{
  backdrop-filter: blur(16px) saturate(180%);
  -webkit-backdrop-filter: blur(16px) saturate(180%);
  background-color: var(--glass-bg);
  border: 1px solid var(--glass-border);
  border-radius: 18px;
  padding: 20px 24px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}}

.btn-accent button {{
  background: var(--accent) !important;
  color: white !important;
  border-radius: 10px !important;
}}

h1, h2, h3, h4, h5, h6 {{
  color: var(--text) !important;
}}

.muted {{ color: var(--muted); }}
.reaction {{ font-size: 2.2rem; line-height: 1; }}
.small {{ font-size: 0.95rem; }}
.center {{ text-align: center; }}
"""

# ==========================
# Localization
# ==========================
@dataclass
class Translator:
    lang: str = 'EN'  # 'EN' or 'AR'

    def t(self, key: str) -> str:
        return STRINGS[self.lang].get(key, key)

    @property
    def rtl(self) -> bool:
        return self.lang == 'AR'

STRINGS: Dict[str, Dict[str, str]] = {
    'EN': {
        'title': 'Trigonometric Limits via Maclaurin',
        'desc': 'Explore limits as x → a for sin, cos, tan using series.',
        'mode': 'Mode',
        'quick': 'Quick Solve',
        'steps': 'Step-by-Step',
        'train': 'Training Mode',
        'expr': 'Expression (only sin, cos, tan with x)',
        'order': 'Series Order (accuracy)',
        'solve': 'Solve',
        'limit_result': 'Limit Result',
        'series_expansions': 'Maclaurin Expansions',
        'simplified_series': 'Simplified Series (after substitution)',
        'final_limit': 'Final Limit as x → a',
        'approach_point': 'Approach point',
        'invalid': 'Invalid input. Use only sin, cos, tan and variable x.',
        'theme': 'Theme',
        'theme_light': 'Light',
        'theme_dark': 'Dark',
        'lang': 'Language',
        'answer': 'Your Answer',
        'check': 'Check Answer',
        'success': 'Great job! Correct answer 🎉',
        'fail': 'Not quite—try again!',
        'examples': 'Examples: sin(x)/x, tan(x)/x, (1-cos(x))/x**2',
    },
    'AR': {
        'title': 'نهايات المثلثات بسلسلة ماكلوران',
        'desc': 'اكتشف النهايات عندما x → a للدوال sin وcos وtan باستخدام السلاسل.',
        'mode': 'الوضع',
        'quick': 'حل سريع',
        'steps': 'خطوة بخطوة',
        'train': 'وضع التدريب',
        'expr': 'التعبير (فقط sin وcos وtan مع المتغير x)',
        'order': 'رتبة السلسلة (الدقة)',
        'solve': 'احسب',
        'limit_result': 'نتيجة النهاية',
        'series_expansions': 'توسيعات ماكلوران',
        'simplified_series': 'سلسلة مبسطة (بعد الاستبدال)',
        'final_limit': 'النهاية عندما x → a',
        'approach_point': 'نقطة الاقتراب',
        'invalid': 'إدخال غير صالح. استخدم فقط sin وcos وtan والمتغير x.',
        'theme': 'المظهر',
        'theme_light': 'فاتح',
        'theme_dark': 'داكن',
        'lang': 'اللغة',
        'answer': 'إجابتك',
        'check': 'تحقق من الإجابة',
        'success': 'عمل رائع! إجابة صحيحة 🎉',
        'fail': 'ليست صحيحة—حاول مرة أخرى!',
        'examples': 'أمثلة: sin(x)/x ، tan(x)/x ، (1-cos(x))/x**2',
    },
}

# ==========================
# Math Engine (OOP)
# ==========================
class MaclaurinEngine:
    def __init__(self):
        self.x = sp.symbols('x')
        self.allowed_locals = {'x': self.x, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan}

    def parse(self, expr_str: str) -> Optional[sp.Expr]:
        try:
            expr = sp.sympify(expr_str, locals=self.allowed_locals)
        except Exception:
            return None
        # Validate only x symbol and allowed functions
        # Symbols other than x
        symbols = {s for s in expr.free_symbols if s != self.x}
        if symbols:
            return None
        # Only allowed functions
        funcs = expr.atoms(sp.Function)
        for f in funcs:
            if f.func not in {sp.sin, sp.cos, sp.tan}:
                return None
        return expr

    def quick_limit(self, expr_str: str, a: sp.Expr = sp.Integer(0)) -> Tuple[bool, Optional[sp.Expr]]:
        expr = self.parse(expr_str)
        if expr is None:
            return False, None
        try:
            L = sp.limit(expr, self.x, a)
            return True, sp.simplify(L)
        except Exception:
            return False, None

    def series_steps(self, expr_str: str, order: int = 7, a: sp.Expr = sp.Integer(0)) -> Tuple[bool, Dict[str, List[str]]]:
        expr = self.parse(expr_str)
        if expr is None:
            return False, {}
        x = self.x
        steps: Dict[str, List[str]] = {
            'expansions': [],
            'simplified': [],
            'final': []
        }
        # Individual expansions for functions present
        present = {
            'sin': any(isinstance(node, sp.sin) for node in sp.preorder_traversal(expr)),
            'cos': any(isinstance(node, sp.cos) for node in sp.preorder_traversal(expr)),
            'tan': any(isinstance(node, sp.tan) for node in sp.preorder_traversal(expr)),
        }
        try:
            if present['sin']:
                s = sp.series(sp.sin(x), x, a, order).removeO()
                steps['expansions'].append(sp.latex(sp.Eq(sp.sin(x), s)))
            if present['cos']:
                c = sp.series(sp.cos(x), x, a, order).removeO()
                steps['expansions'].append(sp.latex(sp.Eq(sp.cos(x), c)))
            if present['tan']:
                t = sp.series(sp.tan(x), x, a, order).removeO()
                steps['expansions'].append(sp.latex(sp.Eq(sp.tan(x), t)))

            series_expr = sp.series(expr, x, a, order).removeO()
            steps['simplified'].append(sp.latex(sp.Eq(sp.Symbol('S(x)'), sp.simplify(series_expr))))
            L = sp.limit(series_expr, x, a)
            steps['final'].append(sp.latex(sp.Eq(sp.Symbol('\\lim_{x\\to a}'), L)))
            return True, steps
        except Exception:
            return False, {}

# ==========================
# UI Controller (OOP)
# ==========================
class App:
    def __init__(self):
        if 'lang' not in st.session_state:
            st.session_state.lang = 'EN'
        if 'theme' not in st.session_state:
            st.session_state.theme = 'Light'
        self.tr = Translator(st.session_state.lang)
        self.engine = MaclaurinEngine()

    def inject_css(self):
        vars_map = LIGHT_VARS if st.session_state.theme == 'Light' else DARK_VARS
        vars_str = '\n  '.join(f"{k}: {v};" for k, v in vars_map.items())
        css = CSS_BASE.format(vars=vars_str)
        # Direction
        dir_css = f"html, body, [data-testid='stAppViewContainer'] {{ direction: {'rtl' if self.tr.rtl else 'ltr'}; }}"
        st.markdown(f"<style>{css}\n{dir_css}</style>", unsafe_allow_html=True)

    def sidebar(self):
        st.sidebar.header("🧭")
        # Language
        lang_label = self.tr.t('lang')
        lang = st.sidebar.radio(lang_label, options=['EN', 'AR'], index=0 if st.session_state.lang=='EN' else 1, key='lang_radio')
        if lang != st.session_state.lang:
            st.session_state.lang = lang
            self.tr = Translator(lang)
        # Theme
        theme_label = self.tr.t('theme')
        theme = st.sidebar.radio(theme_label, options=['Light', 'Dark'], index=0 if st.session_state.theme=='Light' else 1, key='theme_radio')
        if theme != st.session_state.theme:
            st.session_state.theme = theme
        st.sidebar.markdown(f"<div class='small muted'>{self.tr.t('examples')}</div>", unsafe_allow_html=True)

    def header(self):
        st.markdown("<div class='glass-card'>"+
                    f"<h2>🧮 {self.tr.t('title')}</h2>"+
                    f"<div class='muted'>{self.tr.t('desc')}</div>"+
                    "</div>", unsafe_allow_html=True)

    def quick_solve(self):
        x = self.engine.x
        expr_label = self.tr.t('expr')
        expr_str = st.text_input(expr_label, value="sin(x)/x", key='expr_q')
        # Approach point selector
        a_label = self.tr.t('approach_point')
        a_str = st.text_input(a_label, value="0", key='a_q')
        try:
            a_val = sp.sympify(a_str, locals={'pi': sp.pi}) if a_str.strip() else sp.Integer(0)
        except Exception:
            a_val = sp.Integer(0)
        order = st.slider(self.tr.t('order'), min_value=3, max_value=15, value=7, step=2, key='order_q')
        solve = st.button(self.tr.t('solve'), type='primary', key='solve_q')
        if solve:
            ok, result = self.engine.quick_limit(expr_str, a=a_val)
            card_open = "<div class='glass-card'>"
            if not ok:
                st.markdown(card_open + f"<div class='reaction'>😕</div><p>{self.tr.t('invalid')}</p></div>", unsafe_allow_html=True)
                return
            st.markdown(card_open + f"<h4>{self.tr.t('limit_result')}</h4>", unsafe_allow_html=True)
            st.latex(sp.latex(sp.limit(self.engine.parse(expr_str), x, a_val)) + f"= {sp.latex(result)}")
            st.markdown("</div>", unsafe_allow_html=True)
            # Show series as a bonus detail
            ok2, steps = self.engine.series_steps(expr_str, order, a=a_val)
            if ok2:
                st.markdown("<div class='glass-card'>" + f"<h4>{self.tr.t('series_expansions')}</h4>", unsafe_allow_html=True)
                for s in steps['expansions']:
                    st.latex(s)
                st.markdown("</div>", unsafe_allow_html=True)

    def step_by_step(self):
        expr_str = st.text_input(self.tr.t('expr'), value="(1-cos(x))/x**2", key='expr_s')
        a_label = self.tr.t('approach_point')
        a_str = st.text_input(a_label, value="0", key='a_s')
        try:
            a_val = sp.sympify(a_str, locals={'pi': sp.pi}) if a_str.strip() else sp.Integer(0)
        except Exception:
            a_val = sp.Integer(0)
        order = st.slider(self.tr.t('order'), min_value=3, max_value=15, value=7, step=2, key='order_s')
        if st.button(self.tr.t('solve'), key='solve_s'):
            ok, steps = self.engine.series_steps(expr_str, order, a=a_val)
            if not ok:
                st.markdown("<div class='glass-card'><div class='reaction'>😕</div><p>" + self.tr.t('invalid') + "</p></div>", unsafe_allow_html=True)
                return
            st.markdown("<div class='glass-card'>" + f"<h4>{self.tr.t('series_expansions')}</h4>", unsafe_allow_html=True)
            for s in steps['expansions']:
                st.latex(s)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div class='glass-card'>" + f"<h4>{self.tr.t('simplified_series')}</h4>", unsafe_allow_html=True)
            for s in steps['simplified']:
                st.latex(s)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div class='glass-card'>" + f"<h4>{self.tr.t('final_limit')}</h4>", unsafe_allow_html=True)
            for s in steps['final']:
                st.latex(s)
            st.markdown("</div>", unsafe_allow_html=True)

    def training_mode(self):
        x = self.engine.x
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='muted'>{self.tr.t('examples')}</div>", unsafe_allow_html=True)

        # Approach point selector for training mode
        a_label = self.tr.t('approach_point')
        col_ap, _ = st.columns([1,3])
        with col_ap:
            a_str = st.text_input(a_label, value="0", key='a_t')
        try:
            a_val = sp.sympify(a_str, locals={'pi': sp.pi}) if a_str.strip() else sp.Integer(0)
        except Exception:
            a_val = sp.Integer(0)

        # Visual math inputs (LaTeX) with quick buttons
        colA, colB = st.columns(2)
        # Defaults held in session_state
        if 'latex_expr' not in st.session_state:
            st.session_state.latex_expr = r"\frac{\tan(x)}{x}"
        if 'latex_ans' not in st.session_state:
            st.session_state.latex_ans = r"1"

        with colA:
            st.write("**" + self.tr.t('expr') + "**")
            if st_math_input:
                latex_expr = st_math_input(value=st.session_state.latex_expr, height=60, key='latex_expr_input')
            else:
                latex_expr = st.text_input("LaTeX", value=st.session_state.latex_expr, key='latex_expr_fallback')
        with colB:
            st.write("**" + self.tr.t('answer') + "**")
            if st_math_input:
                latex_ans = st_math_input(value=st.session_state.latex_ans, height=60, key='latex_ans_input')
            else:
                latex_ans = st.text_input("LaTeX", value=st.session_state.latex_ans, key='latex_ans_fallback')

        # Quick insertion buttons
        b1, b2, b3, _ = st.columns([1,1,1,3])
        if b1.button("\u2044 كسر", key='btn_frac'):
            # Insert template at end (component will place cursor in numerator by default)
            st.session_state.latex_expr = st.session_state.latex_expr + r" \\frac{}{}"
            st.rerun()
        if b2.button("^ أس", key='btn_pow'):
            st.session_state.latex_expr = st.session_state.latex_expr + r" {}^{}"
            st.rerun()
        if b3.button("√ جذر", key='btn_sqrt'):
            st.session_state.latex_expr = st.session_state.latex_expr + r" \\sqrt{}"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Compute using LaTeX → SymPy
        if st.button(self.tr.t('check'), key='check_t'):
            expr_sym = None
            ans_sym = None
            try:
                if parse_latex:
                    expr_sym = parse_latex(latex_expr)
                    ans_sym = parse_latex(latex_ans)
                else:
                    # Fallback to plain parsing from text
                    expr_sym = self.engine.parse(latex_expr)
                    ans_sym = self.engine.parse(latex_ans)
            except Exception:
                expr_sym = None
                ans_sym = None

            if expr_sym is None or ans_sym is None:
                st.markdown("<div class='glass-card'><div class='reaction'>😕</div><p>" + self.tr.t('invalid') + "</p></div>", unsafe_allow_html=True)
                return

            try:
                L = sp.limit(expr_sym, x, a_val)
            except Exception:
                st.markdown("<div class='glass-card'><div class='reaction'>😕</div><p>" + self.tr.t('invalid') + "</p></div>", unsafe_allow_html=True)
                return

            # Compare
            success = False
            try:
                diff = sp.simplify(L - ans_sym)
                success = diff == 0
                if not success:
                    fnum = float(sp.N(diff.subs({x: 1e-6})))
                    success = abs(fnum) < 1e-6
            except Exception:
                success = False

            if success:
                st.markdown("<div class='glass-card center'><div class='reaction'>👏😄🎉</div><p>" + self.tr.t('success') + "</p></div>", unsafe_allow_html=True)
                b64 = get_applause_b64()
                st.markdown(f"<audio autoplay src='data:audio/wav;base64,{b64}'></audio>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='glass-card center'><div class='reaction'>😕</div><p>" + self.tr.t('fail') + "</p></div>", unsafe_allow_html=True)

    def run(self):
        self.sidebar()
        self.inject_css()
        self.header()
        mode = st.tabs([self.tr.t('quick'), self.tr.t('steps'), self.tr.t('train')])
        with mode[0]:
            self.quick_solve()
        with mode[1]:
            self.step_by_step()
        with mode[2]:
            self.training_mode()

if __name__ == '__main__':
    App().run()
