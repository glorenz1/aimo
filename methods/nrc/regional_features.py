"""NRC lexical-region masks and cosine arithmetic; no labels or solutions."""
import re
import numpy as np

NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")
EPS=1e-30
CORE_FEATURES=['number_logrange','number_negative_fraction','number_decimal_fraction','contextual_update_alignment']
REL=re.compile(r'[=<>]|\\(?:leq?|geq?|ne?q|equiv|cong|sim|mid|parallel|perp)\b|\b(?:equals?|less than|greater than|positive|integer|distinct)\b',re.I)
IDENT=re.compile(r'(?<![A-Za-z])[A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?(?![A-Za-z])')


def cos(a,b):return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),EPS))
def spans(pattern,text):return [(m.start(),m.end()) for m in pattern.finditer(text)]
def role_masks(row):
    text = row['problem']
    start = row['original_span'][0]
    body = row['mask']
    offsets = row['offsets']
    intervals = {
        'numeric': spans(NUMBER, text),
        'relation': spans(REL, text),
        'identifier': spans(IDENT, re.sub(r'\\([A-Za-z]+)', lambda m: ' ' * len(m.group()), text)),
    }
    masks = {'body': body}
    for name, matches in intervals.items():
        masks[name] = [i for i in body if any(
            offsets[i][1] > start + a and offsets[i][0] < start + b
            for a, b in matches
        )]
    return masks
