import json
from decimal import Decimal

def format_number_full(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        decimal_value = Decimal(str(value))
        formatted = format(decimal_value, 'f')
        if '.' in formatted:
            formatted = formatted.rstrip('0').rstrip('.')
        return formatted
    return str(value)

def parse_scientific_to_float(value_str):
    try:
        return float(value_str)
    except:
        return 0.0

class CustomJSONEncoder(json.JSONEncoder):
    """A custom JSON encoder that formats floating-point numbers in full precision.

    This encoder overrides the default JSON encoding to ensure that float values
    are represented with full precision using the format_number_full function.
    It also handles scientific notation in the encoded output by converting it
    back to full precision.
    """
    def encode(self, obj):
        if isinstance(obj, float):
            return format_number_full(obj)
        return super().encode(obj)
    def iterencode(self, obj, _one_shot=False):
        for chunk in super().iterencode(obj, _one_shot):
            if 'e+' in chunk.lower() or 'e-' in chunk.lower():
                import re
                pattern = r'[-]?\d+\.?\d*[eE][+-]?\d+'
                def replacer(match):
                    return format_number_full(float(match.group()))
                chunk = re.sub(pattern, replacer, chunk)
            yield chunk
