import pathlib
p = pathlib.Path('D:/tmp_chunker_v2.py')
txt = p.read_text(encoding='utf-8')

NL = chr(10)
BT3 = chr(0x60) * 3
BS = chr(92)

broken = (
    "    m = re.search(r'" + BT3 + "(?:json)?" + BS + "s*" + NL +
    "?(.*?)" + NL +
    "?" + BS + "s*" + BT3 + "', raw, re.DOTALL)" + NL
)
fixed = (
    "    m = re.search(r'" + BT3 + "(?:json)?" + BS + "s*" + BS + "n?(.*?)" + BS + "n?" + BS + "s*" + BT3 + "', raw, re.DOTALL)" + NL
)

print("broken len:", len(broken), "in text:", broken in txt)
print("fixed len:", len(fixed))

assert broken in txt, "broken block not found"
new_txt = txt.replace(broken, fixed)
p.write_text(new_txt, encoding='utf-8')
print("bytes before:", len(txt), "after:", len(new_txt))

import ast
ast.parse(p.read_text(encoding='utf-8'))
print("syntax ok")
