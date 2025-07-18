#!/usr/bin/env python3
import re

# JavaScript dosyasını oku
with open('static/js/prediction-popup-template-v2.js', 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')

# Potansiyel syntax hatalarını ara
issues = []

# Tek satırlık syntax kontrolü
for i, line in enumerate(lines):
    # Garip karakterler kontrolü
    if ':' in line and not any(x in line for x in ['//', 'http', 'function', 'case', '?', '{', '}']):
        # String içinde olmadığından emin ol
        if line.count('"') % 2 == 0 and line.count("'") % 2 == 0:
            # CSS style değilse
            if 'style=' not in line and 'font-' not in line and 'margin' not in line:
                issues.append(f"Line {i+1}: Suspicious colon - {line.strip()}")

# Parantez dengesizliği kontrolü
open_braces = 0
open_parens = 0
open_brackets = 0

for i, line in enumerate(lines):
    # Yorumları ve string'leri göz ardı et (basit kontrol)
    if '//' in line:
        line = line[:line.index('//')]
    
    open_braces += line.count('{') - line.count('}')
    open_parens += line.count('(') - line.count(')')
    open_brackets += line.count('[') - line.count(']')
    
    if open_braces < 0 or open_parens < 0 or open_brackets < 0:
        issues.append(f"Line {i+1}: Unmatched closing bracket/paren/brace")

# Sonuçları yazdır
print("JavaScript Syntax Check Results:")
print("=" * 50)

if issues:
    for issue in issues[:20]:  # İlk 20 sorunu göster
        print(issue)
else:
    print("No obvious syntax issues found")

print(f"\nFinal balance - Braces: {open_braces}, Parens: {open_parens}, Brackets: {open_brackets}")