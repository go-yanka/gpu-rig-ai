p='/opt/indian-legal-ai/reingest_spec/scripts/gate_preflight.sh'
s=open(p).read()
# Replace the awk regex to anchor on python-start AND filter out bash launchers
old="RUNNING=$(ps -eo pid,cmd | awk '/python[0-9.]* .*gate_g[1-5][a-z_]*\\.py/ && !/awk/ {print}' || true)"
new="RUNNING=$(ps -eo pid,cmd | awk '/^[ \\t]*[0-9]+ +(\\\\/usr\\\\/bin\\\\/)?python[0-9.]* .*gate_g[1-5][a-z_]*\\\\.py/ && !/awk/ && !/bash/ {print}' || true)"
s=s.replace(old, new)
open(p,'w').write(s)
print('patched, contains anchored regex:', 'python[0-9.]*' in s and '!/bash/' in s)
