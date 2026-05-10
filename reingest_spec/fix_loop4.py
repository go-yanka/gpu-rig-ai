p='/opt/indian-legal-ai/reingest_spec/scripts/run_batch_loop.sh'
s=open(p).read()
VENV='/opt/indian-legal-ai/rag/cbic_rag/venv/bin/python3'
SYS='/usr/bin/python3'
PYP='/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag'
# Revert venv python to /usr/bin/python3 + add PYTHONPATH
s=s.replace(VENV, SYS)
# Inject PYTHONPATH into the env line
s=s.replace(
'env DENSE_ONLY=1 RADV_DEBUG=nodcc GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 EMBED_GPUS=4,5,6 /usr/bin/python3 reingest_spec/ingest_v2.py',
f'env DENSE_ONLY=1 RADV_DEBUG=nodcc GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 EMBED_GPUS=4,5,6 PYTHONPATH={PYP} /usr/bin/python3 reingest_spec/ingest_v2.py')
# Also for build_batch and gate evaluators (they likely import same stack)
s=s.replace(
'  /usr/bin/python3 reingest_spec/build_batch.py',
f'  PYTHONPATH={PYP} /usr/bin/python3 reingest_spec/build_batch.py')
s=s.replace(
'  /usr/bin/python3 reingest_spec/evaluators/',
f'  PYTHONPATH={PYP} /usr/bin/python3 reingest_spec/evaluators/')
open(p,'w').write(s)
print('python:', s.count(SYS), '· PYTHONPATH refs:', s.count('PYTHONPATH'))
