p='/opt/indian-legal-ai/reingest_spec/scripts/run_batch_loop.sh'
s=open(p).read()
PY='/opt/indian-legal-ai/rag/cbic_rag/venv/bin/python3'
# Pin python to venv that has fitz/pymupdf installed
s=s.replace(
'  nohup env DENSE_ONLY=1 RADV_DEBUG=nodcc GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 EMBED_GPUS=4,5,6 python3 reingest_spec/ingest_v2.py',
f'  nohup env DENSE_ONLY=1 RADV_DEBUG=nodcc GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 EMBED_GPUS=4,5,6 {PY} reingest_spec/ingest_v2.py')
s=s.replace(
'python3 reingest_spec/build_batch.py',
f'{PY} reingest_spec/build_batch.py')
# Gate evaluators too (they likely import qdrant_client + the rag modules)
s=s.replace('python3 reingest_spec/evaluators/', f'{PY} reingest_spec/evaluators/')
open(p,'w').write(s)
print('patched.', f'venv refs: {s.count(PY)}')
