p='/opt/indian-legal-ai/reingest_spec/scripts/run_batch_loop.sh'
s=open(p).read()
# Inline env on the ingest_v2 invocation so DENSE_ONLY definitely reaches python
s=s.replace(
'  nohup python3 reingest_spec/ingest_v2.py --phase all --collection cbic_v2 --doc-ids "$DOCIDS" >> $LOG 2>&1 &',
'  nohup env DENSE_ONLY=1 RADV_DEBUG=nodcc GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 EMBED_GPUS=4,5,6 python3 reingest_spec/ingest_v2.py --phase all --collection cbic_v2 --doc-ids "$DOCIDS" >> $LOG 2>&1 &')
open(p,'w').write(s)
print('patched, contains DENSE_ONLY env line:', 'env DENSE_ONLY=1' in s)
