p='/opt/indian-legal-ai/reingest_spec/scripts/run_batch_loop.sh'
s=open(p).read()
s=s.replace(
'''  local DOCIDS=$(cat $CSV)
   tr , "\\n" | tr ',' '\\n' | wc -l)
  log "  built $NDOCS doc_ids for batch $N"''',
'''  local DOCIDS=$(cat $CSV)
  local NDOCS=$(echo "$DOCIDS" | tr ',' '\\n' | grep -c .)
  log "  built $NDOCS doc_ids for batch $N"''')
open(p,'w').write(s)
print('patched bytes='+str(len(s)))
