import gguf
r = gguf.GGUFReader('/home/raffaele/Progetti/FaStar/downloads/Qwopus3.6-27B-Coder-MTP-Q4_K_M.gguf')
parts = r.fields['tokenizer.chat_template'].parts
raw = bytes(parts[-1]).decode('utf-8', 'replace')
open('/home/raffaele/Progetti/FaStar/logs/ct_full.txt', 'w').write(raw)
print('template len', len(raw))
for k in r.fields:
    if 'eos' in k or 'bos' in k:
        print('field', k)