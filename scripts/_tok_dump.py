import gguf
r = gguf.GGUFReader('/home/raffaele/Progetti/FaStar/downloads/Qwopus3.6-27B-Coder-MTP-Q4_K_M.gguf')
toks = r.tokens
out = open('/home/raffaele/Progetti/FaStar/logs/tok_ids.txt', 'w')
out.write('r.tokens count %d\n' % len(toks))
for tid in [198, 846, 9419, 74455, 12195, 3710, 2523, 9053, 1358, 271, 6164, 71093, 13151,
           151643, 151644, 151645, 248043, 248044, 248045, 248046, 248047, 248048, 248049,
           248050, 248060, 248065, 248066, 248067, 248068, 248069]:
    if 0 <= tid < len(toks):
        out.write('%d %r\n' % (tid, toks[tid]))
out.close()
print('done')