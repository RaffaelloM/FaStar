from tokenizers import Tokenizer
t = Tokenizer.from_file('/home/raffaele/Progetti/FaStar/tokenizer.json')
out = open('/home/raffaele/Progetti/FaStar/logs/tok_ids.txt', 'w')
ids = [198, 846, 9419, 74455, 12195, 3710, 2523, 9053, 1358, 271, 6164, 71093, 13151,
       151643, 151644, 151645, 248043, 248044, 248045, 248046, 248047, 248048, 248049,
       248050, 248060, 248065, 248066, 248067, 248068, 248069]
for tid in ids:
    try:
        out.write('%d %r\n' % (tid, t.decode([tid])))
    except Exception as e:
        out.write('%d ERR %s\n' % (tid, e))
out.close()
print('done')