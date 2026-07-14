from tokenizers import Tokenizer
t = Tokenizer.from_file('/home/raffaele/Progetti/FaStar/tokenizer.json')
out = open('/home/raffaele/Progetti/FaStar/logs/tok_ids2.txt', 'w')
ids = [198, 846, 9419, 74455, 248043, 248044, 248045, 248046, 248047, 248060, 248065, 248066, 248067, 248068, 248069]
for tid in ids:
    s = t.decode([tid], skip_special_tokens=False)
    out.write('%d -> bytes=%r str=%r\n' % (tid, s.encode('utf-8'), s))
out.close()
print('done')