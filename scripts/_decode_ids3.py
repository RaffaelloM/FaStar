from tokenizers import Tokenizer
t = Tokenizer.from_file('/home/raffaele/Progetti/FaStar/tokenizer.json')
out = open('/home/raffaele/Progetti/FaStar/logs/tok_ids3.txt', 'w')
ids = [198, 846, 9419, 74455, 248044, 248045, 248046, 248060, 248065, 248066, 248067, 248068, 248069]
for tid in ids:
    s = t.decode([tid], skip_special_tokens=False)
    # ascii-escape everything so angle brackets are visible
    esc = s.encode('ascii', 'backslashreplace').decode('ascii')
    out.write('%d -> %r\n' % (tid, esc))
out.close()
print('done')