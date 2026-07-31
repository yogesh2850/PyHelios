import sys, io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(open("/home/yogesh/PyHelios/.claude/jobs/1b0b4757/tmp/test_order.py").read())
lines = [l for l in buf.getvalue().splitlines() if ("===" in l or "order=before" in l or "has " in l or "listObjectData" in l or "NO FRUIT" in l)]
with open("/home/yogesh/PyHelios/.claude/jobs/1b0b4757/tmp/test_order_result.txt", "w") as f:
    f.write("\n".join(lines))
