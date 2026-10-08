"""Browse the photos of a folder on http://localhost:8765: left and right
arrows move between them (Home / End: first / last).

    python scripts/show_folder.py data/goals-test
    python scripts/show_folder.py data/compare"""

import glob
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hymeno.view import PORT  # noqa: E402

PAGE = b"""<html><head><title>photos</title></head>
<body style="margin:0;background:#111;color:#ddd;font:15px sans-serif">
<div style="padding:6px 10px"><b id="name"></b> <span id="where"></span>
 <span style="color:#888">&larr; &rarr; to move, Home / End</span></div>
<img id="photo" style="max-width:100%;max-height:calc(100vh - 40px)">
<script>
let names = [], i = 0;
const show = () => {
  if (!names.length) { document.getElementById('name').textContent = 'no photos'; return; }
  document.getElementById('photo').src = '/photo?name=' + encodeURIComponent(names[i]);
  document.getElementById('name').textContent = names[i];
  document.getElementById('where').textContent = (i + 1) + ' / ' + names.length;
};
fetch('/list').then(r => r.json()).then(d => { names = d; show(); });
document.onkeydown = e => {
  const before = i;
  if (e.key == 'ArrowRight') i = Math.min(names.length - 1, i + 1);
  else if (e.key == 'ArrowLeft') i = Math.max(0, i - 1);
  else if (e.key == 'Home') i = 0;
  else if (e.key == 'End') i = names.length - 1;
  else return;
  e.preventDefault();
  if (i != before) show();
};
</script></body></html>"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    folder = sys.argv[1]
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(folder, "*"))
                   if p.lower().endswith((".jpg", ".jpeg", ".png")))
    if not names:
        sys.exit("no photos in %s/" % folder)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, body, kind):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/list":
                self.reply(json.dumps(names).encode(), "application/json")
            elif url.path == "/photo":
                name = unquote(parse_qs(url.query).get("name", [""])[0])
                if name not in names:  # only the photos listed, nothing else on disk
                    self.send_response(404)
                    self.end_headers()
                    return
                with open(os.path.join(folder, name), "rb") as f:
                    self.reply(f.read(), "image/png" if name.lower().endswith(".png") else "image/jpeg")
            else:
                self.reply(PAGE, "text/html")

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("%d photos from %s on http://localhost:%d (Ctrl+C quits)" % (len(names), folder, PORT))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
