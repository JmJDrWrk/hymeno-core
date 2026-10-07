"""The latest processed photo, live on http://localhost:PORT, to watch the
robot think from a browser. The page asks for each new photo and shows it as
soon as it arrives (MJPEG video makes browsers show one photo late). Viewers
are served from their own threads, so they never slow the brain down."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

PORT = 8765
PAGE = b"""<html><head><title>hymeno</title></head>
<body style='margin:0;background:#000'>
<img id='photo' style='width:100%;max-width:960px'>
<script>
const photo = document.getElementById('photo');
let number = 0;
async function follow() {
  while (true) {
    try {
      const r = await fetch('/next?after=' + number, {cache: 'no-store'});
      if (r.status === 200) {
        number = +r.headers.get('X-Number');
        const old = photo.src;
        photo.src = URL.createObjectURL(await r.blob());
        if (old.startsWith('blob:')) URL.revokeObjectURL(old);
      }
    } catch (e) {
      await new Promise(done => setTimeout(done, 1000));
    }
  }
}
follow();
</script></body></html>"""
WAIT_S = 25  # a request for the next photo is answered empty after this long


class View:
    def __init__(self, port=PORT):
        self._jpeg, self._number = None, 0
        self._new = threading.Condition()
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                url = urlsplit(self.path)
                try:
                    if url.path != "/next":
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html")
                        self.end_headers()
                        self.wfile.write(PAGE)
                        return
                    after = int(parse_qs(url.query).get("after", ["0"])[0])
                    jpeg, number = view.wait(after, WAIT_S)
                    if jpeg is None:
                        self.send_response(204)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Number", str(number))
                    self.end_headers()
                    self.wfile.write(jpeg)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def show(self, jpeg):
        """Sends this JPEG to everyone watching."""
        with self._new:
            self._jpeg, self._number = jpeg, self._number + 1
            self._new.notify_all()

    def wait(self, after, timeout=None):
        """(JPEG, number) of the first photo newer than number `after`;
        (None, after) if none comes within timeout seconds."""
        with self._new:
            if not self._new.wait_for(lambda: self._number > after, timeout):
                return None, after
            return self._jpeg, self._number
