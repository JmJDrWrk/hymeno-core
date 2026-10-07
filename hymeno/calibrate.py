"""Camera calibration by clicks: `python -m hymeno calibrate`.

Two tapes on the floor, one robot-width apart and 1 m long from the front
legs, with screws at 25 and 50 cm. On a photo of them, in the browser, the
user clicks two points on each tape and the base of each screw. From those:
the horizon and centre (where the tapes meet), cm per row (the screws) and the
robot's width (the tapes). Saved to floor.CALIBRATION, which floor.py loads."""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCREWS_CM = (25, 50)
DISAGREE = 0.15  # the two screws giving cm conversions this far apart: a click is probably off


def solve(points, width, height):
    """points: 6 (x, y) pixels: left tape near and far, right tape near and
    far, the 25 cm screw's base, the 50 cm screw's base. Returns the four
    numbers of floor.py (horizon, centre, cm_k, half_width) and warnings.
    Raises ValueError when the clicks make no sense."""
    if len(points) != 6:
        raise ValueError("6 clicks are needed")
    (x1, y1), (x2, y2), (x3, y3), (x4, y4), screw25, screw50 = [(x / width, y / height) for x, y in points]
    if abs(y1 - y2) < 0.01 or abs(y3 - y4) < 0.01:
        raise ValueError("the two points of a tape need to be at different heights")
    # Each tape as x = x0 + slope * y; they meet at the horizon.
    left = (x2 - x1) / (y2 - y1)
    right = (x4 - x3) / (y4 - y3)
    if abs(left - right) < 1e-3:
        raise ValueError("the tapes look parallel: click two points far apart on each")
    horizon = (x3 - right * y3 - (x1 - left * y1)) / (left - right)
    centre = x1 + left * (horizon - y1)
    half_width = (right - left) / 2
    if half_width <= 0:
        raise ValueError("left and right tapes are swapped")
    ks = []
    for (_, y), cm in zip((screw25, screw50), SCREWS_CM):
        if y <= horizon:
            raise ValueError("a screw is above the horizon")
        ks.append(cm * (y - horizon))
    cm_k = sum(ks) / len(ks)
    warnings = []
    if abs(ks[0] - ks[1]) / cm_k > DISAGREE:
        warnings.append("the screws disagree by %.0f%%: check their clicks" % (100 * abs(ks[0] - ks[1]) / cm_k))
    if not 0.2 < horizon < 0.8:
        warnings.append("the horizon is at %.0f%% of the height: is the camera tilted a lot?" % (100 * horizon))
    return {"horizon": horizon, "centre": centre, "cm_k": cm_k, "half_width": half_width,
            "screws_cm": [round(cm_k / (y - horizon), 1) for _, y in (screw25, screw50)],
            "warnings": warnings}


PAGE = """<html><head><title>hymeno calibrate</title></head>
<body style="margin:0;background:#111;color:#eee;font:15px/1.4 sans-serif">
<canvas id="c" style="width:100%;max-width:960px;cursor:crosshair"></canvas>
<div style="padding:8px 14px;max-width:960px">
<div style="margin:10px 0"><b id="step" style="color:#ff0"></b>
 <button id="undo">Undo</button> <button id="save" disabled>Save</button>
 <div id="result" style="margin-top:6px;color:#9cf"></div></div>

<h2 style="margin:4px 0">Camera calibration</h2>
<p style="margin:4px 0;color:#bbb">Tells the robot how its camera sees the floor: where the horizon is, how many
cm each row of the photo is, and how wide the robot looks. Redo it whenever the camera is moved or tilted.</p>
<details><summary><b>1. Before you start: set up the floor</b> (click to read)</summary><ol>
<li>Two strips of tape on the floor, <b>parallel</b>, <b>1 m long</b>, starting at the robot's <b>front legs</b>.
They must be exactly <b>as far apart as the robot is wide</b>.</li>
<li>The robot between them, <b>centred</b>, looking straight along them.</li>
<li>A screw standing on the right tape at <b>25 cm</b> from the front legs, and another at <b>50 cm</b>.</li>
<li>The camera at the resolution you normally use (<b>640x480</b>), and <b>/video closed</b> in the browser.</li>
<li>The photo below was taken when you ran <code>python -m hymeno calibrate</code> (saved as
data/calibration.jpg). If the tapes or screws don't show well: Ctrl+C in the terminal, fix it and run it again.</li>
</ol></details>
<p style="margin:8px 0 2px"><b>2. Click these 6 points on the photo</b>, in this order
(tip: zoom the browser with Ctrl and + to be precise; <b>Undo</b> removes the last click):</p>
<ol id="steps" style="margin:2px 0"></ol>
<p style="margin:8px 0 2px"><b>3. Check the drawing, then Save</b></p>
<ul style="margin:2px 0">
<li>The <span style="color:#0ff">cyan lines</span> (the robot's path) must lie <b>on the tapes</b>.</li>
<li>The <span style="color:#f66">red 25 cm and 50 cm lines</span> must cross <b>the base of their screws</b>.</li>
<li>"Screws read as" should say about 25 and 50 cm. A WARNING means a click is probably off: Undo and redo it.</li>
<li><b>Save</b> writes data/calibration.json. Then start floor, explore or replay again to use it.
To go back to the numbers in the code, delete that file.</li>
</ul>

</div>
<script>
const STEPS = ["Left tape, near: the middle of the left tape, low in the photo (close to the robot)",
  "Left tape, far: the middle of the left tape, as high up as you can still see it clearly",
  "Right tape, near: the middle of the right tape, low in the photo",
  "Right tape, far: the middle of the right tape, as high up as you can still see it clearly",
  "25 cm screw: where it touches the floor (its base, not its head)",
  "50 cm screw: where it touches the floor (its base, not its head)"];
const canvas = document.getElementById('c'), pen = canvas.getContext('2d');
const photo = new Image();
let points = [], solved = null;
photo.onload = () => { canvas.width = photo.width; canvas.height = photo.height; redraw(); };
photo.src = '/photo.jpg';
function line(x1, y1, x2, y2, colour, wide) {
  pen.strokeStyle = colour; pen.lineWidth = wide || 2;
  pen.beginPath(); pen.moveTo(x1, y1); pen.lineTo(x2, y2); pen.stroke();
}
function label(text, x, y) {
  pen.font = '14px sans-serif'; const w = pen.measureText(text).width;
  pen.fillStyle = '#000'; pen.fillRect(x - 3, y - 14, w + 6, 19);
  pen.fillStyle = '#fff'; pen.fillText(text, x, y);
}
function redraw() {
  const W = canvas.width, H = canvas.height;
  pen.drawImage(photo, 0, 0);
  points.forEach((p, i) => {
    pen.fillStyle = i < 4 ? '#ff0' : '#f0f';
    pen.beginPath(); pen.arc(p[0], p[1], 5, 0, 7); pen.fill();
  });
  if (solved) {
    const h = solved.horizon, c = solved.centre, k = solved.cm_k, hw = solved.half_width;
    line(0, h * H, W, h * H, '#888', 1);
    // The robot's path (should lie on the tapes) and the distance lines (should cross the screws).
    for (const side of [-1, 1]) line(c * W, h * H, (c + side * hw * (1 - h)) * W, H, '#0ff', 2);
    for (const cm of [25, 50, 100]) {
      const y = (h + k / cm) * H;
      line(0, y, W, y, '#f33', 1); label(cm + ' cm', 6, y - 4);
    }
  }
  document.getElementById('steps').innerHTML = STEPS.map((text, i) =>
    '<li style="' + (i == points.length ? 'color:#ff0;font-weight:bold' : i < points.length ? 'color:#777' : '') +
    '">' + (i < points.length ? '&#10003; ' : '') + text + '</li>').join('');
  document.getElementById('step').textContent = points.length < 6 ?
    'Now click: ' + STEPS[points.length].split(':')[0] :
    'Done clicking: check the drawing (cyan on the tapes, red lines on the screw bases), then Save.';
  document.getElementById('save').disabled = !solved;
}
canvas.onclick = async e => {
  if (points.length >= 6) return;
  const r = canvas.getBoundingClientRect();
  points.push([(e.clientX - r.left) * canvas.width / r.width, (e.clientY - r.top) * canvas.height / r.height]);
  if (points.length == 6) {
    const answer = await fetch('/solve', {method: 'POST',
      body: JSON.stringify({points: points, width: canvas.width, height: canvas.height})});
    const data = await answer.json();
    const result = document.getElementById('result');
    if (data.error) { result.textContent = 'Error: ' + data.error + ' (undo and click again)'; }
    else {
      solved = data;
      result.textContent = 'horizon ' + data.horizon.toFixed(3) + ', centre ' + data.centre.toFixed(3) +
        ', cm_k ' + data.cm_k.toFixed(2) + ', half width ' + data.half_width.toFixed(2) +
        ' | screws read as ' + data.screws_cm.join(' and ') + ' cm' +
        (data.warnings.length ? ' | WARNING: ' + data.warnings.join('; ') : '');
    }
  }
  redraw();
};
document.getElementById('undo').onclick = () => {
  points.pop(); solved = null; document.getElementById('result').textContent = ''; redraw();
};
document.getElementById('save').onclick = async () => {
  const answer = await fetch('/save', {method: 'POST', body: JSON.stringify(solved)});
  document.getElementById('result').textContent = await answer.text();
};
</script></body></html>"""


def run(jpeg, photo_path, say, port):
    """Serves the clicking page until the calibration is saved (or Ctrl+C)."""
    from . import floor

    saved = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, body, kind):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/photo.jpg":
                self.reply(200, jpeg, "image/jpeg")
            else:
                self.reply(200, PAGE.encode(), "text/html")

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/solve":
                try:
                    answer = solve(data["points"], data["width"], data["height"])
                except ValueError as e:
                    answer = {"error": str(e)}
                self.reply(200, json.dumps(answer).encode(), "application/json")
            elif self.path == "/save":
                keep = {k: data[k] for k in ("horizon", "centre", "cm_k", "half_width")}
                keep.update(measured=time.strftime("%Y-%m-%d %H:%M"), photo=photo_path)
                os.makedirs(os.path.dirname(floor.CALIBRATION), exist_ok=True)
                with open(floor.CALIBRATION, "w") as f:
                    json.dump(keep, f, indent=2)
                self.reply(200, ("Saved to %s. Restart hymeno to use it." % floor.CALIBRATION).encode(), "text/plain")
                say("calibration saved to %s: %s" % (floor.CALIBRATION, json.dumps(keep)))
                saved.set()

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    say("click the points on http://localhost:%d (Ctrl+C quits without saving)" % port)
    try:
        while not saved.wait(0.5):
            pass
        time.sleep(1)  # let the page get its answer
    except KeyboardInterrupt:
        say("stopped, nothing saved")
    server.shutdown()
    return saved.is_set()
