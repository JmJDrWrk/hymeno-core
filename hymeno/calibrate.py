"""Camera calibration by clicks: `python -m hymeno calibrate`.

Two tapes on the floor, one robot-width apart and 1 m long from the front
legs, with screws at 25 and 50 cm. On a photo of them, in the browser, the
user clicks two points on each tape and the base of each screw. From those:
the horizon and centre (where the tapes meet), cm per row (the screws) and the
robot's width (the tapes). Also the leg corners: the bottom corners where the
robot's own legs show, set with two sliders. Each part is saved on its own to
floor.CALIBRATION, which floor.py loads."""

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
 <button id="undo">Undo</button> <button id="save" disabled>Save geometry</button>
 <div id="result" style="margin-top:6px;color:#9cf"></div></div>
<div style="margin:10px 0;padding:6px 8px;border:1px solid #444">
 <b style="color:#fd0">Leg corners</b>
 <label><input type="checkbox" id="legs"> ignore them</label>
 &nbsp; in <input type="range" id="legin" min="0" max="0.5" step="0.01">
 &nbsp; up <input type="range" id="legup" min="0" max="0.8" step="0.01">
 <span id="legvalues"></span> <button id="savelegs">Save leg corners</button>
 <div id="legresult" style="margin-top:6px;color:#9cf"></div></div>

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
<li><b>Save geometry</b> writes it to data/calibration.json. Then start floor, explore or replay again
to use it. To go back to the numbers in the code, delete that file.</li>
</ul>
<p style="margin:8px 0 2px"><b>4. Leg corners</b> (optional, can be done on its own)</p>
<ul style="margin:2px 0">
<li>The robot's own legs show in the bottom corners of the photo. Inside the
<span style="color:#fd0">yellow corners</span> nothing counts as an obstacle.</li>
<li>Move <b>in</b> (how far they reach along the bottom) and <b>up</b> (how far up the sides) until the
yellow covers the legs with a little margin. Untick "ignore them" if the legs never show.</li>
<li>The legs move while walking, so use a photo where they show the most. No tapes are needed for this
part: <code>python -m hymeno calibrate data/runs/&lt;date-time&gt;/00123.jpg</code></li>
<li><b>Save leg corners</b> saves only them; the geometry already saved is kept (and the other way round).</li>
<li>When you are done, Ctrl+C in the terminal.</li>
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
const legs = document.getElementById('legs'), legIn = document.getElementById('legin'),
  legUp = document.getElementById('legup');
fetch('/legs').then(r => r.json()).then(d => {
  legs.checked = d.legs; legIn.value = d.leg_in; legUp.value = d.leg_up; redraw();
});
[legs, legIn, legUp].forEach(control => control.oninput = redraw);
function triangle(points, colour) {
  pen.fillStyle = colour; pen.beginPath(); pen.moveTo(...points[0]);
  points.slice(1).forEach(p => pen.lineTo(...p)); pen.closePath(); pen.fill();
}
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
  const i = +legIn.value, u = +legUp.value;
  document.getElementById('legvalues').textContent = 'in ' + i.toFixed(2) + ', up ' + u.toFixed(2);
  if (legs.checked) {
    triangle([[0, H], [i * W, H], [0, H * (1 - u)]], 'rgba(255, 220, 0, 0.45)');
    triangle([[W, H], [W - i * W, H], [W, H * (1 - u)]], 'rgba(255, 220, 0, 0.45)');
  }
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
document.getElementById('savelegs').onclick = async () => {
  const answer = await fetch('/save-legs', {method: 'POST',
    body: JSON.stringify({legs: legs.checked, leg_in: +legIn.value, leg_up: +legUp.value})});
  document.getElementById('legresult').textContent = await answer.text();
};
document.getElementById('save').onclick = async () => {
  const answer = await fetch('/save', {method: 'POST', body: JSON.stringify(solved)});
  document.getElementById('result').textContent = await answer.text();
};
</script></body></html>"""


def save(path, update):
    """Merges update into the calibration file, keeping what is not updated."""
    kept = {}
    if os.path.exists(path):
        with open(path) as f:
            kept = json.load(f)
    kept.update(update)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(kept, f, indent=2)


def run(jpeg, photo_path, say, port):
    """Serves the calibration page until Ctrl+C. True if anything was saved."""
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
            elif self.path == "/legs":
                now = {"legs": floor.LEGS, "leg_in": floor.LEG_IN, "leg_up": floor.LEG_UP}
                self.reply(200, json.dumps(now).encode(), "application/json")
            else:
                self.reply(200, PAGE.encode(), "text/html")

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            when = time.strftime("%Y-%m-%d %H:%M")
            if self.path == "/solve":
                try:
                    answer = solve(data["points"], data["width"], data["height"])
                except ValueError as e:
                    answer = {"error": str(e)}
                self.reply(200, json.dumps(answer).encode(), "application/json")
                return
            if self.path == "/save":
                update = {k: data[k] for k in ("horizon", "centre", "cm_k", "half_width")}
                update.update(measured=when, photo=photo_path)
            elif self.path == "/save-legs":
                update = {"legs": bool(data["legs"]), "leg_in": float(data["leg_in"]),
                          "leg_up": float(data["leg_up"]), "legs_measured": when, "legs_photo": photo_path}
            else:
                self.reply(404, b"", "text/plain")
                return
            save(floor.CALIBRATION, update)
            self.reply(200, ("Saved to %s. Restart floor, explore or replay to use it." % floor.CALIBRATION).encode(),
                       "text/plain")
            say("saved to %s: %s" % (floor.CALIBRATION, json.dumps(update)))
            saved.set()

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    say("calibrate on http://localhost:%d; Ctrl+C when you are done" % port)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        say("done" if saved.is_set() else "stopped, nothing saved")
    server.shutdown()
    return saved.is_set()
