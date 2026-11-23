#!/usr/bin/env python3
"""System 2: Zero-dependency Local Web UI (http://localhost:8080) with GitHub Model Auto-Updater."""
import os, sys, json, cgi, threading, webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.updater import ROOT_DIR, TESSDATA_DIR, get_active_model_name, check_and_update
from core.ocr_engine import convert_pdf_to_docx

UPLOAD_DIR = os.path.join(ROOT_DIR, "input")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATE = {"running": False, "progress": 0, "status": "Ready", "last_file": "", "model": get_active_model_name()}

HTML_PAGE = """<!DOCTYPE html>
<html lang="my"><head><meta charset="utf-8"><title>Myanmar Dhamma OCR Studio</title>
<style>
body{font-family:system-ui,sans-serif;max-width:720px;margin:32px auto;padding:0 16px;background:#faf7f4;color:#1a1a1a}
.card{background:#fff;border:1px solid #e2d8cc;border-radius:12px;padding:24px;box-shadow:0 2px 8px rgba(0,0,0,.04)}
h1{color:#7a1f1f;margin-top:0;font-size:22px}
.row{margin:12px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button{background:#7a1f1f;color:#fff;border:none;padding:10px 16px;border-radius:8px;cursor:pointer;font-weight:600}
button.sec{background:#4a4a4a}
button:disabled{opacity:.5;cursor:not-allowed}
progress{width:100%;height:18px;margin-top:10px}
#status{background:#f5efe8;padding:10px 12px;border-radius:8px;font-size:14px;margin-top:12px}
a.dl{display:inline-block;margin-top:10px;color:#7a1f1f;font-weight:700}
</style></head><body>
<div class="card">
<h1>Myanmar Dhamma PDF → Word OCR</h1>
<div class="row"><span>Active Model: <strong id="model">...</strong></span>
<button class="sec" onclick="checkUpdate()">Check GitHub Update</button></div>
<form id="f" onsubmit="startOcr(event)">
<div class="row"><input type="file" name="pdf" accept=".pdf" required></div>
<div class="row"><label>Page Offset: <input type="number" name="offset" value="0" style="width:64px"></label>
<label><input type="checkbox" name="crop" checked> Crop Watermarks</label></div>
<button type="submit" id="go">Start OCR → Word (.docx)</button>
</form>
<progress id="p" value="0" max="100"></progress>
<div id="status">Ready</div>
<div id="dl"></div>
</div>
<script>
async function poll(){
  const r = await fetch("/api/status"); const s = await r.json();
  document.getElementById("model").textContent = s.model;
  document.getElementById("p").value = s.progress;
  document.getElementById("status").textContent = s.status;
  document.getElementById("go").disabled = s.running;
  document.getElementById("dl").innerHTML = s.last_file ? `<a class="dl" href="/download/${encodeURIComponent(s.last_file)}">Download ${s.last_file}</a>` : "";
}
async function checkUpdate(){
  document.getElementById("status").textContent = "Checking GitHub for model update...";
  const r = await fetch("/api/update", {method:"POST"}); const d = await r.json();
  alert(d.message); poll();
}
async function startOcr(e){
  e.preventDefault();
  const fd = new FormData(e.target);
  await fetch("/api/ocr", {method:"POST", body:fd});
  poll();
}
setInterval(poll, 1500); poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            b = HTML_PAGE.encode("utf-8")
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        elif self.path == "/api/status":
            self._json(STATE)
        elif self.path.startswith("/download/"):
            fn = os.path.basename(self.path[len("/download/"):])
            fp = os.path.join(OUTPUT_DIR, fn)
            if os.path.exists(fp):
                with open(fp, "rb") as f: data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                self.send_header("Content-Disposition", f"attachment; filename={fn}")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/update":
            logs = []
            updated, info = check_and_update(logs.append)
            STATE["model"] = get_active_model_name()
            STATE["status"] = logs[-1] if logs else "Checked"
            self._json({"updated": updated, "message": "\n".join(logs), "model": STATE["model"]})
        elif self.path == "/api/ocr":
            if STATE["running"]:
                return self._json({"error": "Already running"}, 409)
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers["Content-Type"]})
            fitem = form["pdf"]
            fn = os.path.basename(fitem.filename)
            pdf_path = os.path.join(UPLOAD_DIR, fn)
            with open(pdf_path, "wb") as f: f.write(fitem.file.read())
            offset = int(form.getvalue("offset", "0") or 0)
            crop = "crop" in form
            out_name = os.path.splitext(fn)[0] + "_OCR.docx"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            def _job():
                STATE.update({"running": True, "progress": 0, "last_file": "", "status": f"Running OCR on {fn}..."})
                try:
                    def _prog(done, total, pno):
                        STATE["progress"] = int(done * 100 / total)
                        STATE["status"] = f"OCR {done}/{total} pages ({STATE['progress']}%)"
                    res = convert_pdf_to_docx(pdf_path, out_path, TESSDATA_DIR, lang=STATE["model"], page_offset=offset, crop_hf=crop, progress_cb=_prog)
                    STATE.update({"progress": 100, "last_file": out_name, "status": f"Completed in {res['elapsed_sec']}s!"})
                except Exception as e:
                    STATE["status"] = f"Error: {e}"
                finally:
                    STATE["running"] = False

            threading.Thread(target=_job, daemon=True).start()
            self._json({"started": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    print(f"Myanmar Dhamma OCR Web UI running at http://localhost:{port}")
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
