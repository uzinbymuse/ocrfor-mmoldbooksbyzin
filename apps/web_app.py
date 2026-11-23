#!/usr/bin/env python3
"""Myanmar Dhamma OCR Studio (Local Web UI: http://localhost:8080)
Features:
1. Interactive Page Inspector: Side-by-side Page Image vs v8_clean OCR text comparison
2. Full Book PDF -> Word (.docx) Batch Engine
3. Multi-Model Selector (v8_clean / v4 / v2)
4. GitHub Auto-Updater
"""
import os
import sys
import json
import io
import time
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.updater import ROOT_DIR, TESSDATA_DIR, get_active_model_name, check_and_update
import fitz
import tesserocr
from PIL import Image

SOURCE_BOOKS_DIR = os.path.join(ROOT_DIR, "data", "source_books")
UPLOAD_DIR = os.path.join(ROOT_DIR, "input")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATE = {
    "running": False,
    "progress": 0,
    "status": "Ready",
    "last_file": "",
    "model": get_active_model_name()
}

HTML_STUDIO = """<!DOCTYPE html>
<html lang="my">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Myanmar Dhamma OCR Studio</title>
<style>
:root {
  --primary: #7a1f1f;
  --primary-hover: #9b2828;
  --bg: #f8f6f2;
  --card: #ffffff;
  --border: #dfd7cc;
  --text: #1a1a1a;
  --muted: #666;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Segoe UI", "Noto Sans Myanmar", system-ui, sans-serif; background: var(--bg); color: var(--text); padding: 16px; }
.header { max-width: 1300px; margin: 0 auto 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }
.header h1 { font-size: 22px; color: var(--primary); display: flex; align-items: center; gap: 8px; }
.header .badge { background: #e8d8c8; color: var(--primary); padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: bold; }
.tabs { display: flex; gap: 8px; max-width: 1300px; margin: 0 auto 16px; }
.tab-btn { background: #e6dfd5; border: none; padding: 10px 18px; border-radius: 8px 8px 0 0; font-weight: 600; cursor: pointer; color: #444; }
.tab-btn.active { background: var(--card); color: var(--primary); border-top: 3px solid var(--primary); }
.container { max-width: 1300px; margin: 0 auto; background: var(--card); border: 1px solid var(--border); border-radius: 0 12px 12px 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
.controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; background: #faf8f5; padding: 14px; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 16px; }
select, input[type=number], input[type=file] { padding: 8px 12px; border: 1px solid var(--border); border-radius: 6px; font-size: 14px; }
button { background: var(--primary); color: #fff; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 600; transition: background .15s; font-size: 14px; }
button:hover { background: var(--primary-hover); }
button.sec { background: #4a4a4a; }
button.sec:hover { background: #333; }
button:disabled { opacity: .5; cursor: not-allowed; }
.split-view { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; min-height: 600px; }
@media (max-width: 900px) { .split-view { grid-template-columns: 1fr; } }
.pane { border: 1px solid var(--border); border-radius: 8px; display: flex; flex-direction: column; overflow: hidden; background: #fff; }
.pane-header { background: #f2ece4; padding: 10px 14px; font-weight: 600; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); }
.image-box { flex: 1; overflow: auto; padding: 16px; display: flex; justify-content: center; background: #555; }
.image-box img { max-width: 100%; height: auto; box-shadow: 0 4px 16px rgba(0,0,0,0.3); border-radius: 4px; object-fit: contain; }
.text-box { flex: 1; display: flex; flex-direction: column; }
textarea { width: 100%; flex: 1; padding: 16px; border: none; outline: none; font-family: "Noto Serif Myanmar", "Pyidaungsu", serif; font-size: 17px; line-height: 2.0; resize: none; color: #111; }
.status-bar { margin-top: 16px; padding: 10px 14px; background: #f5efe8; border-radius: 6px; font-size: 14px; display: flex; justify-content: space-between; align-items: center; }
progress { width: 100%; height: 16px; margin-top: 10px; }
.copy-btn { font-size: 12px; padding: 4px 10px; background: #2b7a4b; }
.copy-btn:hover { background: #23633d; }
</style>
</head>
<body>

<div class="header">
  <h1>📖 Myanmar Dhamma OCR Studio <span class="badge" id="activeModelBadge">Model: v8_clean</span></h1>
  <div>
    <button class="sec" onclick="checkUpdate()">🔄 Check GitHub Update</button>
  </div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('inspect')">🔍 စာမျက်နှာအလိုက် ယှဉ်တွဲစစ်ဆေးခြင်း (Page Inspector)</button>
  <button class="tab-btn" onclick="switchTab('batch')">📚 စာအုပ်အပြည့် Word (.docx) သို့ ပြောင်းခြင်း</button>
</div>

<div class="container" id="tabInspect">
  <div class="controls">
    <label><strong>စာအုပ်:</strong>
      <select id="bookSelect" onchange="onBookChange()">
        <option value="">စာအုပ်ရွေးချယ်ပါ...</option>
      </select>
    </label>

    <button onclick="prevPage()">◀ ရှေ့</button>
    <label><strong>စာမျက်နှာ:</strong>
      <input type="number" id="pageNum" value="1" min="1" style="width: 75px;" onchange="loadPage()">
      <span id="pageCount">/ 1</span>
    </label>
    <button onclick="nextPage()">နောက် ▶</button>

    <label><strong>Model:</strong>
      <select id="modelSelect">
        <option value="mya_dhamma_v8_clean" selected>mya_dhamma_v8_clean (နောက်ဆုံးထုတ်)</option>
        <option value="mya_dhamma_v4">mya_dhamma_v4</option>
        <option value="mya_dhamma_v2">mya_dhamma_v2</option>
      </select>
    </label>

    <button onclick="runPageOcr()" id="btnRunOcr">⚡ Run OCR</button>
  </div>

  <div class="split-view">
    <div class="pane">
      <div class="pane-header">
        <span>📄 မူရင်း စာမျက်နှာပုံရိပ် (Original Scan)</span>
        <span id="imgInfo" style="font-size: 12px; color: #666;">-</span>
      </div>
      <div class="image-box">
        <img id="pageImg" src="" alt="စာမျက်နှာ ရွေးချယ်ပါ">
      </div>
    </div>

    <div class="pane">
      <div class="pane-header">
        <span>📝 ဖတ်ရှုရရှိသော စာသား (OCR Text - ပြင်ဆင်နိုင်သည်)</span>
        <div>
          <button class="copy-btn" onclick="copyText()">📋 Copy Text</button>
        </div>
      </div>
      <div class="text-box">
        <textarea id="ocrOutput" placeholder="⚡ Run OCR ခလုတ်ကို နှိပ်ပါက စာသားများ ဤနေရာတွင် ပေါ်လာပါမည်..."></textarea>
      </div>
    </div>
  </div>
</div>

<div class="container" id="tabBatch" style="display:none;">
  <div class="controls">
    <form id="batchForm" onsubmit="startBatchOcr(event)" style="width:100%; display:flex; flex-wrap:wrap; gap:12px; align-items:center;">
      <label><strong>စာအုပ် ရွေးပါ:</strong>
        <select id="batchBookSelect" name="book_file"></select>
      </label>
      <label>သို့မဟုတ် PDF အသစ် Upload လုပ်ပါ:
        <input type="file" name="pdf" accept=".pdf">
      </label>
      <label>Page Offset: <input type="number" name="offset" value="0" style="width:60px"></label>
      <button type="submit" id="btnBatchGo">Word (.docx) သို့ OCR စတင်မည်</button>
    </form>
  </div>
  <progress id="batchProgress" value="0" max="100"></progress>
  <div class="status-bar" id="batchStatus">Ready</div>
  <div id="batchDl" style="margin-top: 14px;"></div>
</div>

<script>
let currentBooks = [];

function switchTab(t) {
  document.querySelectorAll('.tab-btn').forEach((b, i) => b.classList.toggle('active', (t==='inspect'&&i===0) || (t==='batch'&&i===1)));
  document.getElementById('tabInspect').style.display = t==='inspect' ? 'block' : 'none';
  document.getElementById('tabBatch').style.display = t==='batch' ? 'block' : 'none';
}

async function loadBooks() {
  try {
    const res = await fetch('/api/books');
    currentBooks = await res.json();
    const sel1 = document.getElementById('bookSelect');
    const sel2 = document.getElementById('batchBookSelect');
    sel1.innerHTML = '';
    sel2.innerHTML = '<option value="">(Upload လုပ်ထားသော ဖိုင်ကိုသုံးမည်)</option>';
    currentBooks.forEach((b, idx) => {
      const opt = `<option value="${encodeURIComponent(b.name)}" ${idx===0?'selected':''}>${b.title} (${b.pages} မျက်နှာ)</option>`;
      sel1.innerHTML += opt;
      sel2.innerHTML += opt;
    });
    if (currentBooks.length > 0) {
      onBookChange();
    }
  } catch(e) {
    console.error(e);
  }
}

function onBookChange() {
  const sel = document.getElementById('bookSelect');
  const bk = currentBooks.find(b => encodeURIComponent(b.name) === sel.value);
  if (bk) {
    document.getElementById('pageNum').max = bk.pages;
    document.getElementById('pageNum').value = 1;
    document.getElementById('pageCount').textContent = `/ ${bk.pages}`;
    loadPage();
  }
}

function prevPage() {
  const p = document.getElementById('pageNum');
  if (parseInt(p.value) > 1) { p.value = parseInt(p.value) - 1; loadPage(); }
}
function nextPage() {
  const p = document.getElementById('pageNum');
  const max = parseInt(p.max || 9999);
  if (parseInt(p.value) < max) { p.value = parseInt(p.value) + 1; loadPage(); }
}

function loadPage() {
  const sel = document.getElementById('bookSelect');
  const pno = document.getElementById('pageNum').value;
  if (!sel.value) return;
  const imgUrl = `/api/page_image?book=${sel.value}&page=${pno}`;
  document.getElementById('pageImg').src = imgUrl;
  document.getElementById('imgInfo').textContent = `Page ${pno}`;
  document.getElementById('ocrOutput').value = '';
}

async function runPageOcr() {
  const sel = document.getElementById('bookSelect');
  const pno = document.getElementById('pageNum').value;
  const mdl = document.getElementById('modelSelect').value;
  if (!sel.value) return alert('စာအုပ် ရွေးချယ်ပါ');
  
  const btn = document.getElementById('btnRunOcr');
  btn.disabled = true;
  btn.textContent = '⏳ ဖတ်ရှုနေပါသည်...';
  document.getElementById('ocrOutput').value = 'စာမျက်နှာကို mya_dhamma_v8_clean ဖြင့် ဖတ်ရှုနေပါသည်... ခေတ္တစောင့်ဆိုင်းပါ...';

  try {
    const res = await fetch('/api/page_ocr', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ book: decodeURIComponent(sel.value), page: parseInt(pno), model: mdl })
    });
    const data = await res.json();
    if (data.error) {
      document.getElementById('ocrOutput').value = 'Error: ' + data.error;
    } else {
      document.getElementById('ocrOutput').value = data.text;
    }
  } catch(e) {
    document.getElementById('ocrOutput').value = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = '⚡ Run OCR';
  }
}

function copyText() {
  const t = document.getElementById('ocrOutput');
  t.select();
  navigator.clipboard.writeText(t.value);
  alert('စာသားများကို Clipboard သို့ ကူးယူပြီးပါပြီ။');
}

async function checkUpdate() {
  alert('Checking GitHub uzinbymuse/ocrfor-mmoldbooksbyzin...');
  const res = await fetch('/api/update', {method:'POST'});
  const d = await res.json();
  alert(d.message);
}

loadBooks();
</script>
</body>
</html>
"""


class StudioHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def _json(self, data, code=200):
        b = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(url.query)

        if url.path == "/" or url.path == "/index.html":
            b = HTML_STUDIO.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return

        if url.path == "/api/books":
            books = []
            if os.path.exists(SOURCE_BOOKS_DIR):
                for f in sorted(os.listdir(SOURCE_BOOKS_DIR)):
                    if f.lower().endswith(".pdf"):
                        full_path = os.path.join(SOURCE_BOOKS_DIR, f)
                        try:
                            doc = fitz.open(full_path)
                            pages = len(doc)
                            doc.close()
                            title = f.replace("_with_Smart_TOC", "").replace("_with_TOC_and_Buttons", "").replace(".pdf", "").replace("_", " ")
                            books.append({"name": f, "title": title, "pages": pages})
                        except Exception:
                            pass
            self._json(books)
            return

        if url.path == "/api/page_image":
            book_name = qs.get("book", [""])[0]
            pno = int(qs.get("page", ["1"])[0])
            book_path = os.path.join(SOURCE_BOOKS_DIR, book_name)
            if not os.path.exists(book_path):
                self.send_error(404, "Book not found")
                return
            try:
                doc = fitz.open(book_path)
                pno = max(1, min(pno, len(doc)))
                page = doc[pno - 1]
                scale = min(2.0, 1800.0 / page.rect.height)
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                img_data = pix.tobytes("jpeg")
                doc.close()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(img_data)))
                self.end_headers()
                self.wfile.write(img_data)
                return
            except Exception as e:
                self.send_error(500, str(e))
                return

        if url.path == "/api/status":
            self._json(STATE)
            return

        self.send_error(404)

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)

        if url.path == "/api/page_ocr":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
            book_name = data.get("book", "")
            pno = int(data.get("page", 1))
            model_name = data.get("model", "mya_dhamma_v8_clean")

            book_path = os.path.join(SOURCE_BOOKS_DIR, book_name)
            if not os.path.exists(book_path):
                self._json({"error": "Book file not found"}, 404)
                return

            try:
                doc = fitz.open(book_path)
                pno = max(1, min(pno, len(doc)))
                page = doc[pno - 1]
                scale = min(2.5, 2400.0 / page.rect.height)
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                img = Image.open(io.BytesIO(pix.tobytes()))
                doc.close()

                api = tesserocr.PyTessBaseAPI(path=TESSDATA_DIR, lang=model_name)
                api.SetImage(img)
                text = api.GetUTF8Text()
                self._json({"text": text, "page": pno, "model": model_name})
            except Exception as e:
                self._json({"error": str(e)}, 500)
            return

        if url.path == "/api/update":
            ok, info = check_and_update()
            msg = f"Update check complete. Current: {info.get('model_name')}" if ok else "Already up to date."
            self._json({"message": msg})
            return

        self.send_error(404)


def run(port=8080):
    server = HTTPServer(("127.0.0.1", port), StudioHandler)
    print(f"Myanmar Dhamma OCR Studio running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    run()
