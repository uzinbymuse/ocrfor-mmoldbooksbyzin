#!/usr/bin/env python3
"""System 1: Lightweight Desktop GUI (Tkinter) for PDF -> OCR -> Word (.docx) with GitHub Auto-Update."""
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.updater import TESSDATA_DIR, get_active_model_name, check_and_update
from core.ocr_engine import convert_pdf_to_docx


class DhammaOCRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Myanmar Dhamma PDF -> Word OCR (Auto-Update)")
        self.geometry("640x430")
        self.resizable(False, False)

        self.pdf_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self.model_var = tk.StringVar(value=get_active_model_name())
        self.offset_var = tk.IntVar(value=0)
        self.workers_var = tk.IntVar(value=max(1, min(os.cpu_count() or 2, 6)))
        self.crop_var = tk.BooleanVar(value=True)

        frm = ttk.Frame(self, padding=14)
        frm.pack(fill="both", expand=True)

        top = ttk.Frame(frm)
        top.pack(fill="x", pady=(0, 8))
        ttk.Label(top, text="Active Model:", font=("", 10, "bold")).pack(side="left")
        ttk.Label(top, textvariable=self.model_var, foreground="#7A1F1F", font=("", 10, "bold")).pack(side="left", padx=6)
        ttk.Button(top, text="Check GitHub Update", command=self.run_update).pack(side="right")

        r1 = ttk.Frame(frm)
        r1.pack(fill="x", pady=4)
        ttk.Label(r1, text="PDF File:", width=12).pack(side="left")
        ttk.Entry(r1, textvariable=self.pdf_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(r1, text="Browse...", command=self.pick_pdf).pack(side="right")

        r2 = ttk.Frame(frm)
        r2.pack(fill="x", pady=4)
        ttk.Label(r2, text="Word Output:", width=12).pack(side="left")
        ttk.Entry(r2, textvariable=self.out_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(r2, text="Save As...", command=self.pick_out).pack(side="right")

        r3 = ttk.Frame(frm)
        r3.pack(fill="x", pady=8)
        ttk.Label(r3, text="Page Offset:").pack(side="left")
        ttk.Spinbox(r3, from_=0, to=100, width=5, textvariable=self.offset_var).pack(side="left", padx=(4, 14))
        ttk.Label(r3, text="CPU Cores:").pack(side="left")
        ttk.Spinbox(r3, from_=1, to=16, width=4, textvariable=self.workers_var).pack(side="left", padx=(4, 14))
        ttk.Checkbutton(r3, text="Crop Header/Footer Watermarks", variable=self.crop_var).pack(side="left")

        self.btn_start = ttk.Button(frm, text="Start OCR -> Word (.docx)", command=self.start_ocr)
        self.btn_start.pack(fill="x", pady=8)
        self.pbar = ttk.Progressbar(frm, mode="determinate")
        self.pbar.pack(fill="x", pady=4)

        self.log = tk.Text(frm, height=11, state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, pady=(6, 0))
        self.append_log(f"Ready. Active model: {self.model_var.get()}")

    def append_log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def pick_pdf(self):
        p = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if p:
            self.pdf_var.set(p)
            self.out_var.set(os.path.splitext(p)[0] + "_OCR.docx")

    def pick_out(self):
        p = filedialog.asksaveasfilename(defaultextension=".docx", filetypes=[("Word Document", "*.docx")])
        if p:
            self.out_var.set(p)

    def run_update(self):
        def _job():
            check_and_update(lambda m: self.after(0, self.append_log, m))
            self.after(0, self.model_var.set, get_active_model_name())
        threading.Thread(target=_job, daemon=True).start()

    def start_ocr(self):
        pdf_p = self.pdf_var.get().strip()
        out_p = self.out_var.get().strip()
        if not pdf_p or not os.path.exists(pdf_p):
            messagebox.showwarning("Missing PDF", "Please select a valid PDF file.")
            return
        self.btn_start.configure(state="disabled")
        self.pbar["value"] = 0
        self.append_log(f"Starting OCR on {os.path.basename(pdf_p)} using {self.model_var.get()}...")

        def _prog(done, total, pno):
            pct = (done / total) * 100
            self.after(0, lambda: self.pbar.configure(value=pct))
            if done % 10 == 0 or done == total:
                self.after(0, self.append_log, f"Progress: {done}/{total} pages ({pct:.0f}%)")

        def _work():
            try:
                res = convert_pdf_to_docx(
                    pdf_p, out_p, TESSDATA_DIR,
                    lang=self.model_var.get(),
                    page_offset=self.offset_var.get(),
                    workers=self.workers_var.get(),
                    crop_hf=self.crop_var.get(),
                    progress_cb=_prog,
                )
                self.after(0, self.append_log, f"DONE in {res['elapsed_sec']}s -> {res['out_docx']}")
                self.after(0, lambda: messagebox.showinfo("Completed", f"Saved Word file:\n{res['out_docx']}"))
            except Exception as e:
                self.after(0, self.append_log, f"ERROR: {e}")
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, lambda: self.btn_start.configure(state="normal"))

        threading.Thread(target=_work, daemon=True).start()


if __name__ == "__main__":
    DhammaOCRApp().mainloop()
