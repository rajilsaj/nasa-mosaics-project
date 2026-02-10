#!/usr/bin/env python3
# Minimal Cassini CAPS ELS crawler (no external packages)

from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen, Request
import os, sys, time

# ====== CONFIG ======
BASE_URL   = "https://pds-ppi.igpp.ucla.edu/data/cassini-caps-calibrated/data-els/"
DEST_DIR   = "cassini_caps_els"          # local folder to mirror into
YEARS      = None                        # e.g. {"2004","2005"} or None for all
EXTS       = None                        # e.g. {".DAT",".xml",".csv"} or None for all
TIMEOUT    = 60                          # request timeout (seconds)
UA         = "simple-mirror/0.1"
# =====================

class LinkGrabber(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag != "a": return
        href = dict(attrs).get("href")
        if href: self.links.append(href)

def child_links(html):
    p = LinkGrabber(); p.feed(html); hrefs = []
    for h in p.links:
        if h in ("", "../"):                           # skip parent/empty
            continue
        if h.startswith(("/", "?", "#", "icons/")):   # keep only child entries
            continue
        hrefs.append(h)
    return hrefs

def is_dir(href): return href.endswith("/")

def wants_year(path_rel):
    # Only enforce year filter at the root level
    if YEARS is None: return True
    parts = [p for p in path_rel.split("/") if p]
    if len(parts) == 0:   # root page
        return True
    if len(parts) == 1:   # first-level dirs are years (e.g., "2004/")
        return parts[0] in YEARS
    return True

def wants_ext(file_rel):
    if EXTS is None: return True
    lr = file_rel.lower()
    return any(lr.endswith(e.lower()) for e in EXTS)

def get(url):
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=TIMEOUT) as r:
        return r.read()

def download(url, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    # If server sends size, skip if we already have same size
    size_hdr = None
    try:
        req = Request(url, headers={"User-Agent": UA}, method="HEAD")
        with urlopen(req, timeout=TIMEOUT) as r:
            size_hdr = r.headers.get("Content-Length")
    except Exception:
        pass

    if size_hdr and os.path.exists(dst_path):
        if os.path.getsize(dst_path) == int(size_hdr):
            print(f"[skip] {dst_path}")
            return

    tmp = dst_path + ".part"
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=TIMEOUT) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk: break
            f.write(chunk)
    os.replace(tmp, dst_path)
    print(f"[ok]   {dst_path}")

def crawl(base_url, dest_dir, path_rel=""):
    url = urljoin(base_url, path_rel)
    try:
        html = get(url).decode("utf-8", "ignore")
    except Exception as e:
        print(f"[skip-dir] {url} ({e})")
        return

    links = [h for h in child_links(html)]
    # Recurse into subdirectories (create structure first)
    for h in links:
        if is_dir(h):
            if path_rel in ("", "/"):          # at root, enforce year filter
                name = h.rstrip("/")
                if YEARS is not None and name not in YEARS:
                    continue
            new_rel = path_rel + h
            crawl(base_url, dest_dir, new_rel)

    # Download files in this directory
    for h in links:
        if is_dir(h): 
            continue
        file_rel = path_rel + h
        if not wants_year(path_rel): 
            continue
        if not wants_ext(file_rel):
            continue
        file_url = urljoin(base_url, file_rel)
        local    = os.path.join(dest_dir, file_rel)
        try:
            download(file_url, local)
        except Exception as e:
            print(f"[fail] {file_rel}: {e}")
            time.sleep(1)

if __name__ == "__main__":
    # Allow overriding base/out via command line:
    # python simple_mirror.py [BASE_URL] [DEST_DIR]
    if len(sys.argv) >= 2: BASE_URL = sys.argv[1]
    if len(sys.argv) >= 3: DEST_DIR = sys.argv[2]
    # Example filters (uncomment if needed):
    # YEARS = {"2004", "2005"}
    # EXTS  = {".DAT", ".xml", ".csv"}

    # Normalize base
    if not urlparse(BASE_URL).scheme.startswith("http"):
        print("ERROR: BASE_URL must be http(s)"); sys.exit(1)
    if not BASE_URL.endswith("/"): BASE_URL += "/"

    crawl(BASE_URL, DEST_DIR)
    print("Done ->", os.path.abspath(DEST_DIR))

