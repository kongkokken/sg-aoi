# Deploying the PCBA AOI Demo (Free Hosting)

This app runs in **DEMO MODE** out of the box: synthetic board images and
scripted detections under `data/demo/` drive the real rule engine, so **no
trained model and no heavy ML dependencies** (paddlepaddle / anomalib) are
needed. Demo mode auto-activates whenever `data/demo/detections_ok.json`
exists (it is committed to the repo), and the app boots on
`configs/pipeline.demo.yaml` when the `AOI_CONFIG` environment variable is
set (see below).

Two free ways to show it to management:

---

## Path 1 — Streamlit Community Cloud (recommended for a multi-day demo)

Best when viewers should open the link on their own devices over several days.

### Step 1 — Put the code on GitHub (free)

1. Create a free account at <https://github.com> → **Sign up**.
2. Click the **+** (top right) → **New repository**.
   - Name: `pcba-aoi` (anything works)
   - Visibility: **Public** (required on the free Streamlit tier)
   - Do **not** tick "Add a README" (the repo already has one).
   - Click **Create repository**.
3. From this project folder (`pcba-aoi/`), push the existing commit:

   ```bash
   git remote add origin https://github.com/<YOUR-USERNAME>/pcba-aoi.git
   git branch -M main
   git push -u origin main
   ```

   (Git will ask you to sign in — use the browser window that pops up.)

### Step 2 — Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> → **Sign in with GitHub** and authorize.
2. Click **New app** (or **Create app**).
3. Fill in:
   - **Repository:** `<YOUR-USERNAME>/pcba-aoi`
   - **Branch:** `main`
   - **Main file path:** `src/app.py`
4. Click **Advanced settings…** and:
   - **Python version:** select **3.13** (do NOT leave it on the newest
     default — very new Pythons like 3.14 may lack prebuilt wheels for some
     packages, forcing slow source builds that fail; seen in practice with
     `pillow==11.0.0` / Python 3.14: "headers or library files could not be
     found for zlib").
   - Add this environment variable so the app boots on the demo config (also
     fine to skip — demo mode auto-activates either way; this just makes the
     Settings page edit the demo config):

   ```
   AOI_CONFIG=configs/pipeline.demo.yaml
   ```

   (On Streamlit Cloud this can also be set later under
   **App → Settings → Secrets** as `AOI_CONFIG = "configs/pipeline.demo.yaml"`.)
5. Click **Deploy**. First build takes ~2–4 minutes while it installs
   `requirements.txt`.

### Troubleshooting

- **Deploy "stuck" 10+ minutes on `Processing dependencies…`, or the log
  shows `Failed to download and build pillow` / `RequiredDependencyException:
  zlib`:** the app is building on a Python that is too new for the pinned
  packages (the build has already failed — it will not recover). Fix: this
  repo's `requirements.txt` uses minimum-version floors instead of hard pins;
  pull the latest commit, then in Streamlit Cloud either delete the app and
  redeploy with **Python 3.13** (Advanced settings), or **App → Settings →
  Reboot** after the fix is pushed. A healthy build finishes dependency
  install in ~1–3 minutes and ends with the app URL going live.

### Good to know

- **Free tier sleeps when idle.** After a period with no visitors the app
  goes to sleep; the first visit afterwards shows a "waking up" screen and
  takes roughly **~30 s** (cold start). For a meeting, open the URL once a
  few minutes beforehand.
- **Live camera works for every viewer.** On the *Inspection* page, the
  "Take a photo" snapshot uses the **viewer's own webcam/phone camera** in
  their browser — nothing needs to be plugged into the server.
- The app URL is `https://<your-app-name>.streamlit.app` and can be shared
  with anyone — no login needed for viewers.

---

## Path 2 — Instant same-day demo, no account: Cloudflare quick tunnel

Best for a live meeting-room demo **today**, straight from your laptop. No
GitHub or Streamlit account needed; the URL exists only while your laptop
and the script are running.

### Step 1 — Install cloudflared (one time)

On Windows, either:

```powershell
winget install cloudflare.cloudflared
```

or download the MSI from
<https://github.com/cloudflare/cloudflared/releases> (pick
`cloudflared-windows-amd64.msi`) and run it.

### Step 2 — Run the one-click script

Double-click **`start_demo_tunnel.bat`** in this folder (or run it from a
terminal). It will:

1. Activate the local `.venv-demo` environment (installing Streamlit into
   it first if missing),
2. Start the app on `http://localhost:8501`,
3. Start a Cloudflare quick tunnel and print a public
   **`https://<random>.trycloudflare.com`** URL.

Share that URL in the meeting — anyone can open it while the script runs.

### Good to know

- The URL is **temporary**: it dies when you close the window / press
  `Ctrl+C`, and a new random URL is issued each run. Perfect for a single
  meeting; use Path 1 for anything longer.
- Viewers' webcam snapshots work through the tunnel too (Cloudflare serves
  it over HTTPS, which browsers require for camera access).
- Firewall prompt on first run: allow `cloudflared` and Python/network
  access.

---

## What viewers will see

- A **"DEMO MODE — simulated detections"** banner at the top.
- A scenario selector on the *Inspection* page: **OK board**,
  **NG — missing part (R7)**, **NG — wrong part (C3)** — each runs the real
  rule engine and fusion logic, producing the real verdict JSON + annotated
  image in *Review History*.
- *Settings* edits thresholds live; *Dataset & Training* and *Setup Wizard*
  show the roadmap state.

## Local run (for reference)

```bash
# From this folder, with the demo venv:
.venv-demo/Scripts/python -m streamlit run src/app.py
# Force the demo config explicitly (optional — auto-detected anyway):
#   Windows cmd:   set AOI_CONFIG=configs/pipeline.demo.yaml
#   Git Bash:      AOI_CONFIG=configs/pipeline.demo.yaml .venv-demo/Scripts/python -m streamlit run src/app.py
```
