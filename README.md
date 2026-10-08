# KnowSure

KnowSure is a high-reliability verification dashboard for Small Language Models (SLMs). Rather than returning ungrounded answers, KnowSure enforces a strict **Generate → Retrieve → Verify → Decide** pipeline and abstains when ground-truth evidence is insufficient.

> **"Abstention is a feature, not a failure."**

---

## 🚀 Single-Service Architecture

This repository hosts both the static frontend UI and the FastAPI backend API in a single web service optimized for lightweight public deployment on Render's free plan (512 MB RAM limit).

- **Backend**: FastAPI + FastEmbed (ONNX, lightweight vector embeddings) + Gemini API SLM Judge.
- **Frontend**: React + Vite + Tailwind CSS static build served directly by FastAPI under `static/`.

---

## ☁️ Deploying on Render

1. Sign in to [render.com](https://render.com) using your GitHub account.
2. Click **New** → **Blueprint**.
3. Select this `knowsure` repository.
4. When prompted by Render, enter the required environment variables:
   - `GEMINI_API_KEY`: Your Google Gemini API key.
   - `KNOWSURE_WIKI_CONTACT`: Your contact email or project URL (required by Wikipedia API user-agent policies).
5. Click **Apply**. Render will automatically build and launch the service.

---

## ⚖️ License & Credits

Developed by **Team ErrorX**.
