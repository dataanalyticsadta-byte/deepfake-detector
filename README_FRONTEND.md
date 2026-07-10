Frontend (React + Vite)

From the project root:

1. Install dependencies

```bash
cd frontend
npm install
```

2. Run dev server (hot reload)

```bash
npm run dev
```

The dev server runs on `http://localhost:5173` by default. It will proxy requests to your backend if configured; by default the frontend talks directly to `http://localhost:8000`.

3. Build for production

```bash
npm run build
```

This will produce `frontend/dist`. If you then run the Python backend, the server will automatically serve the built files at `/`.

Notes
- The React UI provides realtime webcam streaming (uses `/ws`) and a file upload form that posts to `/analyze`.
- To serve the built React app from the Python server, run `npm run build` then restart the FastAPI server.
