import React from 'react'

const Feature = ({ title, description }) => (
  <div className="feature-card">
    <h3>{title}</h3>
    <p>{description}</p>
  </div>
)

export default function Home({ token, navigate }) {
  return (
    <div className="page-grid">
      <section className="hero-card">
        <span className="badge">Realtime deepfake detection</span>
        <h2>Analyze live video, images, or video files in one app.</h2>
        <p>
          Connect your webcam or upload media to receive classifier scores, heuristic alerts, and a transparent breakdown of why the model flagged the content.
        </p>
        <div className="hero-actions">
          <button onClick={() => navigate('/webcam')}>Start Webcam</button>
          <button onClick={() => navigate('/upload')}>Analyze Upload</button>
        </div>
        <div className="status-pill">
          {token ? 'Logged in: authentication enabled' : 'Not logged in: register or log in for session support'}
        </div>
      </section>

      <aside className="aside-card">
        <h3>Why this matters</h3>
        <p>
          Deepfakes are becoming harder to spot. This website combines live frame analysis and file-based scanning so you can compare scores and verify suspicious content quickly.
        </p>
      </aside>

      <Feature
        title="Live webcam streaming"
        description="Stream your camera frames over WebSocket and review predictions continuously in realtime."
      />
      <Feature
        title="Upload any media"
        description="Upload image or video files and get structured analysis with confidence and heuristic signals."
      />
      <Feature
        title="Authentication support"
        description="Register, login, and optionally keep your session authenticated when using the app."
      />
      <Feature
        title="FastAPI backend"
        description="The backend serves REST endpoints and a WebSocket pipe for the frontend to deliver fast inference and reporting." 
      />
    </div>
  )
}
