import React, { useState } from 'react'
import Home from './pages/Home'
import Upload from './pages/Upload'
import Webcam from './pages/Webcam'
import Login from './pages/Login'
import Register from './pages/Register'
import './styles.css'

const normalizePath = (path) => path.replace(/\/?$/, '') || '/'

export default function App() {
  const [route, setRoute] = useState(normalizePath(window.location.pathname))
  const [token, setToken] = useState(window.localStorage.getItem('deepfake_token'))

  React.useEffect(() => {
    const onPop = () => setRoute(normalizePath(window.location.pathname))
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  const navigate = (path) => {
    window.history.pushState({}, '', path)
    setRoute(normalizePath(path))
  }

  const handleLogout = () => {
    window.localStorage.removeItem('deepfake_token')
    setToken(null)
    navigate('/')
  }

  const handleLogin = (newToken) => {
    setToken(newToken)
    navigate('/')
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Deepfake Detector</p>
          <h1>Realtime analysis for webcam, image, and video uploads</h1>
        </div>

        <nav className="nav-menu">
          <button onClick={() => navigate('/')}>Home</button>
          <button onClick={() => navigate('/webcam')}>Webcam</button>
          <button onClick={() => navigate('/upload')}>Upload</button>
          <button onClick={() => navigate('/login')}>Login</button>
          <button onClick={() => navigate('/register')}>Register</button>
          {token && <button onClick={handleLogout}>Logout</button>}
        </nav>
      </header>

      <main className="page-body">
        {route === '/' && <Home token={token} navigate={navigate} />}
        {route === '/webcam' && <Webcam token={token} />}
        {route === '/upload' && <Upload token={token} />}
        {route === '/login' && <Login onLogin={handleLogin} />}
        {route === '/register' && <Register onRegistered={() => navigate('/login')} />}
        {['/', '/webcam', '/upload', '/login', '/register'].includes(route) === false && (
          <div className="page-card">
            <h2>Page not found</h2>
            <p>The page you requested does not exist. Use the navigation above.</p>
          </div>
        )}
      </main>

      <footer className="footer-bar">
        <span>FastAPI + React</span>
        <span>{token ? 'Authenticated session active' : 'Unauthenticated mode'}</span>
      </footer>
    </div>
  )
}
