import React, { useEffect, useRef, useState } from 'react'

export default function Webcam({ token }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [wsStatus, setWsStatus] = useState('disconnected')
  const [prediction, setPrediction] = useState(null)
  const wsRef = useRef(null)

  useEffect(() => {
    const loc = window.location
    const wsUrl = (loc.protocol === 'https:' ? 'wss://' : 'ws://') + loc.host + '/ws'
    const socket = new WebSocket(wsUrl)

    socket.onopen = () => setWsStatus('connected')
    socket.onclose = () => setWsStatus('disconnected')
    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        setPrediction(data)
      } catch (err) {
        console.error(err)
      }
    }

    wsRef.current = socket
    return () => socket.close()
  }, [])

  useEffect(() => {
    let intervalId
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
        }
        intervalId = window.setInterval(() => {
          if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
          const canvas = canvasRef.current
          const video = videoRef.current
          if (!canvas || !video) return
          canvas.width = video.videoWidth || 640
          canvas.height = video.videoHeight || 480
          const ctx = canvas.getContext('2d')
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
          const dataUrl = canvas.toDataURL('image/jpeg', 0.7)
          wsRef.current.send(JSON.stringify({ frame: dataUrl }))
        }, 300)
      } catch (err) {
        setPrediction({ error: 'Camera access denied or unavailable.' })
      }
    }

    startCamera()
    return () => window.clearInterval(intervalId)
  }, [])

  return (
    <div className="page-card webcam-card">
      <div className="page-header">
        <div>
          <span className="badge">Webcam Analyzer</span>
          <h2>Realtime frame scoring with WebSocket analysis</h2>
        </div>
        <div>{token ? 'Live session active' : 'Live mode without auth'}</div>
      </div>

      <div className="webcam-grid">
        <div className="video-frame">
          <video ref={videoRef} muted playsInline />
          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>
        <div className="prediction-panel">
          <div className="status-line">
            <strong>Status:</strong> {wsStatus}
          </div>
          <div className="prediction-card">
            <p><strong>Prediction:</strong> {prediction?.prediction || '—'}</p>
            <p><strong>Confidence:</strong> {prediction?.confidence ? `${prediction.confidence}%` : '—'}</p>
            <p><strong>Heuristic score:</strong> {prediction?.fake_score ? `${prediction.fake_score}%` : '—'}</p>
            <p><strong>Model status:</strong> {'model_loaded' in prediction ? (prediction.model_loaded ? `Loaded (${prediction.classifier_type || 'unknown'})` : 'Not loaded') : 'Loading...'}</p>
          </div>
          {prediction?.reason_scores && (
            <div className="result-panel">
              <h3>Reason scores</h3>
              <pre>{JSON.stringify(prediction.reason_scores, null, 2)}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
