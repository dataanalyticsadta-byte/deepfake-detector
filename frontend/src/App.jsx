import React, { useRef, useState, useEffect } from 'react'

export default function App(){
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [wsStatus, setWsStatus] = useState('disconnected')
  const [prediction, setPrediction] = useState(null)
  const wsRef = useRef(null)
  const [uploadResult, setUploadResult] = useState(null)

  useEffect(()=>{
    const loc = window.location
    const wsUrl = (loc.protocol === 'https:' ? 'wss://' : 'ws://') + loc.host + '/ws'
    const ws = new WebSocket(wsUrl)
    ws.onopen = ()=> setWsStatus('connected')
    ws.onclose = ()=> setWsStatus('disconnected')
    ws.onmessage = (ev)=>{
      try{ const d = JSON.parse(ev.data); setPrediction(d) }catch(e){ console.error(e) }
    }
    wsRef.current = ws
    return ()=> ws.close()
  }, [])

  async function startCamera(){
    const stream = await navigator.mediaDevices.getUserMedia({video:true})
    videoRef.current.srcObject = stream
    videoRef.current.play()
    const interval = setInterval(()=>{
      if(!wsRef.current || wsRef.current.readyState!==WebSocket.OPEN) return
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
      const dataUrl = canvas.toDataURL('image/jpeg', 0.7)
      wsRef.current.send(JSON.stringify({frame: dataUrl}))
    }, 350)
    return ()=> clearInterval(interval)
  }

  useEffect(()=>{ let stop; startCamera().then(s=>stop=s).catch(()=>{}); return ()=> stop && stop() }, [])

  async function handleUpload(e){
    const f = e.target.files[0]
    if(!f) return
    const form = new FormData(); form.append('file', f)
    setUploadResult('uploading...')
    try{
      const res = await fetch('/analyze', {method:'POST', body: form})
      const j = await res.json()
      setUploadResult(j)
    }catch(err){ setUploadResult('error: '+err.message) }
  }

  return (
    <div className="app">
      <h1>Realtime Deepfake Detector</h1>
      <div className="container">
        <div className="left">
          <video ref={videoRef} width={480} height={360} />
          <canvas ref={canvasRef} width={480} height={360} style={{display:'none'}} />
          <div>Status: {wsStatus}</div>
          <div>
            <strong>Realtime prediction:</strong>
            <div>{prediction ? `${prediction.prediction} (${prediction.confidence}%)` : '—'}</div>
            {prediction?.breakdown && (
              <div>
                <strong>Overall:</strong>
                <div>AI: {prediction.breakdown.AI}% — Real: {prediction.breakdown.Real}%</div>
              </div>
            )}
            {prediction?.reason_scores && (
              <div>
                <strong>Reason scores:</strong>
                <ul>
                  {Object.entries(prediction.reason_scores).map(([label, score]) => (
                    <li key={label}>
                      {label}: {score}% {prediction.reasons?.includes(label) ? '(flagged)' : ''}
                      {prediction.reason_details?.[label] ? ` — ${prediction.reason_details[label]}` : ''}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
        <div className="right">
          <h2>Upload file</h2>
          <input type="file" accept="image/*,video/*" onChange={handleUpload} />
          <h3>Result</h3>
          {uploadResult && typeof uploadResult === 'object' ? (
            <div>
              <div><strong>Prediction:</strong> {uploadResult.prediction} ({uploadResult.confidence}%)</div>
              {uploadResult.breakdown && <div><strong>Overall:</strong> AI: {uploadResult.breakdown.AI}% — Real: {uploadResult.breakdown.Real}%</div>}
              {uploadResult.reason_scores && (
                <div>
                  <strong>Reason scores:</strong>
                  <ul>
                    {Object.entries(uploadResult.reason_scores).map(([label, score]) => (
                      <li key={label}>{label}: {score}% {uploadResult.reasons?.includes(label) ? '(flagged)' : ''}{uploadResult.reason_details?.[label] ? ` — ${uploadResult.reason_details[label]}` : ''}</li>
                    ))}
                  </ul>
                </div>
              )}
              <pre style={{background:'#f6f6f6', padding:8}}>{JSON.stringify(uploadResult.report || {}, null, 2)}</pre>
            </div>
          ) : (
            <pre>{uploadResult}</pre>
          )}
        </div>
      </div>
    </div>
  )
}
