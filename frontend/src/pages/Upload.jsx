import React, {useState} from 'react'

export default function Upload({ token }) {
  const [file, setFile] = useState(null)
  const [resp, setResp] = useState(null)
  const [loading, setLoading] = useState(false)

  async function onUpload() {
    if (!file) return
    setLoading(true)
    const formData = new FormData()
    formData.append('file', file)
    const headers = {}
    if (token) headers.Authorization = `Bearer ${token}`

    try {
      const res = await fetch('/analyze', { method: 'POST', body: formData, headers })
      const data = await res.json()
      setResp(data)
    } catch (err) {
      setResp({ error: err.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-card upload-card">
      <div className="page-header">
        <div>
          <span className="badge">Upload Analyzer</span>
          <h2>Scan an image or video for deepfake signals</h2>
        </div>
        <div>{token ? 'Authenticated upload session' : 'Upload without authentication'}</div>
      </div>

      <div className="upload-form">
        <input type="file" accept="image/*,video/*" onChange={e => setFile(e.target.files[0])} />
        <button onClick={onUpload} disabled={loading || !file}>
          {loading ? 'Analyzing...' : 'Analyze File'}
        </button>
      </div>

      {file && <p className="file-name">Selected: {file.name}</p>}

      <div className="result-panel">
        <h3>Result</h3>
        <pre>{resp ? JSON.stringify(resp, null, 2) : 'No analysis yet.'}</pre>
      </div>
    </div>
  )
}
