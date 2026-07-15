import React, { useState } from 'react'

export default function Login({ onLogin }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [message, setMessage] = useState('')

  async function submit() {
    setMessage('Logging in...')
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Login failed')
      window.localStorage.setItem('deepfake_token', data.access_token)
      setMessage('Logged in successfully')
      onLogin(data.access_token)
    } catch (err) {
      setMessage(err.message)
    }
  }

  return (
    <div>
      <h1>Login</h1>
      <div>
        <label>Username</label>
        <input value={username} onChange={e => setUsername(e.target.value)} />
      </div>
      <div>
        <label>Password</label>
        <input type="password" value={password} onChange={e => setPassword(e.target.value)} />
      </div>
      <button onClick={submit}>Login</button>
      <div>{message}</div>
    </div>
  )
}
