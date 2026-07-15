import React, { useState } from 'react'

export default function Register({ onRegistered }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [message, setMessage] = useState('')

  async function submit() {
    setMessage('Registering...')
    try {
      const res = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Registration failed')
      setMessage('Registration successful, please login')
      onRegistered()
    } catch (err) {
      setMessage(err.message)
    }
  }

  return (
    <div>
      <h1>Register</h1>
      <div>
        <label>Username</label>
        <input value={username} onChange={e => setUsername(e.target.value)} />
      </div>
      <div>
        <label>Password</label>
        <input type="password" value={password} onChange={e => setPassword(e.target.value)} />
      </div>
      <button onClick={submit}>Register</button>
      <div>{message}</div>
    </div>
  )
}
