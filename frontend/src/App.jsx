import { useEffect, useState } from 'react'
import { api } from './api'

export default function App() {
  const [todos, setTodos] = useState([])

  useEffect(() => {
    api.get('/todos/').then(res => setTodos(res.data))
  }, [])

  return (
    <main style={{ padding: 24 }}>
      <h1>Todos</h1>
      <ul>
        {todos.map(t => (
          <li key={t.id}>
            <input type="checkbox" checked={t.done} readOnly /> {t.title}
          </li>
        ))}
      </ul>
    </main>
  )
}
