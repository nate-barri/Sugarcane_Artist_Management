import axios from 'axios'

const API = import.meta.env.VITE_API_URL || ''
export const api = axios.create({
  baseURL: API ? `${API}/api` : '/api', // uses proxy if no env
})
