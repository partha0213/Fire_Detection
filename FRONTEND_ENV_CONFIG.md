# ✅ Frontend Environment Configuration - Complete

## Summary

Your frontend now uses **environment variables** for the backend URL instead of hardcoding `localhost:8000`.

## Changes Made

### 1. **Environment Files Created**

#### `frontend/.env.local`
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```
- Local development configuration
- Used by `npm run dev`
- **NOT committed to git** (.gitignore)

#### `frontend/.env.example`
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```
- Template for developers
- Copy to `.env.local` and modify as needed
- **Committed to git** for reference

### 2. **API Configuration Utility Created**

#### `frontend/lib/apiConfig.ts`
```typescript
export const getApiUrl = (): string => {
    return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
};

export const getWebSocketUrl = (): string => {
    const apiUrl = getApiUrl();
    return apiUrl.replace(/^http/, 'ws');  // http → ws, https → wss
};
```

**Features:**
- ✅ Reads from `NEXT_PUBLIC_API_URL` environment variable
- ✅ Falls back to `http://localhost:8000` if not set
- ✅ Automatically converts HTTP URLs to WebSocket URLs
- ✅ Works on both server and client side

### 3. **Frontend Components Updated**

#### `frontend/app/page.tsx` (Main App)
```typescript
import { getApiUrl } from '../lib/apiConfig';

// Health check
const apiUrl = getApiUrl();
const response = await fetch(`${apiUrl}/health`, { ... });

// File upload detection
const apiUrl = getApiUrl();
const response = await fetch(`${apiUrl}/detect`, { ... });

// Fetch alerts
const apiUrl = getApiUrl();
const response = await fetch(`${apiUrl}/alerts`);
```

**Changes:**
- ✅ Replaced 3 hardcoded `http://localhost:8000` URLs
- ✅ All now use `getApiUrl()` function

#### `frontend/components/MonitorView/index.tsx` (Camera Streaming)
```typescript
import { getWebSocketUrl } from '../../lib/apiConfig';

// Client-side camera (WebSocket)
const wsUrl = getWebSocketUrl();
const ws = new WebSocket(`${wsUrl}/ws/video`);

// Server-side camera (WebSocket)
const wsBaseUrl = getWebSocketUrl();
let wsUrl = `${wsBaseUrl}/ws/stream/${selectedCamera}`;
```

**Changes:**
- ✅ Replaced 2 hardcoded `ws://localhost:8000` URLs
- ✅ All now use `getWebSocketUrl()` function

## Environment Configuration Examples

### Local Development
```bash
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Docker Compose
```bash
# frontend/.env.local (in container)
NEXT_PUBLIC_API_URL=http://backend:8000
```

### Railway Deployment
```bash
# Railway dashboard → Variables
NEXT_PUBLIC_API_URL=https://your-railway-domain.railway.app
```

### Production
```bash
# Production build
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
```

## How to Use

### Local Development
```bash
cd frontend

# Install dependencies
npm install

# Create .env.local (already done)
# Edit if needed

# Start development server
npm run dev

# App will connect to http://localhost:8000
```

### Docker Compose
```bash
# docker-compose.yml
services:
  frontend:
    build: ./frontend
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
    ports:
      - "3000:3000"
  
  backend:
    build: ./backend
    ports:
      - "8000:8000"
```

### Railway Deployment
```bash
# In Railway dashboard:
# 1. Go to your frontend project
# 2. Go to Variables tab
# 3. Add: NEXT_PUBLIC_API_URL=https://your-railway-domain.railway.app
# 4. Redeploy
```

## Files Structure

```
frontend/
├── .env.local                    ← Local dev config (NOT in git)
├── .env.example                  ← Template (in git)
├── lib/
│   └── apiConfig.ts              ← NEW: API URL utilities
├── app/
│   └── page.tsx                  ← UPDATED: Uses getApiUrl()
├── components/
│   └── MonitorView/
│       └── index.tsx             ← UPDATED: Uses getWebSocketUrl()
└── ...
```

## Key Features

✅ **No Hardcoded URLs**
- All API endpoints use environment variable
- Easy to switch between environments

✅ **Automatic WebSocket Conversion**
- HTTP → WS, HTTPS → WSS
- No need to manage separate WebSocket URLs

✅ **Fallback to Localhost**
- If `NEXT_PUBLIC_API_URL` not set, defaults to `http://localhost:8000`
- No broken app in development

✅ **Environment-Specific**
- Different URLs for different environments
- No code changes needed for deployment

✅ **Secure**
- `.env.local` in `.gitignore` (not committed)
- API URL visible to browser (not a secret)

## Important Notes

### ⚠️ NEXT_PUBLIC_ Prefix is Required
```typescript
// ✅ CORRECT: Will be exposed to browser
NEXT_PUBLIC_API_URL=http://localhost:8000

// ❌ WRONG: Will NOT be available in browser
VITE_API_URL=http://localhost:8000
```

The `NEXT_PUBLIC_` prefix tells Next.js to build this variable into the client bundle.

### 🔒 Don't Store Secrets Here
```typescript
// ✅ OK: Public URLs
NEXT_PUBLIC_API_URL=https://api.yourdomain.com

// ❌ NEVER: API keys, passwords, tokens
NEXT_PUBLIC_SECRET_KEY=... (visible to everyone!)
```

### 🌐 URL Format Matters
```typescript
// ✅ Valid
http://localhost:8000
https://localhost:8000
http://backend:8000          (Docker)
https://api.yourdomain.com
https://your-railway-app.railway.app

// ❌ Invalid (missing protocol)
localhost:8000               // Missing http://
backend:8000                 // Missing http://

// ❌ Invalid (WebSocket URLs - handled automatically)
ws://localhost:8000          // Use http:// instead
wss://api.yourdomain.com     // Use https:// instead
```

## Troubleshooting

### Issue: "Cannot connect to API"
```
Check:
1. NEXT_PUBLIC_API_URL is set correctly
2. Backend is running at that URL
3. CORS is enabled on backend
4. Network connectivity

Solution:
# View what URL is being used
# In browser DevTools console:
console.log(process.env.NEXT_PUBLIC_API_URL)
```

### Issue: "Mixed Content" error
```
Cause: Frontend is HTTPS but backend is HTTP
Solution: Use HTTPS for both or HTTP for both
```

### Issue: WebSocket connection fails
```
Check:
1. WebSocket URL is correct (ws:// or wss://)
2. Backend WebSocket endpoints exist
3. Firewall allows WebSocket connections

Debug in browser console:
console.log(getWebSocketUrl())  // Should show ws:// or wss://
```

## Summary of Changes

| File | Change | Status |
|------|--------|--------|
| `frontend/.env.local` | Created (local config) | ✅ |
| `frontend/.env.example` | Created (template) | ✅ |
| `frontend/lib/apiConfig.ts` | Created (utility) | ✅ |
| `frontend/app/page.tsx` | Updated (use env var) | ✅ |
| `frontend/components/MonitorView/index.tsx` | Updated (use env var) | ✅ |

## Next Steps

1. **Start development:**
   ```bash
   cd frontend
   npm run dev
   ```
   Frontend will use `http://localhost:8000` from `.env.local`

2. **For Docker:**
   ```bash
   # Set in docker-compose.yml or Dockerfile
   NEXT_PUBLIC_API_URL=http://backend:8000
   ```

3. **For Railway:**
   ```bash
   # Add to Railway variables
   NEXT_PUBLIC_API_URL=https://your-railway-domain.railway.app
   ```

4. **For Production:**
   ```bash
   # Build with production URL
   NEXT_PUBLIC_API_URL=https://api.yourdomain.com npm run build
   ```

## Benefits

✅ **Flexibility**: Change backend URL without code changes  
✅ **Security**: Different URLs per environment  
✅ **Maintainability**: Single source of truth for API URL  
✅ **Scalability**: Ready for multi-environment deployment  
✅ **DX**: Easy developer onboarding (copy .env.example)  

---

**Frontend is now environment-ready!** 🚀

No more hardcoded URLs. Deploy anywhere with a simple environment variable change.
