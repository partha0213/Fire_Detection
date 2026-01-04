# Environment Configuration Guide

## Frontend Environment Variables

The frontend now uses environment variables for the backend API URL, allowing easy configuration across different deployment environments.

### Setup

#### 1. **Local Development**

Create `frontend/.env.local`:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

#### 2. **Docker Development**

Use the backend service name:
```bash
NEXT_PUBLIC_API_URL=http://backend:8000
```

#### 3. **Railway Deployment**

Set in Railway dashboard:
```bash
NEXT_PUBLIC_API_URL=https://your-railway-domain.railway.app
```

#### 4. **Production**

Use your production domain:
```bash
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
```

## How It Works

### Configuration File: `frontend/lib/apiConfig.ts`

This utility module handles API URL resolution:

```typescript
export const getApiUrl = (): string => {
    return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
};

export const getWebSocketUrl = (): string => {
    const apiUrl = getApiUrl();
    return apiUrl.replace(/^http/, 'ws');  // Convert http→ws, https→wss
};
```

**Key Features:**
- ✅ Automatically detects environment
- ✅ Converts HTTP to WebSocket URLs
- ✅ Fallback to localhost if not configured
- ✅ Works on both server and client side

### Updated Components

#### 1. **frontend/app/page.tsx** (Main App)
```typescript
import { getApiUrl } from '../lib/apiConfig';

const apiUrl = getApiUrl();
const response = await fetch(`${apiUrl}/health`);
const response = await fetch(`${apiUrl}/detect`, { method: 'POST' });
const response = await fetch(`${apiUrl}/alerts`);
```

#### 2. **frontend/components/MonitorView/index.tsx** (Camera Streaming)
```typescript
import { getWebSocketUrl } from '../../lib/apiConfig';

const wsUrl = getWebSocketUrl();
const ws = new WebSocket(`${wsUrl}/ws/video`);
const ws = new WebSocket(`${wsUrl}/ws/stream/${type}`);
```

## Files Modified

### Frontend Structure
```
frontend/
├── .env.local              (NEW - local configuration)
├── .env.example            (NEW - template for developers)
├── lib/
│   └── apiConfig.ts        (NEW - API URL utilities)
├── app/
│   └── page.tsx            (UPDATED - uses environment variables)
└── components/
    └── MonitorView/
        └── index.tsx       (UPDATED - uses environment variables)
```

## Deployment Instructions

### Local Development
```bash
cd frontend
npm install
# Create .env.local with:
# NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

### Docker Compose
```bash
# docker-compose.yml
services:
  frontend:
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
  backend:
    # ... backend config
```

### Railway
1. Go to Railway Dashboard
2. Select your frontend project
3. Go to **Variables**
4. Add: `NEXT_PUBLIC_API_URL=https://your-railway-domain.railway.app`
5. Redeploy

### Production
```bash
# Build with production URL
NEXT_PUBLIC_API_URL=https://api.yourdomain.com npm run build
```

## Important Notes

### ⚠️ NEXT_PUBLIC_ Prefix

- `NEXT_PUBLIC_` variables are **built into the client bundle**
- They are visible in browser (not secrets!)
- Use only for non-sensitive URLs
- **Do not** put API keys or passwords here

### 🔧 URL Format

Supported formats:
```
http://localhost:8000          ✅ Local
https://localhost:8000         ✅ HTTPS local
http://backend:8000            ✅ Docker service
https://api.yourdomain.com     ✅ Production
https://your-railway-app.railway.app  ✅ Railway
```

### 🚀 WebSocket Conversion

Automatic conversion happens:
```
http://example.com  →  ws://example.com
https://example.com  →  wss://example.com
```

## Troubleshooting

### Issue: "Cannot connect to API"
**Check:**
1. `NEXT_PUBLIC_API_URL` is set correctly
2. Backend is running at that URL
3. CORS is enabled on backend
4. URL format is correct (http/https, not ws)

**Solution:**
```bash
# Verify environment variable
echo $NEXT_PUBLIC_API_URL
# Should show your API URL

# Check browser console
# Look for fetch/WebSocket errors with actual URL used
```

### Issue: "Mixed Content" error
**Cause:** Frontend is HTTPS but backend is HTTP
**Solution:** 
- Use HTTPS for both frontend and backend
- Or use HTTP for both in development

### Issue: WebSocket connection fails
**Check:**
1. `getWebSocketUrl()` correctly converts to ws/wss
2. Backend WebSocket endpoints exist
3. Firewall allows WebSocket connections

**Solution:**
```typescript
// Debug in browser console
console.log(getApiUrl());        // Should show API URL
console.log(getWebSocketUrl());  // Should show ws/wss URL
```

## Environment Variables Summary

| Environment | URL | Use |
|-------------|-----|-----|
| Local Dev | `http://localhost:8000` | Default, testing |
| Docker | `http://backend:8000` | Container networking |
| Railway | `https://your-domain.railway.app` | Production |
| Production | `https://api.yourdomain.com` | Final deployment |

## Next Steps

1. ✅ Created `frontend/.env.local` with localhost URL
2. ✅ Created `frontend/.env.example` with documentation
3. ✅ Created `frontend/lib/apiConfig.ts` utility
4. ✅ Updated `frontend/app/page.tsx` to use environment variables
5. ✅ Updated `frontend/components/MonitorView/index.tsx` to use environment variables

**Ready to deploy with environment-specific URLs!** 🚀
