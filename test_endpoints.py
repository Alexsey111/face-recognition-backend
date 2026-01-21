"""Test script to verify ML metrics endpoint."""

import sys
import asyncio

mods_to_remove = [k for k in sys.modules.keys() if k.startswith('app')]
for mod in mods_to_remove:
    del sys.modules[mod]

from app.main import create_app
from httpx import ASGITransport, AsyncClient

async def test():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://test') as client:
        # Test ml/details
        resp = await client.get('/api/v1/ml/details')
        print(f'GET /api/v1/ml/details: {resp.status_code}')
        print(f'Response: {resp.text[:500]}')
        
        # Test system/metrics
        resp2 = await client.get('/api/v1/system/metrics')
        print(f'\nGET /api/v1/system/metrics: {resp2.status_code}')
        print(f'Response: {resp2.text[:500]}')

if __name__ == '__main__':
    asyncio.run(test())