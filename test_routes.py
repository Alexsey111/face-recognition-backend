"""Test script to verify routes are registered correctly."""

import sys
import asyncio

# Clear cache
mods_to_remove = [k for k in sys.modules.keys() if k.startswith('app')]
for mod in mods_to_remove:
    del sys.modules[mod]

from app.main import create_app
from fastapi.routing import Match

async def test():
    app = create_app()
    
    # Test path
    test_path = '/api/v1/health/ml/details'
    
    scope = {
        'type': 'http',
        'method': 'GET',
        'path': test_path,
        'query_string': b'',
        'headers': [],
        'root_path': '',
        'path_params': {},
        'query_params': {},
    }
    
    # Find matching route
    print(f"Testing path: {test_path}")
    print(f"\nAll routes in app.routes:")
    for route in app.routes:
        if hasattr(route, 'path') and 'health' in route.path:
            print(f"  {route.methods} {route.path}")
    
    print(f"\nMatching routes:")
    for route in app.routes:
        match, _ = route.matches(scope)
        if match != Match.NONE:
            endpoint = getattr(route, 'endpoint', None)
            endpoint_name = endpoint.__name__ if endpoint else 'unknown'
            print(f"  Match: {route.path} -> {endpoint_name}")
    
    # Also test system/metrics
    test_path2 = '/api/v1/health/system/metrics'
    scope2 = {**scope, 'path': test_path2}
    print(f"\nTesting path: {test_path2}")
    for route in app.routes:
        match, _ = route.matches(scope2)
        if match != Match.NONE:
            endpoint = getattr(route, 'endpoint', None)
            endpoint_name = endpoint.__name__ if endpoint else 'unknown'
            print(f"  Match: {route.path} -> {endpoint_name}")

if __name__ == '__main__':
    asyncio.run(test())