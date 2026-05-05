import httpx
import asyncio

async def test_get():
    async with httpx.AsyncClient() as client:
        response = await client.get('http://localhost:8000/documents')
        print(response.json())

if __name__ == "__main__":
    asyncio.run(test_get())
