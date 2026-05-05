import httpx
import fitz
import asyncio

async def test_upload():
    # 1. Create a dummy PDF
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(72, 72), "This is a test document uploaded to the dockerized API.")
    pdf_bytes = doc.write()
    doc.close()

    # 2. Upload it
    async with httpx.AsyncClient() as client:
        files = {'file': ('test_upload.pdf', pdf_bytes, 'application/pdf')}
        response = await client.post('http://localhost:8000/documents/upload', files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

if __name__ == "__main__":
    asyncio.run(test_upload())
