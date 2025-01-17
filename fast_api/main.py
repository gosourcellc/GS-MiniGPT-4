import uvicorn
import requests
from fastapi import (
    FastAPI,
    Depends,
    Path,
    HTTPException,
    status,
    Request,
)
from typing import Annotated, Optional

from fast_api.llm.minigtp import MiniGPT
from fast_api.request_models import ImageDetailsRequest
from fast_api.utils import response_to_temp_file

SECRET_TOKEN = "mysecrettoken123"

"""
curl -X 'POST' \
  'https://pb9z4lidq99o0j-8000.proxy.runpod.net/describe?temperature=1&beam_count=1&max_new_tokens=1000' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer mysecrettoken123' \
  -d '{
  "prompt": "describe the image as detailed as possible",
  "image_url": "https://res.cloudinary.com/go-source/image/upload/v1692830079/erp_product/k2weg3ymjauhmlqifplk.jpg"
}'
"""


def get_current_token(request: Request):
    authorization: Optional[str] = request.headers.get("Authorization")
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Authorization header missing"
        )

    token = authorization.split("Bearer ")[-1]

    if token != SECRET_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token or token expired",
        )

    return token


app = FastAPI()

model = MiniGPT()
model.setup(MiniGPT.get_chat_arguments())


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/describe", dependencies=[Depends(get_current_token)])
async def describe(
    item: ImageDetailsRequest,
    temperature: Annotated[float, Path(title="Temperature", gt=0, le=2)] = 1,
    beam_count: Annotated[int, Path(title="Beam search numbers", ge=1, le=10)] = 1,
    max_new_tokens: Annotated[
        int, Path(title="New tokens count", ge=300, le=1000)
    ] = 1000,
):
    response = requests.get(item.image_url, stream=True)
    with response_to_temp_file(response) as tmp_file:
        llm_message = model.prompt_image(
            item.prompt, tmp_file, temperature, beam_count, max_new_tokens
        )
        return {"answer": llm_message}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
