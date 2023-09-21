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
):
    response = requests.get(item.image_url, stream=True)
    with response_to_temp_file(response) as tmp_file:
        llm_message = model.prompt_image(item.prompt, tmp_file, temperature, beam_count)
        return {"answer": llm_message}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
