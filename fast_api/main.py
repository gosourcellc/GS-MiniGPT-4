import uvicorn
from fastapi import (
    FastAPI,
    File,
    Depends,
    Path,
    UploadFile,
    Form,
    HTTPException,
    status,
    Request,
)
from typing import Annotated, Optional

from fast_api.llm.minigtp import MiniGPT


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


@app.post("/describe")
async def root(
    prompt: Annotated[str, Form(title="User prompt", min_length=10, max_length=3000)],
    file: Annotated[UploadFile, File()],
    temperature: Annotated[float, Path(title="Temperature", gt=0, le=2)] = 1,
    beam_count: Annotated[int, Path(title="Beam search numbers", ge=1, le=10)] = 1,
    _current_token: str = Depends(get_current_token),
):
    llm_message = model.prompt_image(prompt, file, temperature, beam_count)
    return {"answer": llm_message}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
