import uvicorn
from fastapi import FastAPI, Path, UploadFile, Form
from typing import Annotated, Optional
from pydantic import BaseModel, Field

from fast_api.llm.minigtp import MiniGPT


class ImageDetailsRequest(BaseModel):
    prompt: str = Field(
        default=None, title="User prompt", min_length=10, max_length=3000
    )


app = FastAPI()

model = MiniGPT()
model.setup(MiniGPT.get_chat_arguments())


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/describe")
async def root(
    prompt: Annotated[str, Form(title="User prompt", min_length=10, max_length=3000)],
    file: UploadFile,
    temperature: Annotated[float, Path(title="Temperature", gt=0, le=2)] = 1,
    beam_count: Annotated[int, Path(title="Beam search numbers", ge=1, le=10)] = 1,
):
    llm_message = model.prompt_image(prompt, file, temperature, beam_count)
    return {"answer": llm_message}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
