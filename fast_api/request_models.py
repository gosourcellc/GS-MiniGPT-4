from pydantic import BaseModel


class ImageDetailsRequest(BaseModel):
    prompt: str
    image_url: str
