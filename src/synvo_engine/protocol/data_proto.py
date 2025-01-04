from typing import List, Literal, Union

from pydantic import BaseModel, Field


class SFTChatDataText(BaseModel):
    type: Literal["text"]
    text: str


class SFTChatDataImageURL(BaseModel):
    url: str


class SFTChatDataImage(BaseModel):
    type: Literal["image_url"]
    image_url: SFTChatDataImageURL


class SFTChatDataAudioURL(BaseModel):
    url: str


class SFTChatDataAudio(BaseModel):
    type: Literal["audio_url"]
    audio_url: SFTChatDataAudioURL


# Hf dataset needs field to be the same across columns
class HFDataContent(BaseModel):
    type: Literal["text", "image_url", "audio_url"]
    text: str
    image_url: SFTChatDataImageURL


SFTChatDataContent = Union[
    SFTChatDataText, SFTChatDataImage, SFTChatDataAudio, HFDataContent
]


class SFTChatDataMessages(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: List[SFTChatDataContent]


class SFTChatData(BaseModel):
    messages: List[SFTChatDataMessages]
    id: int


class PreferenceData(BaseModel):
    id: int
    chosen: List[SFTChatDataMessages]
    rejected: List[SFTChatDataMessages]
    prompt: List[SFTChatDataMessages]
