from typing import List, Tuple


def process_mm_info(conversation: List[dict], use_audio_in_video: bool = False) -> Tuple[List[str], List[str], List[str]]:
    """Collect audio, image and video paths from a multimodal conversation."""
    audios = []
    images = []
    videos = []
    for message in conversation:
        for item in message.get("content", []):
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "audio":
                audios.append(item.get("audio"))
            elif t == "image":
                images.append(item.get("image"))
            elif t == "video":
                if use_audio_in_video:
                    videos.append(item.get("video"))
                else:
                    audios.append(item.get("video"))
    return audios, images, videos
