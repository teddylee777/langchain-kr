import base64
import requests
from IPython.display import Image, display
import os


class MultiModal:
    def __init__(self, model, system_prompt=None, user_prompt=None):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.init_prompt()

    def init_prompt(self):
        if self.system_prompt is None:
            self.system_prompt = "You are a helpful assistant who helps users to write a report related to images in Korean."
        if self.user_prompt is None:
            self.user_prompt = "Explain the images as an alternative text in Korean."

    # 이미지를 base64로 인코딩하는 함수 (URL)
    def encode_image_from_url(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            image_content = response.content
            if url.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            elif url.lower().endswith(".png"):
                mime_type = "image/png"
            else:
                mime_type = "image/unknown"
            return f"data:{mime_type};base64,{base64.b64encode(image_content).decode('utf-8')}"
        else:
            raise Exception("Failed to download image")

    # 이미지를 base64로 인코딩하는 함수 (파일)
    def encode_image_from_file(self, file_path):
        with open(file_path, "rb") as image_file:
            image_content = image_file.read()
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in [".jpg", ".jpeg"]:
                mime_type = "image/jpeg"
            elif file_ext == ".png":
                mime_type = "image/png"
            else:
                mime_type = "image/unknown"
            return f"data:{mime_type};base64,{base64.b64encode(image_content).decode('utf-8')}"

    # 이미지 경로에 따라 적절한 함수를 호출하는 함수
    def encode_image(self, image_path):
        if image_path.startswith("http://") or image_path.startswith("https://"):
            return self.encode_image_from_url(image_path)
        else:
            return self.encode_image_from_file(image_path)

    def display_image(self, encoded_image):
        display(Image(url=encoded_image))

    def create_messages(
        self, image_url, system_prompt=None, user_prompt=None, display_image=True
    ):
        encoded_image = self.encode_image(image_url)
        if display_image:
            self.display_image(encoded_image)

        system_prompt = (
            system_prompt if system_prompt is not None else self.system_prompt
        )

        user_prompt = user_prompt if user_prompt is not None else self.user_prompt

        # 인코딩된 이미지를 사용하여 다른 처리를 수행할 수 있습니다.
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"{encoded_image}"},
                    },
                ],
            },
        ]
        return messages

    def invoke(
        self, image_url, system_prompt=None, user_prompt=None, display_image=True
    ):
        messages = self.create_messages(
            image_url, system_prompt, user_prompt, display_image
        )
        response = self.model.invoke(messages)
        return response.content

    def stream(
        self, image_url, system_prompt=None, user_prompt=None, display_image=True
    ):
        messages = self.create_messages(
            image_url, system_prompt, user_prompt, display_image
        )
        response = self.model.stream(messages)
        return response
