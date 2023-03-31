import io
from bson.binary import Binary
from slugify import slugify
from PIL import Image
from domain.User import User
import re


def filename_is_empty(filename: str) -> bool:
    return filename == ''


def file_is_empty(image: Binary) -> bool:
    return image == Binary(io.BytesIO().getvalue())


class Article:
    __step_prefix = "step_"
    __file_step_prefix = "file_step_"
    __title_step_prefix = "title_step_"

    def __init__(self, Title: str, Subtitle: str, Articledetail: str, Datum: str, Autor: str, url: str, user: 'User', bild: str, bildnachweis : str):
        self.Title = Title
        self.Subtitle = Subtitle
        self.Articledetail = Articledetail
        self.Datum = Datum
        self.Autor = Autor
        self.url = url
        self.user = user
        self.bild = bild
        self.bildnachweis = bildnachweis

    @classmethod
    def from_web_form(cls, user: 'User', form: dict, images: dict) -> 'Article':
        url = slugify(form['title']).lower()
        title = form['title']
        title_image = [images['title_image']]
        binary_title_image = cls.convert_files_to_binary(title_image)
        step_titles = cls.extract_list(cls.__title_step_prefix, form)
        steps = cls.extract_list(cls.__step_prefix, form)
        images = cls.extract_list(cls.__file_step_prefix, images)
        binary_images = cls.convert_files_to_binary(images)
        description = form['description']

        return cls(url, title, description, binary_title_image.pop(), step_titles, steps, user, binary_images)

    @classmethod
    def from_dict(cls, d: dict) -> 'Article':
        pattern = r'[(http://)|\w]*?[\w]*\.[-/\w]*\.\w*[(/{1})]?[#-\./\w]*[(/{1,})]?'
        print(d)
        Title = d['Title']
        Subtitle = d["Subtitle"]
        Articledetail = d["Articledetail"]
        Datum = d["Datum"]
        Autor = d["Autor"]
        url = d["url"]
        if 'user' in d:
            user = d["user"]
        else:
            user = 'Testuser'
        bild = re.findall(pattern, d["bild"])[0]
        bildnachweis = d["bildnachweis"]

        print(bild)
        return cls(Title, Subtitle, Articledetail, Datum, Autor, url, user, bild, bildnachweis)

    @classmethod
    def from_web_form_or_default(cls, user: 'User', previous_version: 'Article', form: dict, images: dict) -> 'Article':
        url = slugify(form['title']).lower()
        title = form['title']
        title_image = [images['title_image']]
        description = form['description']
        if len(title_image) == 1 and title_image[0].filename == '':
            binary_title_image = [previous_version['title_image']]
        else:
            binary_title_image = cls.convert_files_to_binary(title_image)
        step_titles = cls.extract_list(cls.__title_step_prefix, form)
        steps = cls.extract_list(cls.__step_prefix, form)
        images = cls.extract_list(cls.__file_step_prefix, images)
        binary_images = cls.convert_files_to_binary(images)
        for idx, binary_image in enumerate(binary_images):
            if file_is_empty(binary_image):
                binary_images[idx] = previous_version['files'][idx]

        return cls(url, title, description, binary_title_image.pop(), step_titles, steps, user, binary_images)

    @classmethod
    def from_mongo_response(cls, response) -> 'Article':
        return cls.from_dict(response)

    @staticmethod
    def convert_files_to_binary(images) -> list:
        binary_files = []
        for image in images:
            if filename_is_empty(image.filename):
                binary_files.append(Binary(io.BytesIO().getvalue()))
            else:
                img = Image.open(image)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format)
                binary_files.append(Binary(img_byte_arr.getvalue()))
        return binary_files

    @staticmethod
    def extract_list(prefix: str, form: dict):
        steps = dict(filter(lambda item: item[0].startswith(prefix),
                            form.items()))
        return list(steps.values())

    def to_dict(self) -> dict:
        return {'url': self.url, "title_image": self.title_image, "title": self.title, "step_titles": self.step_titles,
                "steps": self.steps, "files": self.files, "description": self.description, "user": self.user.__dict__}