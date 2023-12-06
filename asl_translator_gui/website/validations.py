from django.core.exceptions import ValidationError

valid_mp4 = ['.mp4']
valid_json = ['.json']


def validate_mp4(value):
    if not value.name.lower().endswith(tuple(valid_mp4)):
        raise ValidationError("File must be in MP4 format.")

def validate_file_size(value):
    max_size = 100 * 1024 * 1024  # 100MB
    if value.size > max_size:
        raise ValidationError("File must be smaller than 100MB.")
    

def validate_json(value):
    if not value.name.lower().endswith(tuple(valid_json)):
        raise ValidationError("File must be in JSON format.")

