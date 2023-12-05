from django.core.exceptions import ValidationError

def validate_file_format(value):
    if not value.endswith('.mp4'):
        raise ValidationError("File must be in MP4 format.")


def validate_file_size(value):
    max_size = 100 * 1024 * 1024  # 100MB
    if value.size > max_size:
        raise ValidationError("File must be smaller than 100MB.")