class IngestError(Exception):
    status = 400


class PayloadTooLarge(IngestError):
    status = 413


class UnsupportedMedia(IngestError):
    status = 415


class InvalidContent(IngestError):
    status = 422
