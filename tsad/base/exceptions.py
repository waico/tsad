class ArgumentNotFoundException(Exception):
    def __init__(self, message):
        super().__init__(message)


class UnsupportedTaskResultException(Exception):
    def __init__(self, message):
        super().__init__(message)
