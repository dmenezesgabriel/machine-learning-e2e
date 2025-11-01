import types

from src.logger import logger


def error_message_detail(
    message: Exception, error_detail: types.ModuleType
) -> str:
    _, _, exc_tb = error_detail.exc_info()  # type: ignore[attr-defined]
    error_message = (
        f"Error occurred in script: "
        f"{exc_tb.tb_frame.f_code.co_filename}, "
        f"line number: {exc_tb.tb_lineno}, "
        f"error message: {str(message)}"
    )
    return error_message


class CustomException(Exception):
    def __init__(self, message: Exception, error_detail: types.ModuleType):
        super().__init__(message)
        self.error_message = error_message_detail(message, error_detail)

    def __str__(self) -> str:
        return self.error_message


if __name__ == "__main__":
    import sys

    try:
        a = 1 / 0
    except Exception as error:
        logger.error(error)
        raise CustomException(error, sys)
